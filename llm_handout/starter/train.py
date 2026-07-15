"""Baseline trainer. It WORKS and it is MEDIOCRE ON PURPOSE. Your hour goes
into changing what it does — schedule, init, optimizer, architecture,
tokenizer — inside the hard caps.

HARD CAPS (checked at grading, violations = disqualified run):
  * max 2,000 optimizer steps in the run that produces your checkpoint
  * max 2,000,000 total parameters
  * training text: the provided train_corpus.txt only
  * pure PyTorch / numpy / stdlib; no pretrained anything

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod
from muon import Muon, NorMuon

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step, total_steps, warmup_steps, peak_lr, min_lr,
           schedule="cosine", decay_frac=0.1):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if schedule == "wsd":
        # stays at peak_lr, then decays to min_lr over the last decay_frac of steps
        decay_start = total_steps * (1 - decay_frac)
        if step < decay_start:
            return peak_lr
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        ratio = max(min_lr, 1e-12) / peak_lr
        return peak_lr * (ratio ** progress)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-3, help="peak LR")
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--tie_weights", type=int, default=None,
                     help="override Config.tie_weights (0/1); default keeps Config's value")
    ap.add_argument("--qk_norm", type=int, default=None,
                     help="override Config.qk_norm (0/1): RMSNorm(q)/RMSNorm(k) per head")
    ap.add_argument("--rope", type=int, default=None,
                     help="override Config.rope (0/1): rotary pos emb instead of learned pos_emb")
    ap.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    ap.add_argument("--decay_frac", type=float, default=0.1,
                     help="wsd schedule only: fraction of steps spent in the decay phase")
    ap.add_argument("--optimizer", choices=["adamw", "muon", "normuon"], default="adamw")
    ap.add_argument("--muon_lr", type=float, default=0.02,
                     help="peak LR for the Muon/NorMuon param group (hidden >=2D weights only); "
                          "--lr still governs the AdamW group (embeddings/head/1D params)")
    ap.add_argument("--normuon_beta2", type=float, default=0.95,
                     help="normuon only: EMA decay for the per-row (per-neuron) second-moment "
                          "statistic applied after Newton-Schulz orthogonalization")
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    if args.tie_weights is not None:
        cfg.tie_weights = bool(args.tie_weights)
    if args.qk_norm is not None:
        cfg.qk_norm = bool(args.qk_norm)
    if args.rope is not None:
        cfg.rope = bool(args.rope)
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    min_lr = args.lr * args.min_lr_ratio
    if args.optimizer in ("muon", "normuon"):
        # Muon/NorMuon on the hidden 2D weights, AdamW on everything else
        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            is_hidden_2d = p.ndim == 2 and not any(
                s in name for s in ("tok_emb", "pos_emb", "head"))
            (muon_params if is_hidden_2d else adamw_params).append(p)
        muon_min_lr = args.muon_lr * args.min_lr_ratio
        if args.optimizer == "muon":
            hidden_opt = Muon(muon_params, lr=args.muon_lr, weight_decay=args.weight_decay)
        else:
            hidden_opt = NorMuon(muon_params, lr=args.muon_lr, beta2=args.normuon_beta2,
                                  weight_decay=args.weight_decay)
        opts = [
            (hidden_opt, args.muon_lr, muon_min_lr),
            (torch.optim.AdamW(adamw_params, lr=args.lr, weight_decay=args.weight_decay), args.lr, min_lr),
        ]
    else:
        opts = [(torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay), args.lr, min_lr)]

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        lrs = []
        for opt, peak, mn in opts:
            lr_i = get_lr(step, args.steps, args.warmup_steps, peak, mn,
                          args.schedule, args.decay_frac)
            for g in opt.param_groups:
                g["lr"] = lr_i
            lrs.append(lr_i)
            opt.zero_grad(set_to_none=True)
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt, _, _ in opts:
            opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            lr_str = "/".join(f"{v:.2e}" for v in lrs)
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr_str}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")

    # every public config attribute is saved — if you add fields to Config,
    # they ride along automatically and evaluate.py rebuilds the same model
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
