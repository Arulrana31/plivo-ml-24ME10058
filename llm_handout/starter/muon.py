"""Muon optimizer (Keller Jordan et al., https://kellerjordan.github.io/posts/muon/,
reference impl https://github.com/KellerJordan/Muon/blob/master/muon.py) for >=2D
hidden-layer weight matrices. Pure PyTorch matmuls (Newton-Schulz iteration) — no
custom/compiled kernels, no external optimizer library.

Coefficients, momentum update, and the row/col LR-scaling factor below are copied
verbatim from the reference implementation (verified against the upstream source,
not reconstructed from memory) — only bfloat16 is dropped since we're CPU-only.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    """Muon for >=2D hidden weights only. Use AdamW for embeddings/head/1D params."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                         ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"],
                                      beta=group["momentum"], ns_steps=group["ns_steps"],
                                      nesterov=group["nesterov"])
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        return loss


def normuon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95,
                    ns_steps=5, nesterov=True):
    """RUNLOG Run 12 — NorMuon (Li et al., arXiv:2510.05491, reference impl
    github.com/zichongli5/NorMuon/blob/main/normuon.py, fetched verbatim via raw.githubusercontent
    to avoid transcription drift — an earlier WebFetch AI-summary of the same paper mis-stated the
    rescaling step as an absolute '0.2*lr*sqrt(mn)/frobenius-norm' formula; the real reference code
    instead rescales to exactly preserve the pre-normalization Frobenius norm, which is what's
    implemented below). Adds row-wise (per-neuron) second-moment normalization AFTER Muon's
    Newton-Schulz orthogonalization, to fix uneven per-neuron update norms that plain Muon produces.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)  # per-row (per-neuron) mean square
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add(1e-10)
    update = update * step_size
    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update = update * (vnorm / vnorm_new.add(1e-10))  # rescale to preserve the original update norm
    update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class NorMuon(torch.optim.Optimizer):
    """NorMuon for >=2D hidden weights only. Use AdamW for embeddings/head/1D params (same
    split as Muon)."""

    def __init__(self, params, lr=0.02, momentum=0.95, beta2=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, nesterov=nesterov,
                         ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                update = normuon_update(p.grad, state["momentum_buffer"],
                                         state["second_momentum_buffer"],
                                         beta=group["momentum"], beta2=group["beta2"],
                                         ns_steps=group["ns_steps"], nesterov=group["nesterov"])
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        return loss
