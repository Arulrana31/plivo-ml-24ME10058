"""A small GPT in plain PyTorch. Yours to modify or replace entirely —
attention, SSM, whatever — as long as evaluate.py still works and the
parameter cap holds.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256      # byte-level tokenizer default
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = False   # <- one of many things worth questioning
    qk_norm = False       # RMSNorm(q), RMSNorm(k) per head before dot-product (Qwen3/Gemma3-style;
                           # see RUNLOG Run 7 — cites Henry et al. 2020 arXiv:2010.04245 for the
                           # original QKNorm and Anson & Aitchison arXiv:2511.21377 for context)
    rope = False           # rotary position embeddings (Su et al. arXiv:2104.09864) instead of a
                           # learned absolute pos_emb table


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rope = cfg.rope
        if self.rope:
            # theta_j = 1 / 10000^(2j/head_dim), j = 0..head_dim/2-1 (Su et al. eq. 15)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            t = torch.arange(cfg.block_size).float()
            freqs = torch.outer(t, inv_freq)  # (block_size, head_dim/2)
            self.register_buffer("rope_cos", freqs.cos(), persistent=False)
            self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def _apply_rope(self, x):
        # x: (B, n_head, T, head_dim). Split-half rotate_half formulation, equivalent to the
        # per-pair 2x2 rotation matrix [[cos,-sin],[sin,cos]] (Su et al. / standard RoPE impls).
        T, D = x.size(2), x.size(-1)
        x1, x2 = x[..., :D // 2], x[..., D // 2:]
        cos = self.rope_cos[:T][None, None, :, :]
        sin = self.rope_sin[:T][None, None, :, :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rope:
            q = self._apply_rope(q)
            k = self._apply_rope(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.use_pos_emb = not cfg.rope
        if self.use_pos_emb:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        # baseline init: plain normal, one std for everything. (GPT-2/muP-
        # style scaled init was tried and reverted — see RUNLOG Run 2: it
        # underperforms this flat init at 2000-step horizon across a 5x LR
        # sweep.)
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.use_pos_emb:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
