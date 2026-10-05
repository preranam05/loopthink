"""Looped (recurrent-depth) transformer, after Geiping et al. 2025 ("Huginn").

Layout:  embed -> prelude (run once) -> core (looped r times, input-injected) -> coda -> heads

Stability choices (needed for fp16 on T4/P100):
  * sandwich RMSNorm (before AND after) on every core sub-layer
  * QK-norm on attention; small init on residual output projections
  * RMSNorm and attention softmax computed in fp32 regardless of autocast dtype
  * optional per-loop max-activation logging (model.track_act = True)

A dense baseline is the same class with n_core = 0 (prelude-only stack).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d: int = 512
    n_heads: int = 8
    n_prelude: int = 2
    n_core: int = 2          # 0 => dense baseline (no loop)
    n_coda: int = 2
    max_seq: int = 1024
    state_init_std: float = 0.4   # std of the random initial state s0 (relative to RMSNorm'd unit scale)
    attn_fp32: bool = True
    label_loops: int = 8          # loops used when encoding intent label text
    inject: bool = True           # False = ablation: prelude output is only the initial state, never re-injected
    pad_id: int = 0

    @property
    def is_looped(self) -> bool:
        return self.n_core > 0

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------- layers
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_seq: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        f = torch.outer(torch.arange(max_seq).float(), inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, x):  # x: B,H,T,D
        T = x.shape[2]
        c, s = self.cos[:T], self.sin[:T]
        xf = x.float()
        h = xf.shape[-1] // 2
        x1, x2 = xf[..., :h], xf[..., h:]
        out = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
        return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.nh, self.hd = cfg.n_heads, cfg.d // cfg.n_heads
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)
        self.fp32 = cfg.attn_fp32

    def forward(self, x, rot, pad_mask=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k = rot(self.qn(q)), rot(self.kn(k))
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None, None]
        if pad_mask is not None:  # right-padded; True = real token
            mask = mask & pad_mask[:, None, None, :]
        if self.fp32:
            att = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.hd)
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            y = (att @ v.float()).to(x.dtype)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        h = int(8 * d / 3 / 64 + 0.999) * 64
        self.gate_up = nn.Linear(d, 2 * h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, sandwich: bool):
        super().__init__()
        self.sandwich = sandwich
        self.n1, self.n2 = RMSNorm(cfg.d), RMSNorm(cfg.d)
        self.attn, self.mlp = Attention(cfg), SwiGLU(cfg.d)
        if sandwich:
            self.n1b, self.n2b = RMSNorm(cfg.d), RMSNorm(cfg.d)

    def forward(self, x, rot, pm=None):
        if self.sandwich:
            x = self.n1b(x + self.attn(self.n1(x), rot, pm))
            x = self.n2b(x + self.mlp(self.n2(x)))
        else:
            x = x + self.attn(self.n1(x), rot, pm)
            x = x + self.mlp(self.n2(x))
        return x


# ----------------------------------------------------------------------------- model
class LoopedModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.prelude = nn.ModuleList([Block(cfg, False) for _ in range(cfg.n_prelude)])
        self.core = nn.ModuleList([Block(cfg, True) for _ in range(cfg.n_core)])
        self.coda = nn.ModuleList([Block(cfg, False) for _ in range(cfg.n_coda)])
        self.adapter = nn.Linear(2 * d, d, bias=False) if cfg.is_looped else None  # input injection
        self.norm_f = RMSNorm(d)
        self.rot = Rotary(d // cfg.n_heads, cfg.max_seq)
        # intent-decision head: pooled query vs. label-text key, cosine scored
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        # fixed s0 used in eval mode -> deterministic answers regardless of batch composition
        g = torch.Generator().manual_seed(1234)
        self.register_buffer("s0_fixed", torch.randn(1, cfg.max_seq, d, generator=g) * cfg.state_init_std)
        self.label_table = None   # [C, d] intent keys; initialised from label-text encodings (set_label_table)
        self.track_act = False
        self.act_log: list[torch.Tensor] = []
        self._init_weights()

    # -- init
    def _init_weights(self):
        eff_layers = self.cfg.n_prelude + self.cfg.n_coda + max(1, self.cfg.n_core * 6)
        small = 0.02 / math.sqrt(2 * eff_layers)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = small if (name.endswith("attn.o.weight") or name.endswith("mlp.down.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    def num_params(self, exclude_embed=False):
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if exclude_embed else n

    # -- pieces
    def embed_prelude(self, idx, pm=None):
        x = self.embed(idx)
        for b in self.prelude:
            x = b(x, self.rot, pm)
        return x

    def init_state(self, e):
        B, T, _ = e.shape
        if not self.cfg.inject:
            return e
        if self.training:
            return torch.randn_like(e) * self.cfg.state_init_std
        return self.s0_fixed[:, :T].expand(B, -1, -1).to(e.dtype)

    def core_step(self, s, e, pm=None):
        x = self.adapter(torch.cat([s, e if self.cfg.inject else torch.zeros_like(e)], dim=-1))
        for b in self.core:
            x = b(x, self.rot, pm)
        return x

    def coda_out(self, s, pm=None):
        x = s
        for b in self.coda:
            x = b(x, self.rot, pm)
        return self.norm_f(x)

    def run_loops(self, idx, pm=None, n_loops: int = 6, grad_last_k: int | None = None, keep_states=False):
        """Returns (final_state, prelude_output, list_of_states)."""
        e = self.embed_prelude(idx, pm)
        if not self.cfg.is_looped:
            return e, e, [e]
        s = self.init_state(e)
        states = []
        for i in range(n_loops):
            if grad_last_k is not None and i < n_loops - grad_last_k:
                with torch.no_grad():
                    s = self.core_step(s, e, pm)
            else:
                s = self.core_step(s, e, pm)
            if self.track_act:
                self.act_log.append(s.detach().abs().amax().float())
            if keep_states:
                states.append(s)
        return s, e, states

    # -- language modelling
    def lm_loss(self, idx, targets, n_loops=6, grad_last_k=None):
        s, _, _ = self.run_loops(idx, None, n_loops, grad_last_k)
        h = self.coda_out(s)
        logits = F.linear(h, self.embed.weight)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

    # -- intent decisions
    @staticmethod
    def pool(h, pm):
        m = pm.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def query_vec(self, s, pm):
        h = self.coda_out(s, pm)
        return F.normalize(self.q_proj(self.pool(h, pm)).float(), dim=-1)

    def encode_labels(self, lab_idx, lab_pm, n_loops=None):
        s, _, _ = self.run_loops(lab_idx, lab_pm, n_loops or self.cfg.label_loops)
        h = self.coda_out(s, lab_pm)
        return F.normalize(self.k_proj(self.pool(h, lab_pm)).float(), dim=-1)

    def set_label_table(self, emb: torch.Tensor):
        """Cache label-text embeddings as a trainable intent table. Encoding labels with the same
        encoder on every step collapses (all keys identical) when training from scratch, so we
        compute them once from the (pretrained) model and fine-tune them as parameters."""
        self.label_table = nn.Parameter(emb.detach().float().clone())

    def intent_keys(self):
        return F.normalize(self.label_table.float(), dim=-1)

    def intent_logits(self, q, label_emb):
        return self.logit_scale.exp().clamp(max=100.0) * (q @ label_emb.T)

    def classify_trajectory(self, idx, pm, n_loops, label_emb, score_loops=None, grad_last_k=None):
        """Intent logits at each loop in score_loops (1-indexed; default: every loop).
        Returns list of (loop, logits[B,C], pooled_state[B,d])."""
        e = self.embed_prelude(idx, pm)
        if not self.cfg.is_looped:
            return [(1, self.intent_logits(self.query_vec(e, pm), label_emb), self.pool(e, pm).float())]
        score_loops = set(score_loops or range(1, n_loops + 1))
        s = self.init_state(e)
        out = []
        for i in range(1, n_loops + 1):
            if grad_last_k is not None and i <= n_loops - grad_last_k:
                with torch.no_grad():
                    s = self.core_step(s, e, pm)
            else:
                s = self.core_step(s, e, pm)
            if self.track_act:
                self.act_log.append(s.detach().abs().amax().float())
            if i in score_loops:
                out.append((i, self.intent_logits(self.query_vec(s, pm), label_emb), self.pool(s, pm).float()))
        return out


def build_model(cfg: ModelConfig | dict) -> LoopedModel:
    if isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    return LoopedModel(cfg)


PRESETS = {
    # the real thing (~27M incl. 8.4M tied embedding)
    "looped-27m": dict(d=512, n_heads=8, n_prelude=2, n_core=2, n_coda=2),
    "dense-6":    dict(d=512, n_heads=8, n_prelude=6, n_core=0, n_coda=0),   # param-matched
    "dense-16":   dict(d=512, n_heads=8, n_prelude=16, n_core=0, n_coda=0),  # FLOP-matched at r~6
    # tiny CPU config for smoke tests / the sandbox kill test
    "tiny":       dict(d=128, n_heads=4, n_prelude=1, n_core=1, n_coda=1, max_seq=64),
    "tiny-dense": dict(d=128, n_heads=4, n_prelude=3, n_core=0, n_coda=0, max_seq=64),
}
