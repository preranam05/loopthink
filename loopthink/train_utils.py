"""Muon, WSD schedule, recurrence sampling, precision handling, time-based resumable checkpoints."""
from __future__ import annotations

import glob
import math
import os
import random
import time

import numpy as np
import torch


# ----------------------------------------------------------------------------- Muon
@torch.no_grad()
def newton_schulz5(G, steps: int = 5, eps: float = 1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Momentum + orthogonalised update for 2-D hidden weight matrices (Jordan et al. 2024).
    Runs Newton-Schulz in fp32 so it is safe on GPUs without bf16 (T4/P100)."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"].mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                u = newton_schulz5(g, group["ns_steps"]) * max(1.0, p.size(0) / p.size(1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u.to(p.dtype), alpha=-group["lr"])


def make_optimizers(model, lr_adam=3e-3, lr_muon=0.02, wd=0.0, use_muon=True):
    """Muon for 2-D matrices inside transformer blocks; AdamW for embeddings, norms, heads, scalars."""
    muon, adam = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_block = any(n.startswith(k) for k in ("prelude.", "core.", "coda.")) or n.startswith("adapter")
        (muon if (use_muon and p.dim() == 2 and in_block) else adam).append(p)
    opts = [torch.optim.AdamW(adam, lr=lr_adam, betas=(0.9, 0.95), weight_decay=wd, eps=1e-8)]
    if muon:
        opts.append(Muon(muon, lr=lr_muon, weight_decay=wd))
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
    return opts


def wsd_lr(step: int, total: int, warmup: int, decay_frac: float = 0.2, floor: float = 0.0):
    """Warmup-Stable-Decay multiplier."""
    if step < warmup:
        return (step + 1) / warmup
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return 1.0
    prog = (step - decay_start) / max(1, total - decay_start)
    return floor + (1 - floor) * (1 - prog)


def set_lr(opts, mult: float):
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * mult


# ----------------------------------------------------------------------------- recurrence sampling
def sample_r(mean: float = 6.0, sigma: float = 0.5, lo: int = 1, hi: int = 16, rng=random) -> int:
    """Log-normal-Poisson depth sampling (Huginn-style), clipped to [lo, hi]."""
    tau = rng.gauss(math.log(mean) - sigma ** 2 / 2, sigma)
    lam = math.exp(tau)
    # Poisson via numpy for simplicity
    r = int(np.random.poisson(max(lam - 1, 0.0))) + 1
    return max(lo, min(hi, r))


# ----------------------------------------------------------------------------- device / precision
def get_device() -> str:
    """cuda > mps (Apple Silicon GPU) > cpu. Override with LOOPTHINK_DEVICE=cpu|mps|cuda."""
    forced = os.environ.get("LOOPTHINK_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def resolve_precision(precision: str, device: str) -> str:
    if precision != "auto":
        return precision
    return ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "fp32"


def use_manual_attn(attn: str, precision: str, device: str) -> bool:
    """Manual fp32-softmax attention is only needed for fp16 stability; otherwise use fused SDPA (faster)."""
    if attn != "auto":
        return attn == "manual"
    return resolve_precision(precision, device) == "fp16"


def precision_setup(precision: str, device: str):
    """Returns (autocast_ctx_factory, GradScaler or None).
    auto: cuda -> bf16 if supported else fp16; mps -> fp32 (safest on Apple GPUs); cpu -> fp32."""
    off = (lambda: torch.autocast(device_type="cpu", enabled=False)), None
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "fp32"
    if precision == "fp32" or device == "cpu":
        return off
    if device == "mps":
        try:  # experimental: MPS autocast + scaler need a recent PyTorch
            dt = torch.float16 if precision == "fp16" else torch.bfloat16
            with torch.autocast("mps", dtype=dt):
                pass
            scaler = torch.amp.GradScaler("mps") if precision == "fp16" else None
            print(f"MPS mixed precision: {precision} (experimental; use --precision fp32 if you see NaNs)")
            return (lambda: torch.autocast("mps", dtype=dt)), scaler
        except Exception as e:
            print(f"MPS {precision} unavailable ({e}); falling back to fp32")
            return off
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 not supported on this GPU (T4/P100) - use --precision fp16")
        return (lambda: torch.autocast("cuda", dtype=torch.bfloat16)), None
    if precision == "fp16":
        return (lambda: torch.autocast("cuda", dtype=torch.float16)), torch.amp.GradScaler("cuda", init_scale=2 ** 14)
    raise ValueError(precision)


# ----------------------------------------------------------------------------- checkpoints
class Checkpointer:
    """Saves every `every_minutes` (Kaggle sessions die at 12h). Atomic writes; keeps the last `keep`."""

    def __init__(self, ckpt_dir: str, every_minutes: float = 30, keep: int = 2):
        self.dir, self.every, self.keep = ckpt_dir, every_minutes * 60, keep
        os.makedirs(ckpt_dir, exist_ok=True)
        self.last = time.time()

    def due(self) -> bool:
        return time.time() - self.last >= self.every

    def save(self, state: dict, step: int):
        path = os.path.join(self.dir, f"ckpt_{step:08d}.pt")
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        for old in sorted(glob.glob(os.path.join(self.dir, "ckpt_*.pt")))[: -self.keep]:
            os.remove(old)
        self.last = time.time()
        return path

    @staticmethod
    def latest(*dirs: str):
        found = []
        for d in dirs:
            if d:
                found += glob.glob(os.path.join(d, "**", "ckpt_*.pt"), recursive=True)
        return max(found, key=lambda p: int(os.path.basename(p)[5:13])) if found else None


def rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                mps=torch.mps.get_rng_state() if get_device() == "mps" else None)


def set_rng_state(st):
    random.setstate(st["py"]); np.random.set_state(st["np"]); torch.set_rng_state(st["torch"].cpu())
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
    if st.get("mps") is not None and get_device() == "mps":
        torch.mps.set_rng_state(st["mps"].cpu())


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
