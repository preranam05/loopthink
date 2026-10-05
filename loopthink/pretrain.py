"""Pretrain a looped (or dense baseline) LM on token shards.

  python -m loopthink.pretrain --data_dir /kaggle/input/fineweb-tok --out /kaggle/working/run \
      --preset looped-27m --tokens 1e9 --precision fp16 --resume

Resumable: checkpoints every --ckpt_minutes and exits cleanly after --max_hours, so a Kaggle
session can be restarted with --resume (point --resume_from at the previous run's output dataset).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from .train_utils import get_device

from .data import TokenShard
from .model import PRESETS, ModelConfig, build_model
from .train_utils import (Checkpointer, use_manual_attn, make_optimizers, precision_setup, rng_state, sample_r,
                          seed_all, set_lr, set_rng_state, wsd_lr)


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="dir with train.bin / val.bin (uint16)")
    p.add_argument("--out", required=True)
    p.add_argument("--preset", default="looped-27m", choices=list(PRESETS))
    p.add_argument("--vocab_size", type=int, default=16384)
    p.add_argument("--tokens", type=float, default=1e9)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr_adam", type=float, default=3e-3)
    p.add_argument("--lr_muon", type=float, default=0.02)
    p.add_argument("--no_muon", action="store_true")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--decay_frac", type=float, default=0.2)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--mean_r", type=float, default=6.0)
    p.add_argument("--max_r", type=int, default=16)
    p.add_argument("--fixed_r", type=int, default=0, help=">0 disables random r (ablation)")
    p.add_argument("--bptt", type=int, default=4, help="backprop through last k loops")
    p.add_argument("--no_injection", action="store_true", help="ablation: prelude output used only as s0, not re-injected")
    p.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--attn", default="auto", choices=["auto", "manual", "sdpa"],
                   help="auto: manual fp32-softmax attention for fp16 runs, fused SDPA otherwise")
    p.add_argument("--ckpt_minutes", type=float, default=30)
    p.add_argument("--max_hours", type=float, default=11.3)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_from", default="", help="extra dir to search for checkpoints (e.g. /kaggle/input/prev-run)")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=0, help="debug: stop after N steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model, val: TokenShard, args, autocast, device, loops=(1, 4, 8, 16, 32)):
    model.eval()
    out = {}
    loops = loops if model.cfg.is_looped else (1,)
    for r in loops:
        tot = 0.0
        for i in range(args.eval_batches):
            x, y = val.batch(args.batch_size, device)
            with autocast():
                tot += model.lm_loss(x, y, n_loops=r).item()
        out[f"val_loss_r{r}"] = tot / args.eval_batches
    model.train()
    return out


def main(argv=None):
    args = get_args(argv)
    device = get_device()
    seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cfg = ModelConfig(vocab_size=args.vocab_size, max_seq=max(args.seq_len, 64), inject=not args.no_injection,
                      attn_fp32=use_manual_attn(args.attn, args.precision, device), **{
        k: v for k, v in PRESETS[args.preset].items() if k != "max_seq"})
    model = build_model(cfg).to(device)
    print(f"params: {model.num_params()/1e6:.2f}M total, {model.num_params(True)/1e6:.2f}M non-embedding | "
          f"device={device} attention={'manual-fp32' if cfg.attn_fp32 else 'sdpa'}")
    opts = make_optimizers(model, args.lr_adam, args.lr_muon, use_muon=not args.no_muon)
    autocast, scaler = precision_setup(args.precision, device)

    train = TokenShard(os.path.join(args.data_dir, "train.bin"), args.seq_len, seed=args.seed)
    val = TokenShard(os.path.join(args.data_dir, "val.bin"), args.seq_len, seed=args.seed + 1)
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    total_steps = int(args.tokens // tokens_per_step)
    ck = Checkpointer(os.path.join(args.out, "checkpoints"), args.ckpt_minutes)

    step = 0
    if args.resume:
        path = Checkpointer.latest(ck.dir, args.resume_from)
        if path:
            st = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model"])
            for o, s in zip(opts, st["opts"]):
                o.load_state_dict(s)
            if scaler and st.get("scaler"):
                scaler.load_state_dict(st["scaler"])
            step = st["step"]
            train.set_rng_state(st["data_rng"]); set_rng_state(st["rng"])
            print(f"resumed from {path} at step {step}")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(dict(model=cfg.to_dict(), train=vars(args), total_steps=total_steps), f, indent=2)

    fwd = torch.compile(model.lm_loss) if args.compile else model.lm_loss
    log_f = open(os.path.join(args.out, "log.jsonl"), "a")
    t0, t_start, last_log = time.time(), time.time(), step
    tps_hist = []
    model.train()
    while step < total_steps:
        if args.max_steps and step >= args.max_steps:
            break
        if (time.time() - t_start) / 3600 > args.max_hours:
            print("time budget reached - checkpointing and exiting for resume")
            break
        set_lr(opts, wsd_lr(step, total_steps, args.warmup, args.decay_frac))
        r = args.fixed_r or sample_r(args.mean_r, hi=args.max_r)
        model.track_act = step % args.log_every == 0
        model.act_log = []
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x, y = train.batch(args.batch_size, device)
            with autocast():
                loss = fwd(x, y, n_loops=r, grad_last_k=args.bptt) / args.grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
            loss_acc += loss.item()
        if scaler:
            for o in opts:
                scaler.unscale_(o)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        for o in opts:
            scaler.step(o) if scaler else o.step()
        if scaler:
            scaler.update()
        for o in opts:
            o.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0 or step == 1:
            acts = [a.item() for a in model.act_log]
            rec = dict(step=step, loss=loss_acc, r=r, grad_norm=gn, lr_mult=wsd_lr(step, total_steps, args.warmup),
                       max_act_per_loop=[round(a, 2) for a in acts], max_act=max(acts) if acts else None,
                       loss_scale=scaler.get_scale() if scaler else None,
                       tok_per_s=tokens_per_step * (step - last_log) / (time.time() - t0) if step > 1 else None,
                       time=time.strftime("%Y-%m-%d %H:%M:%S"), **_mem_stats(device))
            t0, last_log = time.time(), step
            if rec["tok_per_s"]:
                tps_hist.append(rec["tok_per_s"])
                ref = sorted(tps_hist)[len(tps_hist) // 2]
                if len(tps_hist) >= 4 and rec["tok_per_s"] < 0.3 * ref:
                    print(f"WARNING: throughput fell to {rec['tok_per_s']:.0f} tok/s (median {ref:.0f}). "
                          "Check Activity Monitor for memory pressure/swap, that the Mac is plugged in, "
                          "and that Low Power Mode is off. Ctrl-C and re-run with --resume is safe.")
            print(json.dumps({k: v for k, v in rec.items() if k != "max_act_per_loop"}))
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            if acts and max(acts) > 3e4:
                print("WARNING: activations approaching fp16 max (65504)")
            if not math.isfinite(loss_acc):
                print("non-finite loss; scaler will skip, but check logs")
        if step % args.eval_every == 0:
            ev = evaluate(model, val, args, autocast, device)
            ev["step"] = step
            print("EVAL", json.dumps(ev)); log_f.write(json.dumps(ev) + "\n"); log_f.flush()
        if ck.due():
            ck.save(_state(model, opts, scaler, step, train, cfg), step)

    path = ck.save(_state(model, opts, scaler, step, train, cfg), step)
    if step >= total_steps:
        torch.save(dict(model=model.state_dict(), cfg=cfg.to_dict()), os.path.join(args.out, "final.pt"))
        ev = evaluate(model, val, args, autocast, device)
        print("FINAL", json.dumps(ev))
        with open(os.path.join(args.out, "final_eval.json"), "w") as f:
            json.dump(ev, f, indent=2)
    print("saved", path)


def _mem_stats(device):
    """GPU memory in GB, so a slowdown can be matched to memory growth / swapping."""
    try:
        if device == "mps":
            return dict(mem_alloc_gb=round(torch.mps.current_allocated_memory() / 1e9, 2),
                        mem_driver_gb=round(torch.mps.driver_allocated_memory() / 1e9, 2))
        if device == "cuda":
            return dict(mem_alloc_gb=round(torch.cuda.memory_allocated() / 1e9, 2),
                        mem_driver_gb=round(torch.cuda.memory_reserved() / 1e9, 2))
    except Exception:
        pass
    return {}


def _state(model, opts, scaler, step, train, cfg):
    return dict(model=model.state_dict(), opts=[o.state_dict() for o in opts],
                scaler=scaler.state_dict() if scaler else None, step=step,
                data_rng=train.rng_state(), rng=rng_state(), cfg=cfg.to_dict())


if __name__ == "__main__":
    main()
