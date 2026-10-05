"""Latency / throughput / cost benchmark.

  python -m loopthink.bench --runs runs/ft_official runs/dense6 runs/dense16 --taus 0.3 0.5 0.7 \
      --gpu_hour_usd 0.35

Reports, per model and setting: p50/p95 single-query latency, batched throughput (q/s) for
fixed-r vs adaptive (mask) vs adaptive (compact), and cost per 1,000 queries at a given $/hour.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from .train_utils import get_device, sync

from .data import load_clinc
from .engine import AdaptiveClassifier


DEVICE = get_device()


def timeit(fn, n):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); sync(DEVICE); ts.append(time.perf_counter() - t)
    return np.array(ts)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--clinc", default="data")
    p.add_argument("--taus", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    p.add_argument("--max_loops", type=int, default=16)
    p.add_argument("--fixed_r", type=int, default=0, help="fixed-r baseline depth (default: training mean r ~6)")
    p.add_argument("--single_n", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--batch_n", type=int, default=10)
    p.add_argument("--oos_frac", type=float, default=0.1, help="share of out-of-scope traffic in the mix")
    p.add_argument("--gpu_hour_usd", type=float, default=0.35)
    p.add_argument("--out", default="bench_results.json")
    a = p.parse_args(argv)
    device = get_device()
    raw = load_clinc(a.clinc)
    rng = random.Random(0)
    ins, oos = [t for t, _ in raw["test"]], [t for t, _ in raw["oos_test"]]
    def sample(n):
        return [rng.choice(oos) if rng.random() < a.oos_frac else rng.choice(ins) for _ in range(n)]
    single_q, batches = sample(a.single_n), [sample(a.batch_size) for _ in range(a.batch_n)]
    fixed_r = a.fixed_r or 6
    rows = []
    for run in a.runs:
        c = AdaptiveClassifier.from_run(run, device)
        looped = c.model.cfg.is_looped
        settings = [("fixed", fixed_r, 0.0)] + ([("adaptive", a.max_loops, t) for t in a.taus] if looped else [])
        for kind, ml, tau in settings:
            modes = ["fixed"] if kind == "fixed" else ["mask", "compact"]
            for mode in modes:
                c.classify(single_q[:4], ml, tau, mode=mode)  # warmup
                it = iter(single_q)
                lat = timeit(lambda: c.classify([next(it)], ml, tau, mode=mode), a.single_n)
                bt = iter(batches)
                bl = timeit(lambda: c.classify(next(bt), ml, tau, mode=mode), a.batch_n)
                loops = np.mean([d.loops_used for b in batches[:3] for d in c.classify(b, ml, tau, mode=mode)])
                qps = a.batch_size / bl.mean()
                rows.append(dict(run=os.path.basename(run.rstrip("/")), looped=looped, mode=mode,
                                 max_loops=ml, tau=tau, avg_loops=float(loops),
                                 p50_ms=float(np.percentile(lat, 50) * 1e3), p95_ms=float(np.percentile(lat, 95) * 1e3),
                                 batch_qps=float(qps),
                                 usd_per_1k=float(a.gpu_hour_usd / 3600 * 1000 / qps)))
                print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()}))
    with open(a.out, "w") as f:
        json.dump(dict(device=device, gpu_hour_usd=a.gpu_hour_usd, rows=rows), f, indent=2)
    print("| run | mode | τ | avg loops | p50 ms | p95 ms | batch q/s | $/1k |\n|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run']} | {r['mode']} | {r['tau']} | {r['avg_loops']:.2f} | {r['p50_ms']:.1f} | "
              f"{r['p95_ms']:.1f} | {r['batch_qps']:.0f} | {r['usd_per_1k']:.5f} |")


if __name__ == "__main__":
    main()
