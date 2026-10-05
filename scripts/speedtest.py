"""How fast is this machine? Runs ~60 real training steps of the 27M looped model on random tokens
(no download needed) and prints tokens/sec plus ETAs for the kill test and the main run.

  caffeinate -i python scripts/speedtest.py                 # Mac (auto-detects MPS)
  python scripts/speedtest.py --batch_size 32 --grad_accum 2 # Kaggle T4
"""
import argparse, json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from loopthink import pretrain
from loopthink.train_utils import get_device

p = argparse.ArgumentParser()
p.add_argument("--batch_size", default="16")
p.add_argument("--grad_accum", default="4")
p.add_argument("--steps", type=int, default=60)
p.add_argument("--precision", default="auto")
p.add_argument("--attn", default="auto")
a = p.parse_args()
with tempfile.TemporaryDirectory() as d:
    rng = np.random.default_rng(0)
    for n in ("train.bin", "val.bin"):
        rng.integers(3, 16384, 3_000_000).astype(np.uint16).tofile(os.path.join(d, n))
    out = os.path.join(d, "run")
    pretrain.main(["--data_dir", d, "--out", out, "--preset", "looped-27m", "--batch_size", a.batch_size,
                   "--grad_accum", a.grad_accum, "--max_steps", str(a.steps), "--log_every", str(max(2, a.steps // 3)),
                   "--eval_every", "100000", "--tokens", "1e12", "--precision", a.precision, "--attn", a.attn, "--ckpt_minutes", "1e9"])
    recs = [json.loads(l) for l in open(os.path.join(out, "log.jsonl")) if '"tok_per_s"' in l]
    tps = [r["tok_per_s"] for r in recs if r.get("tok_per_s")]
    if not tps:
        sys.exit("not enough steps to measure; use --steps 30 or more")
    t = float(np.median(tps[1:] or tps))
print(f"\ndevice={get_device()} precision={a.precision} attn={a.attn}  ~{t:,.0f} tokens/sec (mean r=6, random depth)")
for name, tok in (("kill test (200M)", 2e8), ("main run (1B)", 1e9), ("5 ablations (5x200M)", 1e9)):
    print(f"  {name:22s} ~{tok / t / 3600:6.1f} hours")
print("Compare with the same script on Kaggle (T4) to decide where the 1B run goes.")
