"""Phase 0 precision test: 1,000 steps in fp16 with loss scaling at r=16; confirm nothing overflows.

  python scripts/precision_test.py --data_dir /kaggle/input/fineweb-tok --out /kaggle/working/prec
PASS if: no non-finite loss after the scaler settles, max activation stays well under 65504,
and the loss scale doesn't collapse (repeated overflow -> scale keeps halving).
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from loopthink import pretrain

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--r", type=int, default=16)
p.add_argument("--precision", default="auto")
p.add_argument("--preset", default="looped-27m")
p.add_argument("--vocab_size", type=int, default=16384)
a, rest = p.parse_known_args()
pretrain.main(["--data_dir", a.data_dir, "--out", a.out, "--preset", a.preset, "--vocab_size", str(a.vocab_size),
               "--precision", a.precision, "--fixed_r", str(a.r), "--bptt", str(a.r), "--max_steps", str(a.steps),
               "--log_every", "10", "--eval_every", "100000", "--tokens", "1e12", "--warmup", "100", "--ckpt_minutes", "1e9"] + rest)
recs = [json.loads(l) for l in open(os.path.join(a.out, "log.jsonl")) if '"loss"' in l]
acts = [r["max_act"] for r in recs if r.get("max_act") is not None]
scales = [r["loss_scale"] for r in recs if r.get("loss_scale")]
bad = [r["step"] for r in recs[5:] if not math.isfinite(r["loss"])]
verdict = dict(steps=recs[-1]["step"], max_activation=max(acts), final_loss=recs[-1]["loss"],
               min_loss_scale=min(scales) if scales else None, final_loss_scale=scales[-1] if scales else None,
               nonfinite_logged_steps=bad)
verdict["PASS"] = (not bad) and max(acts) < 1e4 and (not scales or min(scales) >= 8)
print(json.dumps(verdict, indent=2))
json.dump(verdict, open(os.path.join(a.out, "precision_verdict.json"), "w"), indent=2)
