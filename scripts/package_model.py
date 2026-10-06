"""Pack a trained router into one archive for release, and print the checksum the container build needs.

  python scripts/package_model.py --run runs/mine

Publish the .tar.gz as a release file, then build with:
  docker build -f Dockerfile.router --build-arg MODEL_URL=<download url> --build-arg MODEL_SHA256=<checksum> -t my-router .
"""
import argparse
import hashlib
import os
import sys
import tarfile

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="a folder written by train_custom.py or encoder_baseline.py --save_model")
p.add_argument("--out", default="", help="archive path (default: <run>/router-model.tar.gz)")
a = p.parse_args()
parts = ["router.json", "head.pt", "examples.json", "encoder"]
missing = [f for f in parts if not os.path.exists(os.path.join(a.run, f))]
if missing:
    sys.exit(f"{a.run} is missing {', '.join(missing)}; train with --save_model first")
out = a.out or os.path.join(a.run, "router-model.tar.gz")
with tarfile.open(out, "w:gz") as tar:
    for f in parts:
        tar.add(os.path.join(a.run, f), arcname=f)
h = hashlib.sha256(open(out, "rb").read()).hexdigest()
print(f"{out}  ({os.path.getsize(out) / 1e6:.1f} MB)\nMODEL_SHA256={h}")
