"""Download the released router model and unpack it. Used by Dockerfile.router at build time.

  python scripts/fetch_model.py <url> <sha256> <target dir>

The archive is checked against the expected SHA-256 before anything is unpacked, so a changed or
corrupted download fails the build instead of shipping.
"""
import hashlib
import io
import os
import sys
import tarfile
import urllib.request

url, want, target = sys.argv[1], sys.argv[2].lower(), sys.argv[3]
with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "loopthink-build"}), timeout=300) as r:
    blob = r.read()
got = hashlib.sha256(blob).hexdigest()
if got != want:
    sys.exit(f"model checksum mismatch: expected {want}, got {got} ({len(blob)} bytes from {url})")
os.makedirs(target, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
    for m in tar.getmembers():          # refuse paths that would land outside the target directory
        if m.name.startswith(("/", "..")) or ".." in m.name.split("/") or not (m.isfile() or m.isdir()):
            sys.exit(f"unsafe entry in model archive: {m.name}")
    tar.extractall(target)
for f in ("router.json", "head.pt", "examples.json", "encoder/config.json"):
    if not os.path.exists(os.path.join(target, f)):
        sys.exit(f"model archive is missing {f}")
print(f"model ok: {len(blob) / 1e6:.1f} MB, sha256 {got[:12]}..., unpacked to {target}")
