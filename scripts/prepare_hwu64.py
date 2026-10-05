"""Build data/hwu64.json in CLINC150's format so every script runs on it unchanged.

  python scripts/prepare_hwu64.py

HWU64 (Liu et al. 2019): ~11k queries to a home assistant, 64 intents across 21 scenarios
(alarm, calendar, music, IoT, weather, transport, cooking, ...), standard train/valid/test split
as used in DialoGLUE / Few-Shot-Intent-Detection.
Unknowns, two kinds:
  official setup -> CLINC150's out-of-scope queries (far-OOS)
  heldout setup  -> 16 of the 64 intents hidden from training (near-OOS). Run with --n_heldout 16.
"""
import json, os, ssl, sys, urllib.request

BASE = "https://raw.githubusercontent.com/jianguoz/Few-Shot-Intent-Detection/main/Datasets/HWU64/"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def fetch(path):
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    with urllib.request.urlopen(BASE + path, context=ctx, timeout=60) as r:
        return r.read().decode().splitlines()


def split(name):
    texts, labels = fetch(f"{name}/seq.in"), fetch(f"{name}/label")
    assert len(texts) == len(labels)
    return [[t.strip(), l.strip()] for t, l in zip(texts, labels) if t.strip()]


def main():
    sys.path.insert(0, ROOT)
    tr, va, te = split("train"), split("valid"), split("test")
    clinc = os.path.join(ROOT, "data", "data_full.json")
    if not os.path.exists(clinc):
        from loopthink.data import load_clinc
        load_clinc(os.path.join(ROOT, "data"))
    c = json.load(open(clinc))
    out = dict(train=tr, val=va, test=te, oos_val=c["oos_val"], oos_test=c["oos_test"],
               _source="HWU64 (Liu et al. 2019, DialoGLUE split) + CLINC150 out-of-scope queries as far-OOS unknowns")
    path = os.path.join(ROOT, "data", "hwu64.json")
    json.dump(out, open(path, "w"))
    print(f"wrote {path}: {len(tr)} train / {len(va)} val / {len(te)} test, "
          f"{len({l for _, l in tr})} intents, {len(c['oos_test'])} OOS test")


if __name__ == "__main__":
    main()
