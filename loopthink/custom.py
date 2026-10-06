"""Bring your own intents: read a CSV of labelled requests, check it, split it, and summarise results.

Pure Python (no torch), so the checks run anywhere and are easy to test. scripts/train_custom.py uses
these and then trains with the same code path as the benchmark runs.
"""
from __future__ import annotations

import csv
import collections
import random
import re

TEXT_COLS = ("text", "request", "query", "utterance", "message", "sentence")
LABEL_COLS = ("intent", "label", "category", "class")
MIN_PER_INTENT = 10


class DataError(ValueError):
    """A problem with the input file that the person supplying it can fix."""


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def _pick(header, names, what):
    low = [h.strip().lower() for h in header]
    for n in names:
        if n in low:
            return low.index(n)
    raise DataError(f"no {what} column found. Header is {header}; expected one of: {', '.join(names)}")


def read_examples(path):
    """Read (text, intent) pairs from a CSV with a header row. Returns (pairs, notes).

    Blank rows and exact duplicates are dropped and reported in notes. Intent names are lower-cased
    with spaces turned into underscores, so 'Reset Password' and 'reset_password' are one intent."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise DataError(f"{path} has no data rows")
    ti, li = _pick(rows[0], TEXT_COLS, "text"), _pick(rows[0], LABEL_COLS, "intent")
    pairs, seen, blank, dup, clash = [], {}, 0, 0, []
    for r in rows[1:]:
        if len(r) <= max(ti, li):
            blank += 1; continue
        text, label = _norm(r[ti]), re.sub(r"[^a-z0-9]+", "_", _norm(r[li]).lower()).strip("_")
        if not text or not label:
            blank += 1; continue
        key = text.lower()
        if key in seen:
            if seen[key] != label:
                clash.append((text, seen[key], label))
            dup += 1; continue
        seen[key] = label; pairs.append((text, label))
    notes = []
    if blank:
        notes.append(f"dropped {blank} rows with an empty text or intent")
    if dup:
        notes.append(f"dropped {dup} duplicate requests")
    if clash:
        t, a, b = clash[0]
        notes.append(f"{len(clash)} requests appear under two different intents (kept the first), e.g. "
                     f"\"{t}\" as both {a} and {b}. Conflicting labels cap the accuracy any model can reach")
    return pairs, notes


def read_texts(path):
    """Read unlabelled requests (one per row). A header row naming a text column is optional."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r and _norm(r[0])]
    if rows and rows[0][0].strip().lower() in TEXT_COLS:
        rows = rows[1:]
    out, seen = [], set()
    for r in rows:
        t = _norm(r[0])
        if t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    return out


def check(pairs, min_per_intent=MIN_PER_INTENT):
    """Raise DataError if the data cannot train a router; return warnings worth showing otherwise."""
    counts = collections.Counter(l for _, l in pairs)
    if len(counts) < 2:
        raise DataError(f"need at least 2 intents, found {len(counts)}")
    small = sorted((n, l) for l, n in counts.items() if n < min_per_intent)
    if small:
        raise DataError(f"every intent needs at least {min_per_intent} examples. Too few: "
                        + ", ".join(f"{l} ({n})" for n, l in small[:8]) + (" ..." if len(small) > 8 else ""))
    warn = []
    thin = sorted(l for l, n in counts.items() if n < 30)
    if thin:
        warn.append(f"{len(thin)} intents have fewer than 30 examples ({', '.join(thin[:5])}{' ...' if len(thin) > 5 else ''}); "
                    "expect their accuracy to be less stable")
    lo, hi = min(counts.values()), max(counts.values())
    if hi > 10 * lo:
        warn.append(f"intent sizes are very uneven ({lo} to {hi} examples); the small ones tend to lose to the large ones")
    return warn


def split_examples(pairs, val=0.15, test=0.15, seed=0):
    """Stratified split: each intent keeps the same share in train, val and test (at least 2 in each of
    val and test, so every intent is represented when the threshold is set and when accuracy is scored)."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for t, l in pairs:
        by[l].append(t)
    out = dict(train=[], val=[], test=[])
    for l in sorted(by):
        xs = by[l][:]; rng.shuffle(xs)
        nv, nt = max(2, round(len(xs) * val)), max(2, round(len(xs) * test))
        out["val"] += [[t, l] for t in xs[:nv]]
        out["test"] += [[t, l] for t in xs[nv:nv + nt]]
        out["train"] += [[t, l] for t in xs[nv + nt:]]
    for k in out:
        rng.shuffle(out[k])
    return out


def to_dataset(splits, unknown=None, seed=0):
    """The JSON layout the training code reads (same as CLINC150's data_full.json)."""
    unk = list(unknown or [])
    random.Random(seed).shuffle(unk)
    h = len(unk) // 2
    return dict(train=splits["train"], val=splits["val"], test=splits["test"],
                oos_train=[], oos_val=[[t, "oos"] for t in unk[:h]], oos_test=[[t, "oos"] for t in unk[h:]])


def bootstrap_ci(correct, n_boot=2000, seed=0):
    """95% bootstrap interval for an accuracy, given a list of 0/1 outcomes."""
    rng = random.Random(seed)
    n = len(correct)
    if n == 0:
        return (float("nan"), float("nan"))
    means = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def per_intent(y_true, y_pred, intents):
    """[(intent, n, accuracy)] sorted weakest first, and the most common confusions [(true, predicted, n)]."""
    tot, hit, conf = collections.Counter(), collections.Counter(), collections.Counter()
    for t, p in zip(y_true, y_pred):
        tot[t] += 1
        if t == p:
            hit[t] += 1
        else:
            conf[(t, p)] += 1
    rows = sorted(((intents[i], tot[i], hit[i] / tot[i]) for i in tot), key=lambda r: (r[2], r[0]))
    pairs = [(intents[t], intents[p], n) for (t, p), n in conf.most_common(8)]
    return rows, pairs


def routing_table(conf, correct, thresholds):
    """For each confidence threshold: share escalated, and accuracy of what the classifier keeps."""
    out = []
    for name, thr in thresholds:
        keep = [c for x, c in zip(conf, correct) if x > thr]
        esc = 1 - len(keep) / max(1, len(conf))
        out.append(dict(name=name, threshold=thr, escalated=esc, kept=len(keep),
                        kept_accuracy=(sum(keep) / len(keep)) if keep else float("nan")))
    return out
