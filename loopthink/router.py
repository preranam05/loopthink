"""Decision router: a small classifier answers; low-confidence queries go to an LLM that picks from the
classifier's top-k shortlist or says out_of_scope. If the LLM fails, the classifier's answer is returned
and the failure is recorded, so the service degrades instead of erroring.

The escalation rule (-max_softmax >= threshold, threshold set on in-scope validation data) is the one
measured offline in scripts/cascade.py and scripts/llm_validate.py.
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time

import numpy as np

from .llm import OOS


def softmax(x):
    z = x - x.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def decide_from_probs(p, intents, threshold, text="", llm=None, allow_llm=True, top_k=5):
    """Pure decision logic for one query. p: probabilities over intents. llm: fn(text, candidates) -> dict."""
    order = np.argsort(-p)
    top = [dict(intent=intents[i], prob=round(float(p[i]), 4)) for i in order[:top_k]]
    conf = float(p[order[0]])
    escalate = -conf >= threshold
    out = dict(answer=intents[order[0]], confidence=round(conf, 4), top=top, escalate=bool(escalate),
               source="model", reason=None, llm_ms=None)
    if not escalate:
        out["reason"] = f"confident (max prob {conf:.2f} > {-threshold:.2f})"
        return out
    out["reason"] = f"low confidence (max prob {conf:.2f} <= {-threshold:.2f})"
    if not allow_llm or llm is None:
        out["source"] = "model_no_llm"
        out["reason"] += "; LLM disabled - returning model answer"
        return out
    try:
        r = llm(text, [t["intent"] for t in top])
        out.update(answer=r["label"], source="llm", llm_ms=round(r["ms"], 1))
        out["reason"] += f"; LLM chose {r['label']}" + (" (declined: not a supported request)" if r["label"] == OOS else "")
    except Exception as e:  # degrade gracefully
        out["source"] = "model_fallback"
        out["reason"] += f"; LLM failed ({type(e).__name__}: {str(e)[:80]}) - returning model answer"
    return out


class Router:
    def __init__(self, predict_probs, intents, threshold, llm=None, top_k=5, meta=None):
        self.predict_probs, self.intents, self.threshold = predict_probs, intents, threshold
        self.llm, self.top_k, self.meta = llm, top_k, meta or {}
        self.presets, self.info = {}, {}

    @classmethod
    def from_dir(cls, run_dir, val_rate=0.10, llm=None, top_k=5, device="cpu"):
        """Load an encoder saved by scripts/encoder_baseline.py --save_model."""
        import torch
        from transformers import AutoModel, AutoTokenizer
        cfg = json.load(open(os.path.join(run_dir, "router.json")))
        tok = AutoTokenizer.from_pretrained(os.path.join(run_dir, "encoder"))
        enc = AutoModel.from_pretrained(os.path.join(run_dir, "encoder")).to(device).eval()
        head = torch.nn.Linear(enc.config.hidden_size, len(cfg["intents"]))
        head.load_state_dict(torch.load(os.path.join(run_dir, "head.pt"), map_location="cpu"))
        head = head.to(device).eval()
        max_len = cfg.get("max_len", 48)

        @torch.no_grad()
        def predict_probs(texts):
            t = tok(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            h = enc(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).to(h.dtype)
            return softmax(head((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())

        thr = cfg["thresholds_by_val_rate"][str(val_rate)]
        r = cls(predict_probs, cfg["intents"], thr, llm, top_k,
                meta=dict(base_model=cfg["base_model"], setup=cfg["setup"], val_rate=val_rate, threshold=thr))
        r.presets = {k: round(-float(v), 3) for k, v in cfg["thresholds_by_val_rate"].items()}   # rate -> min confidence
        r.info = dict(custom=bool(cfg.get("custom")), dataset=cfg.get("dataset"), ui=cfg.get("ui"),
                      test_accuracy=(cfg.get("report") or {}).get("test_accuracy", cfg.get("test_acc")))
        return r

    def decide(self, text, allow_llm=True, min_confidence=None, top_k=None):
        """min_confidence: escalate at or below this max probability (default: the validated threshold)."""
        t0 = time.perf_counter()
        p = self.predict_probs([text])[0]
        model_ms = (time.perf_counter() - t0) * 1000
        thr = self.threshold if min_confidence is None else -float(min_confidence)
        out = decide_from_probs(p, self.intents, thr, text, self.llm, allow_llm, top_k or self.top_k)
        out["model_ms"] = round(model_ms, 2)
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out


class RequestLog:
    """Append-only JSONL request log plus rolling metrics over the most recent `window` requests."""

    def __init__(self, path=None, window=5000, store_text=True):
        self.path, self.store_text = path, store_text
        self.recent = collections.deque(maxlen=window)
        self.lock = threading.Lock()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            if os.path.exists(path):
                for line in open(path).readlines()[-window:]:
                    try:
                        self.recent.append(json.loads(line))
                    except Exception:
                        pass

    def add(self, text, out):
        rec = dict(ts=time.time(), answer=out["answer"], source=out["source"], escalate=out["escalate"],
                   confidence=out["confidence"], latency_ms=out["latency_ms"], llm_ms=out.get("llm_ms"))
        if self.store_text:
            rec["text"] = text
        with self.lock:
            self.recent.append(rec)
            if self.path:
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

    def metrics(self, last_minutes=None):
        with self.lock:
            rs = list(self.recent)
        if last_minutes:
            cut = time.time() - 60 * last_minutes
            rs = [r for r in rs if r["ts"] >= cut]
        n = len(rs)
        if not n:
            return dict(requests=0)
        lat = np.array([r["latency_ms"] for r in rs])
        llm = [r["llm_ms"] for r in rs if r.get("llm_ms") is not None]
        src = collections.Counter(r["source"] for r in rs)
        return dict(requests=n,
                    escalation_rate=round(float(np.mean([r["escalate"] for r in rs])), 4),
                    out_of_scope_rate=round(float(np.mean([r["answer"] == OOS for r in rs])), 4),
                    llm_failure_rate=round(src.get("model_fallback", 0) / max(1, sum(r["escalate"] for r in rs)), 4),
                    sources=dict(src),
                    latency_ms=dict(p50=round(float(np.percentile(lat, 50)), 1), p95=round(float(np.percentile(lat, 95)), 1)),
                    llm_ms_p50=round(float(np.median(llm)), 1) if llm else None,
                    top_answers=collections.Counter(r["answer"] for r in rs).most_common(10),
                    since=rs[0]["ts"])
