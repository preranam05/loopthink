import json
import os

import numpy as np
import pytest

from loopthink.llm import OOS
from loopthink.router import RequestLog, Router, decide_from_probs, softmax

INTENTS = ["balance", "transfer", "freeze_account", "book_flight", "alarm", "timer"]


def probs(top, conf):
    p = np.full(len(INTENTS), (1 - conf) / (len(INTENTS) - 1)); p[INTENTS.index(top)] = conf
    return p


THR = -0.6   # escalate when max prob <= 0.6


def test_confident_query_stays_on_model():
    llm = lambda t, c: pytest.fail("LLM must not be called")
    out = decide_from_probs(probs("balance", 0.95), INTENTS, THR, "what's my balance", llm)
    assert out["answer"] == "balance" and out["source"] == "model" and not out["escalate"]


def test_low_confidence_goes_to_llm_with_shortlist():
    seen = {}
    def llm(text, cands):
        seen["cands"] = cands; return dict(label="transfer", ms=12.0)
    out = decide_from_probs(probs("balance", 0.4), INTENTS, THR, "move money", llm, top_k=3)
    assert out["escalate"] and out["source"] == "llm" and out["answer"] == "transfer"
    assert seen["cands"][0] == "balance" and len(seen["cands"]) == 3


def test_llm_can_decline():
    out = decide_from_probs(probs("alarm", 0.3), INTENTS, THR, "what's the meaning of life",
                            lambda t, c: dict(label=OOS, ms=5.0))
    assert out["answer"] == OOS and "declined" in out["reason"]


def test_llm_failure_degrades_to_model_answer():
    def llm(t, c): raise TimeoutError("ollama down")
    out = decide_from_probs(probs("timer", 0.35), INTENTS, THR, "x", llm)
    assert out["answer"] == "timer" and out["source"] == "model_fallback" and "TimeoutError" in out["reason"]


def test_llm_disabled():
    out = decide_from_probs(probs("timer", 0.35), INTENTS, THR, "x", lambda t, c: dict(label="alarm", ms=1), allow_llm=False)
    assert out["answer"] == "timer" and out["source"] == "model_no_llm"


def test_threshold_matches_offline_cascade_rule():
    """The router must escalate exactly the queries scripts/cascade.py counts as escalated."""
    rng = np.random.default_rng(0)
    val, test = softmax(rng.normal(0, 3, (500, 6))), softmax(rng.normal(0, 3, (300, 6)))
    thr = float(np.quantile(-val.max(-1), 1 - 0.1))          # as encoder_baseline --save_model / llm_validate
    offline = (-test.max(-1)) >= thr                          # as cascade.py deployable points
    online = np.array([decide_from_probs(p, INTENTS, thr)["escalate"] for p in test])
    assert (offline == online).all() and 0.03 < online.mean() < 0.3


def test_request_log_metrics(tmp_path):
    path = str(tmp_path / "req.jsonl")
    lg = RequestLog(path)
    for i in range(10):
        out = decide_from_probs(probs("balance", 0.9 if i < 7 else 0.3), INTENTS, THR, "q",
                                lambda t, c: dict(label=OOS, ms=40.0))
        out["latency_ms"] = 5.0 if i < 7 else 45.0
        lg.add("q", out)
    m = lg.metrics()
    assert m["requests"] == 10 and m["escalation_rate"] == 0.3 and m["out_of_scope_rate"] == 0.3
    assert m["sources"] == {"model": 7, "llm": 3}
    assert RequestLog(path).metrics()["requests"] == 10       # survives restart


def test_api_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import serve.router_api as api
    r = Router(lambda texts: np.stack([probs("freeze_account", 0.9 if "card" in t else 0.2) for t in texts]),
               INTENTS, THR, llm=lambda t, c: dict(label=OOS, ms=3.0), meta=dict(base_model="stub", threshold=THR))
    monkeypatch.setattr(api, "_router", r)
    monkeypatch.setattr(api, "_log", RequestLog(str(tmp_path / "log.jsonl")))
    c = TestClient(api.app)
    a = c.post("/decide", json={"text": "freeze my card"}).json()
    b = c.post("/decide", json={"text": "tell me a poem"}).json()
    assert a["answer"] == "freeze_account" and a["source"] == "model"
    assert b["answer"] == OOS and b["source"] == "llm"
    m = c.get("/metrics").json()
    assert m["requests"] == 2 and m["escalation_rate"] == 0.5
    assert c.get("/health").json()["status"] == "ok" and "Decision router" in c.get("/dashboard").text


def test_router_loads_saved_encoder(tmp_path):
    transformers = pytest.importorskip("transformers")
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]")); tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(["book a flight", "freeze my card", "set a timer"] * 10,
                            trainers.WordPieceTrainer(vocab_size=100, special_tokens=["[PAD]", "[UNK]"]))
    d = tmp_path / "run"; enc_dir = d / "encoder"
    transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]").save_pretrained(enc_dir)
    cfg = transformers.BertConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    transformers.BertModel(cfg).save_pretrained(enc_dir)
    torch.save(torch.nn.Linear(16, len(INTENTS)).state_dict(), d / "head.pt")
    json.dump(dict(base_model="tiny", setup="official", intents=INTENTS, max_len=16,
                   thresholds_by_val_rate={"0.1": -0.5}), open(d / "router.json", "w"))
    r = Router.from_dir(str(d), 0.1)
    out = r.decide("freeze my card", allow_llm=False)
    assert out["answer"] in INTENTS and out["source"] == "model_no_llm" or not out["escalate"]
    assert sum(t["prob"] for t in out["top"]) <= 1.0001 and out["model_ms"] > 0
