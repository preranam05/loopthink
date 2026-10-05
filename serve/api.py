"""FastAPI service.   MODEL_DIR=runs/ft_official uvicorn serve.api:app --port 8000

POST /classify        {"text": "...", "max_loops": 16, "tau": 0.5}
POST /classify_batch  {"texts": [...], "max_loops": 16, "tau": 0.5, "mode": "compact"}
GET  /health, /intents
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from loopthink.engine import AdaptiveClassifier
from loopthink.train_utils import get_device

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
DEFAULT_TAU = float(os.environ.get("TAU", "0.5"))
DEFAULT_MAX_LOOPS = int(os.environ.get("MAX_LOOPS", "16"))
torch.set_num_threads(int(os.environ.get("THREADS", str(os.cpu_count() or 1))))

app = FastAPI(title="loopthink", version="0.1.0",
              description="Intent classification that decides how long to think and flags queries it doesn't recognise.")
_clf: AdaptiveClassifier | None = None


def clf() -> AdaptiveClassifier:
    global _clf
    if _clf is None:
        _clf = AdaptiveClassifier.from_run(MODEL_DIR, get_device())
    return _clf


class Req(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    max_loops: int = Field(DEFAULT_MAX_LOOPS, ge=1, le=64, description="compute budget")
    tau: float = Field(DEFAULT_TAU, ge=0.0, le=1.0, description="margin needed to stop early")


class BatchReq(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256)
    max_loops: int = Field(DEFAULT_MAX_LOOPS, ge=1, le=64)
    tau: float = Field(DEFAULT_TAU, ge=0.0, le=1.0)
    mode: str = Field("compact", pattern="^(compact|mask|fixed)$")


def _out(d):
    r = asdict(d)
    r["in_scope"] = d.converged   # "didn't converge" == "I don't know this"
    return r


@app.get("/health")
def health():
    c = clf()
    return {"status": "ok", "intents": len(c.intents), "looped": c.model.cfg.is_looped}


@app.get("/intents")
def intents():
    return clf().intents


@app.post("/classify")
def classify(req: Req):
    t0 = time.perf_counter()
    d = clf().classify([req.text], req.max_loops, req.tau)[0]
    return {**_out(d), "latency_ms": round((time.perf_counter() - t0) * 1e3, 2)}


@app.post("/classify_batch")
def classify_batch(req: BatchReq):
    t0 = time.perf_counter()
    ds = clf().classify(req.texts, req.max_loops, req.tau, mode=req.mode)
    if not ds:
        raise HTTPException(400, "empty")
    return {"results": [_out(d) for d in ds], "latency_ms": round((time.perf_counter() - t0) * 1e3, 2)}
