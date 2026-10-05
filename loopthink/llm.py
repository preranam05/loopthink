"""LLM fallback: prompt, label-constrained output and the Ollama call. Shared by the router and
scripts/llm_validate.py so the served system is exactly the one that was measured."""
from __future__ import annotations

import json
import time
import ssl
import urllib.error
import urllib.request

OOS = "out_of_scope"

try:  # use certifi's CA bundle so HTTPS works on python.org Python on macOS without extra setup
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = None
OOS_RULE = (f"Use {OOS} when the request is not clearly one of the listed tasks - e.g. questions about unrelated "
            "topics or services this assistant does not offer. Do not force a loosely related label; "
            "generic labels such as what_can_i_ask_you are only for questions about the assistant itself.")


def build_system_prompt(all_intents):
    lines = "\n".join(list(all_intents) + [OOS])
    return ("You are the intent router of a general-purpose virtual assistant.\n"
            "Classify the user's message into exactly one label from this list:\n"
            f"{lines}\n\n{OOS_RULE}\n"
            'Reply with JSON only: {"intent": "<label>"}')


def build_shortlist_prompt(with_examples=False):
    base = ("You are the intent router of a customer assistant. A fast classifier proposed candidate labels for the "
            "user's message, most likely first. Pick the single correct label from the candidates.\n")
    if with_examples:
        base += ("Each candidate is followed by real example messages that belong to it. Choose a candidate only if "
                 "the user's message asks for the same thing as its examples.\n")
    return base + f"{OOS_RULE}\n" + 'Reply with JSON only: {"intent": "<label>"}'


def shortlist_message(text, candidates, examples=None):
    """examples: optional {label: [example messages]} shown under each candidate."""
    if not examples:
        return f"Message: {text}\nCandidates: {', '.join(candidates)}"
    lines = []
    for c in candidates:
        ex = examples.get(c)
        lines.append(f"- {c}" + (": " + " | ".join(f'"{e}"' for e in ex) if ex else
                                 (": none of the candidates fits" if c == OOS else "")))
    return f"Message: {text}\nCandidates:\n" + "\n".join(lines)


class ExampleIndex:
    """Nearest training examples per label (TF-IDF cosine). Training split only, so nothing leaks from test."""

    def __init__(self, train_pairs, intents):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.texts = [t for t, _ in train_pairs]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.X = self.vec.fit_transform(self.texts)          # rows are L2-normalised
        self.rows = {}
        for r, (_, y) in enumerate(train_pairs):
            self.rows.setdefault(intents[y], []).append(r)

    def lookup(self, text, labels, k):
        q = self.vec.transform([text])
        out = {}
        for lab in labels:
            rows = self.rows.get(lab)
            if not rows:
                continue
            sims = (self.X[rows] @ q.T).toarray().ravel()
            best, seen = [], set()
            for j in sims.argsort()[::-1]:
                t = self.texts[rows[j]]
                if t.lower() not in seen and t.lower() != text.lower():
                    seen.add(t.lower()); best.append(t)
                if len(best) == k:
                    break
            out[lab] = best
        return out


def schema(labels):
    return {"type": "object", "properties": {"intent": {"type": "string", "enum": list(labels)}}, "required": ["intent"]}


def ask_ollama(host, model, system, text, timeout, fmt="json"):
    body = json.dumps(dict(model=model, stream=False, format=fmt, keep_alive="30m",
                           options=dict(temperature=0, num_predict=24),
                           messages=[dict(role="system", content=system), dict(role="user", content=text)])).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    ms = (time.time() - t0) * 1000
    raw = out.get("message", {}).get("content", "")
    try:
        label = str(json.loads(raw).get("intent", "")).strip().lower().replace(" ", "_")
    except Exception:
        label = ""
    return dict(label=label, raw=raw, ms=ms, prompt_tokens=out.get("prompt_eval_count"),
                output_tokens=out.get("eval_count"))


def shortlist_llm(host, model, timeout=10.0, provider="ollama", api_key=None):
    """Returns fn(text, candidates) -> dict(label, ms, ...) using the measured shortlist setup."""
    system = build_shortlist_prompt()
    ask = make_asker(provider, host, api_key)

    def call(text, candidates):
        cands = list(candidates) + ([OOS] if OOS not in candidates else [])
        r = ask(model, system, shortlist_message(text, cands), timeout, cands)
        if r["label"] not in cands:
            raise ValueError(f"LLM returned invalid label {r['label']!r}")
        return r
    return call


# ----------------------------------------------------------------------------- hosted LLMs (OpenAI-compatible)
PROVIDERS = {  # base URLs of OpenAI-compatible chat-completions APIs
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}
KEY_ENV = {"openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY", "gemini": "GEMINI_API_KEY"}


def ask_openai(base_url, api_key, model, system, text, timeout, labels=None, _strict=[True]):
    """Chat-completions call constrained to `labels` via a JSON schema (falls back to plain JSON mode
    for providers that reject strict schemas). Token counts come from the API's own usage field."""
    def post(response_format):
        body = dict(model=model, temperature=0, max_tokens=24, response_format=response_format,
                    messages=[dict(role="system", content=system), dict(role="user", content=text)])
        if "gpt-oss" in model or "qwen3" in model:   # reasoning models: their thinking counts toward max_tokens
            body.update(max_tokens=1024, reasoning_effort="low")
        req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}", "User-Agent": "loopthink/0.2"})
        for attempt in range(40):   # free tiers rate-limit: wait as long as the API asks, then retry
            try:
                t_call = time.time()
                with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
                    out = json.loads(r.read())
                out["_ms"] = (time.time() - t_call) * 1000   # time of the successful call only, not the waiting
                return out
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 39:
                    if e.code not in (400, 429):
                        e.msg = f"{e.msg}: {e.read().decode(errors='ignore')[:200]}"
                    raise
                wait = e.headers.get("retry-after")
                time.sleep(min(90.0, float(wait) + 1 if wait else 5.0 + 2 * attempt))
    t0 = time.time()
    if labels and _strict[0]:
        try:
            out = post({"type": "json_schema", "json_schema": {"name": "intent", "strict": True,
                        "schema": {**schema(labels), "additionalProperties": False}}})
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            _strict[0] = False  # provider doesn't support strict schemas; remember and use JSON mode
            out = post({"type": "json_object"})
    else:
        out = post({"type": "json_object"})
    ms = out.get("_ms", (time.time() - t0) * 1000)
    raw = out["choices"][0]["message"].get("content") or ""
    try:
        label = str(json.loads(raw).get("intent", "")).strip().lower().replace(" ", "_")
    except Exception:
        label = ""
    u = out.get("usage", {})
    return dict(label=label, raw=raw, ms=ms, prompt_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"))


def make_asker(provider="ollama", host=None, api_key=None):
    """fn(model, system, text, timeout, labels) -> dict, for any supported provider."""
    if provider == "ollama":
        host = host or "http://localhost:11434"
        return lambda model, system, text, timeout, labels=None: ask_ollama(
            host, model, system, text, timeout, schema(labels) if labels else "json")
    import os
    base = host or PROVIDERS[provider]
    key = api_key or os.environ.get(KEY_ENV.get(provider, "LLM_API_KEY")) or os.environ.get("LLM_API_KEY")
    if not key:
        raise SystemExit(f"Set {KEY_ENV.get(provider, 'LLM_API_KEY')} for provider {provider}")
    return lambda model, system, text, timeout, labels=None: ask_openai(base, key, model, system, text, timeout, labels)
