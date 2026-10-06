# Stand the router up on your own intents

The benchmark numbers in this repository say how the design behaves on public data. This guide gets you the
same router, console and report on **your** list of intents, so you can judge it on your own requests.

You need Python 3.10+, the repository's requirements, and a spreadsheet of example requests. Training takes a
few minutes on a laptop CPU for a few hundred rows.

## 1. Prepare a CSV

Two columns with a header row: the request text and the intent it belongs to.

```csv
text,intent
i forgot my password,reset_password
vpn won't connect from home,vpn_issue
```

- Column names can be `text`/`request`/`query`/`message` and `intent`/`label`/`category`.
- At least 10 examples per intent, or the script stops and says which intents are short. 30 or more per intent
  gives steadier results.
- Use real requests if you have them. Wording invented by one person is easier than real traffic and will
  flatter the scores.
- Optional but recommended: a second CSV of requests you do **not** handle, one per row. Without it the report
  cannot say how well out-of-scope requests are caught.

## 2. Train and get a report

```bash
python scripts/train_custom.py --data my_requests.csv --unknown my_unknown.csv --out runs/mine --name "My assistant"
```

Try it first on the bundled made-up example:

```bash
python scripts/train_custom.py --data examples/helpdesk/requests.csv --unknown examples/helpdesk/unknown.csv --out runs/helpdesk --name "IT helpdesk"
```

The script checks the file (blank rows, duplicates, the same request under two intents, intents that are too
small), holds out 15% of each intent for testing and 15% for setting the threshold, fine-tunes the classifier,
and writes `runs/mine/REPORT.md`:

- accuracy of the small model alone on the held-out requests, with a 95% interval
- for each threshold setting: the share of requests sent to the LLM and the accuracy of what the small model keeps
- if you supplied unknown requests: how many each setting would send to the LLM, and how many would get a
  confident wrong answer
- the weakest intents and the most common mix-ups, which usually point at labels that overlap

## 3. Look at it in the console

```bash
ROUTER_DIR=runs/mine LLM_MODEL= uvicorn serve.router_api:app --port 8000     # small model only
ROUTER_DIR=runs/mine uvicorn serve.router_api:app --port 8000                # with a local Ollama LLM
```

Open http://localhost:8000. The example chips and the "Play sample requests" button now use held-out requests
from your file. To use a hosted LLM, set `LLM_PROVIDER` and `LLM_MODEL` and put the key in the provider's
environment variable (for example `GROQ_API_KEY`).

To measure the LLM stage on your data with real calls:

```bash
python scripts/llm_validate.py --run runs/mine --model qwen2.5:7b --hint_topk 5 --examples 3
```

## 4. Ship it

```bash
python scripts/package_model.py --run runs/mine
```

This writes `runs/mine/router-model.tar.gz` and prints its SHA-256. Publish the archive somewhere the build can
download it (a release file works), then build the same container the pipeline builds:

```bash
docker build -f Dockerfile.router --build-arg MODEL_URL=<download url> --build-arg MODEL_SHA256=<checksum> -t my-router .
docker run -p 8000:8000 -e LLM_MODEL= my-router
```

`.github/workflows/ci.yml` shows the rest: start the container, send it a request, publish the image, roll it
out. Change the request in the smoke test to one of your own.

## What to expect, and what not to

- A few hundred clean examples usually give a usable first router. They do not tell you how it behaves on real
  traffic; the held-out test set is small and comes from the same file.
- Requests close to one of your intents but not actually supported are the hard case. Expect a share of them to
  get a confident wrong answer from the small model. The report measures that share when you supply unknowns.
- Two intents that are often mixed up usually need clearer definitions or more examples, not a bigger model.
- The cost and latency comparison in the console uses per-call figures measured on the CLINC150 benchmark, not
  your provider's billing.
- Request text can be personal data. The service only stores it when `STORE_TEXT=1`; decide that deliberately.
