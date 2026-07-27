# LLM Security Supervisor

LLM Security Supervisor is a FastAPI proxy that inspects an LLM's proposed tool calls before returning them to an agent. A local Ollama-hosted classifier evaluates each action and returns either `PASS` or `BLOCK`. Blocked actions are returned as HTTP `403 Forbidden` responses.

The project supports OpenAI-compatible `/v1/chat/completions` and Anthropic-compatible `/v1/messages` clients.

## How it works

```text
Agent client
    |
    v
FastAPI security proxy (src/middleware.py)
    | forwards request
    v
Upstream LLM provider
    | returns a proposed tool call
    v
Local supervisor model in Ollama
    |
    +-- PASS  -> return the upstream response
    `-- BLOCK -> return HTTP 403; the agent never receives the tool call
```

The supervisor receives the user preferences, extracted agent history, and the proposed action. It is trained to produce only `{"decision": "PASS"}` or `{"decision": "BLOCK"}`. If the supervisor service fails or produces an unparseable response, the proxy blocks the action.

## Repository layout

```text
.
|-- src/
|   `-- middleware.py              # FastAPI OpenAI/Anthropic proxy
|-- scripts/
|   |-- data_pipeline.py           # Builds balanced training/evaluation data
|   |-- train_supervisor.py        # Full Kaggle/Unsloth fine-tuning run
|   |-- train_supervisor_smoketest.py
|   |-- eval_supervisor.py         # Direct Ollama evaluation
|   |-- test_middleware.py         # Manual end-to-end proxy checks
|   |-- audit_data.py
|   `-- migrate_classifier_jsonl.py # Local-only legacy migration helper
|-- data/
|   |-- supervisor_train.jsonl
|   `-- supervisor_eval.jsonl
|-- models/                        # Local GGUF artifacts (git-ignored)
|-- reports/                       # Generated evaluation reports (git-ignored)
|-- Modelfile                      # Ollama model definition
|-- .env.example                   # Environment-variable template
`-- requirements.txt
```

### Responsibilities

| Path | What it does | When to use it |
| --- | --- | --- |
| `src/__init__.py` | Marks `src` as a Python package so `src.middleware` can be imported and launched with Uvicorn. | Leave it in place; it has no runtime configuration. |
| `src/middleware.py` | The production FastAPI application. It forwards OpenAI- and Anthropic-compatible requests, inspects returned tool calls, asks the supervisor for a decision, and returns the upstream response or HTTP 403. | Start the proxy with `python -m uvicorn src.middleware:app --port 8080`. |
| `scripts/data_pipeline.py` | Downloads/reads ToolSafe and OpenAgentSafety data, converts it to the supervisor prompt format, balances labels, and writes the train/evaluation JSONL files. | Run when rebuilding `data/` from source datasets. |
| `scripts/train_supervisor_smoketest.py` | Small Kaggle/Unsloth training run that validates the training, evaluation, and GGUF-export path before an expensive full run. | Run first when changing training code or dependencies. |
| `scripts/train_supervisor.py` | Full Kaggle/Unsloth fine-tuning workflow. It trains the Gemma supervisor and exports a Q4_K_M GGUF artifact. | Run only for a full model retrain. |
| `scripts/eval_supervisor.py` | Sends the evaluation set directly to the Ollama supervisor and calculates accuracy, precision, recall, F1, and the confusion matrix. | Use to evaluate a registered supervisor model; writes `reports/eval_results.json`. |
| `scripts/test_middleware.py` | Manual end-to-end checks of the running proxy, using fixed benign/malicious cases and dataset-driven cases. | Use after starting Ollama and the middleware. |
| `scripts/audit_data.py` | Read-only integrity and class-balance audit for the JSONL datasets. | Use after generating or modifying datasets. |
| `scripts/migrate_classifier_jsonl.py` | Local-only legacy helper that converts existing JSONL rows to the JSON-only `PASS`/`BLOCK` completion format. It is git-ignored. | Use only when migrating legacy data; it rewrites the data files. |
| `data/supervisor_train.jsonl` | Balanced examples used for fine-tuning. | Attach/upload for the full or smoke-test training run. |
| `data/supervisor_eval.jsonl` | Held-out balanced examples used by the direct evaluation and integration runner. | Do not use for training. |
| `models/` | Local GGUF model artifacts, including the production and smoke-test models. Git ignores this directory because files are large. | Place or download model artifacts here, then register the production artifact with Ollama. |
| `reports/` | Generated evaluation JSON reports. Git ignores this directory. | Inspect `eval_results.json` after a direct evaluation. |
| `docs/session.md` | Local development history, decisions, and session-by-session notes. It is git-ignored. | Reference for prior work and operational context in this working copy. |
| `docs/archive/` | Local preserved historical instructions and script copies; the directory is git-ignored. | Reference only; do not run these copies as the application. |
| `Modelfile` | Ollama definition that points at the production GGUF and supplies the required chat template. | Run `ollama create supervisor-proxy -f Modelfile` after replacing the production GGUF. |
| `.env.example` | Example environment variables for endpoints, model names, and mock mode. | Copy it to `.env` and fill in local values; never commit `.env`. |
| `requirements.txt` | Runtime and data-pipeline Python dependencies. | Install with `pip install -r requirements.txt`. |

## Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/) running locally for supervisor inference
- The [Claude Code CLI](https://code.claude.com/docs/en/overview) for the protected Claude Code workflow
- The production GGUF in `models/gemma-4-e2b-supervisor.Q4_K_M.gguf`

For dataset generation, install Git and use a network-enabled environment. Fine-tuning is designed for a Kaggle T4 GPU environment with Unsloth; see [Training](#training).

## Quick start

Create and activate a virtual environment, then install the runtime and data-pipeline dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set the values for your environment. Do not commit `.env`.

Register the bundled GGUF with Ollama:

```powershell
ollama create supervisor-proxy -f Modelfile
```

For a generic OpenAI- or Anthropic-compatible client, start the proxy from the
repository root:

```powershell
python -m uvicorn src.middleware:app --host 127.0.0.1 --port 8080
```

Point clients to the proxy:

```powershell
$env:OPENAI_BASE_URL = "http://localhost:8080/v1"
$env:ANTHROPIC_BASE_URL = "http://localhost:8080"
```

## Configuration

The proxy loads `.env` from the repository root. These variables are supported:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SUPERVISOR_URL` | `http://localhost:11434` | Ollama or OpenAI-compatible supervisor endpoint |
| `SUPERVISOR_MODEL` | `supervisor-proxy:latest` | Model used to classify tool calls |
| `UPSTREAM_OPENAI_URL` | `http://localhost:11434` | OpenAI-compatible upstream base URL |
| `UPSTREAM_ANTHROPIC_URL` | `https://api.anthropic.com` | Anthropic-compatible upstream base URL |
| `OLLAMA_API_KEY` | empty | Bearer token for remote Ollama-compatible services |
| `MOCK_UPSTREAM` | `False` | Enable deterministic mock upstream responses for local proxy tests |
| `UPSTREAM_TIMEOUT_SECONDS` | `120` | Maximum time the proxy waits for an upstream model response |
| `MIDDLEWARE_PORT` | `8080` | Port used by `python src/middleware.py` |

The Anthropic route supports streamed responses. To ensure a tool call is never
released before it is inspected, the proxy buffers one complete upstream stream,
checks every `tool_use` block, then forwards the unchanged stream on `PASS`.
This adds one-response latency, especially with cloud models.

## Use the proxy with Claude Code and Ollama

`ollama launch claude` connects Claude Code directly to Ollama on port 11434;
it does not pass through this proxy. Start the proxy and launch Claude Code with
port 8080 as its Anthropic base URL instead.

### First-time setup

From the repository root, with the virtual environment from Quick Start still
active, install the Python dependencies and authenticate the local Ollama daemon
to Ollama Cloud:

```powershell
python -m pip install -r requirements.txt
ollama signin
ollama run gemma4:31b-cloud "Reply with OK only."
ollama create supervisor-proxy -f Modelfile
```

If Ollama is not already running as its local service, start it in a separate
terminal with `ollama serve` before continuing.

Copy `.env.example` to `.env` and set these values:

```dotenv
SUPERVISOR_URL=http://127.0.0.1:11434
SUPERVISOR_MODEL=supervisor-proxy:latest
UPSTREAM_ANTHROPIC_URL=http://127.0.0.1:11434
UPSTREAM_OPENAI_URL=http://127.0.0.1:11434
MOCK_UPSTREAM=False
MIDDLEWARE_PORT=8080
```

### Run a protected Claude Code session

Open two PowerShell windows in the repository directory.

**Window 1 - start the security proxy and leave it running:**

```powershell
python -m uvicorn src.middleware:app --host 127.0.0.1 --port 8080
```

**Window 2 - launch the installed Claude Code CLI through the proxy:**

```powershell
$env:ANTHROPIC_AUTH_TOKEN = "ollama"
$env:ANTHROPIC_API_KEY = ""
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8080"
claude --model gemma4:31b-cloud
```

The variables in Window 2 apply to that PowerShell session only; repeat them in
each new terminal that starts Claude Code. In Window 1, each model-proposed
tool call must produce `Intercepted streamed tool call` followed by a
`Supervisor decision`. A `BLOCK` produces HTTP 403, so Claude Code does not
receive the tool call to execute. Do not use `--dangerously-skip-permissions`:
keep Claude Code's own permission prompts as a second control.

For a quick live check, ask Claude Code to run `git status`. The proxy terminal
should log the proposed action and its decision before Claude Code runs it.


## Run checks and evaluation

With Ollama and the middleware running:

```powershell
# Evaluate direct supervisor decisions against the complete evaluation set.
python scripts/eval_supervisor.py --model supervisor-proxy

# Exercise the proxy with manual and dataset-driven scenarios using Ollama Cloud.
$env:OPENAI_TEST_MODEL = "gemma4:31b-cloud"
python scripts/test_middleware.py
```

The direct evaluation writes `reports/eval_results.json`. The integration runner expects the proxy at `http://localhost:8080`; use `MOCK_UPSTREAM=True` for its mock-response path.

## Generate data

The data pipeline obtains ToolSafe and OpenAgentSafety records, balances the classes, and writes the resulting JSONL files into `data/`:

```powershell
python scripts/data_pipeline.py
python scripts/audit_data.py
```

If available in the local working copy, `scripts/migrate_classifier_jsonl.py`
updates legacy JSONL records to the JSON-only classifier completion format.

## Training

Fine-tuning is designed for a Kaggle notebook with Internet enabled and a T4 GPU accelerator. The current workflow produces a JSON-only binary classifier that returns `PASS` or `BLOCK`; it is not a general-purpose chat fine-tune.

### 1. Prepare the Kaggle inputs

First verify the local data before upload:

```powershell
python scripts/audit_data.py
```

Create a Kaggle Dataset containing these two files from `data/`:

- `supervisor_train.jsonl`
- `supervisor_eval.jsonl`

In Kaggle, create a new notebook, attach that dataset, select a **T4 GPU** accelerator, and enable **Internet**. A T4 x2 instance also works because the scripts intentionally restrict training to one visible GPU.

### 2. Install the notebook dependencies

Run this in the first Kaggle notebook cell:

```python
!pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git" -q
!pip install trl==0.19.1 gguf -q
```

Then copy the contents of the desired script from this repository into a notebook cell. The scripts locate attached JSONL files recursively beneath `/kaggle/input`, so no hard-coded Kaggle dataset slug is required.

### 3. Run the smoke test first

Run [`scripts/train_supervisor_smoketest.py`](scripts/train_supervisor_smoketest.py) before the full training script. Its purpose is to validate the complete training-to-GGUF-export path with a smaller sample, including LoRA saving, quantization, and cleanup.

Treat the smoke test as successful only when all of the following are true:

- The Unsloth startup banner reports `Num GPUs = 1`.
- The log reports more than zero supervised tokens in the first example.
- Training loss decreases without `NaN` or `inf` values.
- Evaluation completes without an out-of-memory error.
- The final smoke-test artifact appears in the Kaggle Output tab as `gemma-4-e2b-supervisor-smoketest.Q4_K_M.gguf`.

The script deliberately stages large GGUF conversion intermediates in `/tmp` because `/kaggle/working` has limited disk capacity. Do not change that staging behavior unless the available disk budget is understood.

### 4. Run the full fine-tuning job

After a successful smoke test, run [`scripts/train_supervisor.py`](scripts/train_supervisor.py). It trains on the complete `supervisor_train.jsonl` dataset and exports:

```text
gemma-4-e2b-supervisor.Q4_K_M.gguf
```

Fine-tuning and GGUF export are GPU- and disk-intensive. Keep the script's CUDA setup at the top of the file, especially `CUDA_VISIBLE_DEVICES=0`, and keep `packing=False`; both are required by the current Unsloth/Gemma configuration.

### 5. Bring the model back to the local project

Download the production GGUF from the Kaggle Output tab and put it in `models/` with the exact filename expected by the `Modelfile`:

```text
models/gemma-4-e2b-supervisor.Q4_K_M.gguf
```

Register and evaluate the new model locally:

```powershell
ollama create supervisor-proxy -f Modelfile
python scripts/eval_supervisor.py --model supervisor-proxy
```

The training scripts deliberately constrain visible CUDA devices and contain workarounds for Unsloth evaluation memory pressure. Keep those settings when reproducing the Kaggle workflow.

## Security notes

- The proxy decides only on tool calls generated by the upstream response. The agent client must use the proxy endpoint for enforcement.
- A failed supervisor request is fail-closed: it blocks the proposed action.
- `allow_origins=["*"]` is currently enabled for dashboard/client compatibility. Restrict CORS origins before exposing the proxy beyond a trusted local network.
- Classifier results are probabilistic. Review `reports/eval_results.json` and add representative adversarial cases before relying on the proxy in higher-risk environments.
