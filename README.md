# arbiter

Serve typed-decision models — Laya or your own — on your GPU or your Mac, with a Jev-compatible
API.

**This fork adds one endpoint:** `POST /v1/chat/completions`, a guard for tools that already
speak OpenAI's chat protocol — point your agent's approval model at it and every command gets a
single-word verdict in a few milliseconds. The upstream project is
[0xBakeer/arbiter](https://github.com/0xBakeer/arbiter); this branch is a pull request against it.
See [Use it from an OpenAI client](#use-it-from-an-openai-client).

Read the story behind it: [My cat woke me at five on a Sunday, so I built a local Jev](https://blog.0xbakeer.com/my-cat-woke-me-at-five-on-a-sunday-so-i-built-a-local-jev-1350583cddc6?sharedUserId=0xbakeer)

![How it works](docs/media/how-it-works.gif)

What it does with a state and a set of questions, and where it sits next to an LLM.

## Why the name, and what this is

A typed-decision model answers questions about a piece of text in **one forward pass** — no
tokens generated, no sampling, no loop. You hand it a state and a set of questions, and it
returns an answer and a probability distribution for each of them at once. There is no prose to
read back and nothing to argue with: it arbitrates, and your code decides what to do with the
numbers.

This repository is the serving layer around that and deliberately nothing more — a small HTTP
server that speaks TypeSafe's **Jev** API, so a client written against Jev works against this by
changing the base URL; cross-request micro-batching; routing between checkpoints; a playground;
and the measurements that picked every default. The model is a plug: **Laya** today, with its
three checkpoints (English, multilingual, and a typed-decisions fine-tune) resident at once and
automatic routing between them, and whatever is trained here next behind the same interface
([engines/README.md](engines/README.md)). So is the machine: one recipe per accelerator under
[recipes/](recipes), NVIDIA and Apple Silicon both measured.

On one GB10 with the card to itself it answers a single question in **20.9 ms** and fifty
questions in one call in **152 ms**, measured end-to-end over HTTP; on an M2 Max Mac the same
calls take **30.3 ms** and **462 ms**. Realistic states — a support ticket, an email, a diff — carry more tokens and more
questions, and land in the **tens of milliseconds**; the captured runs are in
[docs/use-cases.md](docs/use-cases.md). It uses about 5 GB of GPU memory, which is little enough
to sit next to a large language model on the same card.

![One call, many typed answers](docs/assets/one-call-typed-answers.svg)

---

## Install

Python 3.12 or newer, and one of the two lanes below. Each has its own notes and its own
numbers, because the wheels and the measurements differ by machine:

- **NVIDIA GPU → [recipes/nvidia](recipes/nvidia/README.md)**, with a recent driver. The torch
  wheels come from the CUDA 13.0 index, which has both aarch64 and x86_64 builds, so this is not
  specific to any one box.
- **Apple Silicon → [recipes/apple](recipes/apple/README.md)**, torch's MPS backend from plain
  PyPI. fp32 parameters, no CUDA graphs, and a slower cold load; the recipe says why. There is
  also an opt-in third engine on that lane, `ARBITER_ENGINE=laya_mlx`, which runs the community
  MLX port instead of torch: a quarter off a single question and a cold load of two seconds
  instead of ninety.

`./run.sh setup` picks the lane from `uname`, and the device is detected rather than configured:
`ARBITER_DEVICE` defaults to cuda if there is a CUDA device, else mps, else cpu.

```bash
git clone https://github.com/0xBakeer/arbiter.git
cd arbiter
./run.sh setup      # .venv, torch, the deps, and the three checkpoints (2.3 GB)
./run.sh serve      # http://localhost:8010
```

![Clone, setup, serve, on a Mac](docs/media/install.gif)

```bash
./run.sh status     # /healthz, /readyz, /v1/models
./run.sh smoke      # the playground page and seven real requests across all three checkpoints
./run.sh examples   # a support ticket through examples/support_triage.py
./run.sh bench      # the numbers below, about ninety seconds
./run.sh test       # both pytest suites; no GPU and no running server needed
./run.sh stop
```

## Playground

![The playground answering a support ticket, then a German invoice](docs/media/playground.gif)

`./run.sh serve` also serves a page at `http://localhost:8010/`: paste a state, add questions of
all three types, and read the answers with their probabilities, the checkpoint that answered and
the latency. It is one self-contained file, [`playground/index.html`](playground/index.html),
served from the same origin as the API, so it needs no configuration and the server needs no
CORS headers. To work on the page without a GPU, `python3 playground/serve_stub.py` serves it
with stub answers on port 8011, and `--proxy http://localhost:8010` forwards the calls to a real
server while keeping the page same-origin.

## arbiter plays

Six small games the model plays in the browser: snake, hopper, crossing, paddle, mines and a
dungeon crawl. Each tick the game writes its board down as a short piece of text, asks typed
questions about it, and plays the answer — one `POST /v1/systemone`, one forward pass, no tokens
generated. With `./run.sh serve` running, `open http://localhost:8010/showcase/`. The pages also
open from disk with no server at all, playing a recorded run.

It is worth being plain about what this shows: *this is a showcase of speed and of typed
decisions, not a claim about intelligence.* The scores are measured against a random and a
hand-written baseline on the same seeds, and the tables say where the model beats them and where
it does not.

[showcase/README.md](showcase/README.md) — the games, the harness, the measurements and how to
add one.

## Ask it something

```bash
curl -s localhost:8010/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for the same subscription this month and the second charge has not been refunded. I have emailed support twice with no reply.",
  "questions": {
    "urgency":     {"type": "score",  "instructions": "How urgent is this message?",
                    "criteria": ["not urgent", "can wait a day", "same day", "immediate"]},
    "category":    {"type": "choice", "instructions": "Which queue should this go to?",
                    "criteria": {"billing": "payments, invoices, refunds",
                                 "technical": "the product does not work",
                                 "account": "login, profile, permissions"}},
    "needs_human": {"type": "noul",   "instructions": "This message needs a human rather than an automated reply."}
  }
}'
```

```json
{
  "model": "laya-english",
  "answers": {
    "urgency":     {"type": "score",  "score": 2.12, "legend": {"0": "not urgent", "...": "..."},
                    "probabilities": {"0": 0.0121, "1": 0.1755, "2": 0.4913, "3": 0.3211},
                    "confidence": 0.18},
    "category":    {"type": "choice", "choice": "billing",
                    "probabilities": {"billing": 0.9841, "technical": 0.0092, "account": 0.0067},
                    "confidence": 0.93},
    "needs_human": {"type": "noul",   "noul": 0.002, "confidence": 0.998}
  },
  "usage": {"input_tokens": 210, "output_tokens": 0},
  "routing": {"model": "english", "reason": "English Latin text"},
  "latency_ms": 15.2
}
```

`routing`, `latency_ms`, `confidence` and `action` are additions; everything else is the Jev
response shape, and a Jev client ignores extra keys.

### The API

| endpoint | what it does |
|---|---|
| `POST /v1/systemone` | the Jev contract. `POST /v1/predict` is an alias |
| `GET /healthz` | the process is up |
| `GET /readyz` | the checkpoints are loaded and warm — this is the one to gate traffic on |
| `GET /v1/models` | the three checkpoints and their aliases, OpenAI-style |
| `GET /metrics` | Prometheus text: requests, questions, latency, batch size, queue depth |

Question types are Jev's: `noul` (a probability that a statement holds, criteria optional),
`choice` (up to 255 named options), `score` (2 to 10 ordered levels). Violations come back as
`422` with a JSON error body. Set `ARBITER_API_KEY` and the server requires
`Authorization: Bearer <key>`, answering `401` otherwise. More question rows in flight than
`ARBITER_MAX_QUEUE` gets `529`.

### Routing

The `model` field decides which checkpoint answers.

| you send | what runs |
|---|---|
| `"jev-latest"`, `"laya"`, `"auto"`, or nothing | the router picks |
| `"laya-english"` / `"english"` / `"en"` | English — ModernBERT-large, 512 tokens |
| `"laya-multilingual"` / `"multilingual"` / `"multi"` | multilingual — mmBERT-base, 1024 tokens, 100+ languages |
| `"laya-typed-decisions"` / `"typed-decisions"` / `"typed"` | the typed-decisions fine-tune, 1024 tokens |

Left to itself the router reads the state's script: anything non-Latin goes to multilingual,
Latin text that does not look English goes to multilingual, the rest goes to English. This is
not a nicety. The English checkpoint does not degrade gently off English, it collapses — 0.100
on 20-option Hindi intent classification against 0.050 for guessing — and it stays confident
while doing it. Detection costs microseconds, so the routing is free.

`typed-decisions` is never selected automatically. Ask for it by name.

## Examples

Nine runnable scripts in [`examples/`](examples), each one a real decision with the thresholds
in the caller and a review band in the middle. They need nothing but a Python 3 and a running
server: [`examples/arbiter_client.py`](examples/arbiter_client.py) is a single dependency-free file
whose API mirrors the hosted SDK, so code written against Jev ports by changing the import and
the base URL.

| Use case | Script | What it decides | Route |
|---|---|---|---|
| Support tickets | `support_triage.py` | department, urgency, frustration, refund, churn | `auto` · `review` · `escalate` |
| Inbox triage | `email_triage.py` | category, phishing, action needed, reply-by | `auto` · `review` · `escalate` · `block` |
| Agent shell commands | `tool_call_guard.py` | destructive, secrets, leaves repo, network, blast radius | `allow` · `ask` · `deny` |
| Pull requests | `pr_risk_gate.py` | credentials, migration, shared infra, rollback, blast radius | `allow` · `review` · `block` |
| Monitoring alerts | `alert_triage.py` | service, root cause, severity, duplicate of an open incident | `suppress` · `ticket` · `page` |
| Model selection | `model_router.py` | complexity, needs tools, needs long context | `fast` · `cascade` · `powerful` |
| RAG passages | `rag_relevance.py` | per-passage relevance, hierarchical past 12 | `keep` · `?` · `drop` |
| Message screening | `moderation.py` | jailbreak, harmful, PII, off-topic, severity | `allow` · `review` · `block` |
| Invoices (typed-decisions) | `invoice_fields.py` | currency, amount band, duplicate, approval, bank change | `pay` · `approve` · `review` |

```bash
python examples/support_triage.py                 # a built-in sample
python examples/email_triage.py --sample de       # watch the router pick multilingual
git diff main | python examples/pr_risk_gate.py   # exit code is the gate: 0/1/2
python examples/moderation.py --state msg.txt --json
```

Each script prints one row per question with a probability bar, the checkpoint that answered and
the latency, then the route its own thresholds chose. Captured output for every one of them is
in [`docs/use-cases.md`](docs/use-cases.md); the shapes they are built from — fan-out,
confidence-gated routing, composite scoring, hierarchical intent, cascade — are in
[`docs/patterns.md`](docs/patterns.md).

## Use it from your coding agent

![The guard judging Bash commands inside Claude Code](docs/media/claude-code.gif)

[`integrations/mcp/arbiter_mcp.py`](integrations/mcp/arbiter_mcp.py) is an MCP server over the same
endpoint: `arbiter_check`, `arbiter_classify`, `arbiter_score`, `arbiter_gate` and `arbiter_decide`, each
returning structured content plus one line of text. Point any MCP client at it:

```bash
pip install "mcp>=2"
claude mcp add arbiter --env ARBITER_URL=http://localhost:8010 -- python3 $PWD/integrations/mcp/arbiter_mcp.py
codex  mcp add arbiter --env ARBITER_URL=http://localhost:8010 -- python3 $PWD/integrations/mcp/arbiter_mcp.py
```

stdio is the default, which is what an agent spawning the file as a subprocess wants. On a shared
machine, run it once beside the server instead of once per agent host, and every client dials it:

```bash
python3 integrations/mcp/arbiter_mcp.py --transport streamable-http --host 0.0.0.0 --port 8020
# then point the client at http://HOST:8020/mcp (Hermes example, config.yaml):
#   mcp_servers:
#     arbiter:
#       transport: http
#       url: http://HOST:8020/mcp
```

`--transport stdio | sse | streamable-http`, and every flag has an `ARBITER_MCP_*` environment
twin (`ARBITER_MCP_TRANSPORT`, `_HOST`, `_PORT`, `_PATH`, `_MESSAGE_PATH`) so a systemd unit can
carry the configuration without a shell in the loop; see
[`systemd/arbiter-mcp.service`](systemd/arbiter-mcp.service). The default bind is `127.0.0.1`:
serving the LAN is a deliberate act, so say `--host 0.0.0.0` on purpose. The bridge imports
nothing but the standard library and the MCP SDK, so it needs a 57 MB venv, not the 5.6 GB one
the torch server lives in.

[`integrations/claude-code/`](integrations/claude-code) is a plugin that adds a skill (when to
use which primitive, how to shape state and questions, why thresholds belong in your code) and a
`PreToolUse` hook that judges every `Bash` command before it runs — allow, ask or deny, in tens
of milliseconds, failing open when the server is not there. Ready-made configuration for Codex,
OpenCode, omp and any generic `.mcp.json` client, plus a GitHub Actions job that gates pull
requests, is in [`integrations/`](integrations) and documented in
[`docs/integrations.md`](docs/integrations.md).

## Use it from an OpenAI client

Not every agent speaks Jev or MCP. Many have an "approval model" or "smart-approval" hook that
just calls an OpenAI-compatible `POST /v1/chat/completions` and expects the reply to be **one
word**. This fork adds exactly that endpoint, so any of them can gate on the arbiter with no
adapter code:

```bash
curl -s localhost:8010/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "laya-english",
  "messages": [{
    "role": "user",
    "content": "The following command was flagged as: deletes files outside the repo\n\n<command>\nrm -rf /var/log/app\n</command>\n\nAssess the ACTUAL risk of the shell operations in this command.\n\nRespond with exactly one word: APPROVE, DENY, or ESCALATE"
  }]
}'
```

```json
{
  "id": "gate-1789923664857",
  "object": "chat.completion",
  "created": 1789923664,
  "model": "laya-english",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "DENY"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
  "arbiter_gate": "Arbiter: risk 0.87 of 1, led by destroys_data 0.92 (confirm at 0.50, refuse at 0.78). (102 ms)"
}
```

`choices[0].message.content` is the whole decision:

| reply | the arbiter said | the client should |
|---|---|---|
| `APPROVE` | `allow` | run it, no human |
| `ESCALATE` | `ask` (or the arbiter is down) | show the user the approval prompt |
| `DENY` | `deny` | stop, do not run |

The extra `arbiter_gate` key carries the reason and the latency; a strict OpenAI client ignores
it, and a gate-aware client can surface it. The endpoint reuses the same guard policy as the
`arbiter_gate` MCP tool, so both entry points agree by construction.

A few details that matter when wiring it up:

- **The command is read from a `<command>` block** in the prompt, not from the whole message.
  The flagged command must sit inside `<command> … </command>`; an optional `flagged as: …` line
  is picked up too and fed to the decision. A read-only command short-circuits to a decision in
  roughly 0 ms without a forward pass; anything else is one `POST` to the router. A request with
  no `<command>` block (or no `messages` array, or a body that is not JSON) comes back `422`.
- **It fails open.** Every failure path returns the same one-word `ESCALATE` body, so a client
  that reads `choices[0].message.content` and treats anything but `APPROVE`/`DENY` as "ask the
  user" stays correct no matter what the HTTP status is: `503` while the models are still
  loading, `422` for a malformed request, and `200` for a thrown inference. A dead sidecar must
  never wedge the shell, and this is how it is guaranteed.
- **It is the same server.** No extra process, no extra model, no new dependency — the endpoint
  rides on the arbiter already serving `/v1/systemone` on the same port, and honours
  `ARBITER_API_KEY` exactly like the rest of the API.

## The numbers

Questions cycling through all three types, measured end-to-end over HTTP, on each machine's own
shipped defaults — a GB10 in autocast bf16, an M2 Max in fp32. Full method and the rest of the
figures in [bench/results.md](bench/results.md).

| | 1 question | 5 | 10 | 50 | throughput, 4-question calls, 1 / 8 / 32 callers |
|---|---:|---:|---:|---:|---:|
| GB10, LLM idle | **20.9 ms** | **31.1 ms** | **40.0 ms** | **152.4 ms** | 135 / 287 / 301 q/s |
| GB10, next to a busy LLM | 326.6 ms | 278.6 ms | 258.1 ms | 381.6 ms | 23 / 88 / 153 q/s |
| Apple M2 Max, fp32 | **30.3 ms** | **63.1 ms** | **107.4 ms** | **462.3 ms** | 72 / 102 / 106 q/s |
| Apple M2 Max, `laya_mlx` engine, fp32 | **22.7 ms** | **59.2 ms** | **102.7 ms** | **448.4 ms** | 75 / 103 / 109 q/s |
| model card, T4 english | 39.5 ms | — | 158.6 ms | 771 ms | — |
| model card, T4 multilingual | 32.8 ms | — | 72.3 ms | 337 ms | — |

The M2 Max ceiling arrives at eight callers and does not move after that, on either engine. The
MLX row is the opt-in Apple-silicon engine and was taken with the torch server still resident on
the same GPU; its paired control and the rest of its numbers -- including a cold load of 2.1 s
against 88-101 s -- are in [bench/results.md](bench/results.md). The T4 figures are the model
card's own, in-process SDK calls on older hardware; they are here for scale, not as a ranking.

### The two settings that are worth understanding

Both of them are CUDA stories, and both end differently on a Mac: `autocast` there means fp32,
`graphs` is refused outright, and `fp16` and `bf16` lose more precision than bf16 does here.
[recipes/apple](recipes/apple/README.md) has that side.

**`ARBITER_DTYPE`** — `autocast` (default) keeps fp32 parameters and runs the matmuls in bf16, as
the SDK does. `bf16` converts the parameters and drops autocast, which is 15–35% faster:
14.0 ms instead of 20.9 at one question, 380 questions/s instead of 287 at concurrency 8.

It is not the default because of what it costs. Against the SDK as reference, `autocast` is
identical to four decimals on all 22 equivalence questions — batching rows from different
callers into one forward changes nothing whatsoever. `bf16` moves reported probabilities by up
to **2.1e-2** and flips one argmax, on a five-level score question whose own confidence was
0.157. Laya's proposition is calibrated probabilities, and the model card is already explicit
that they ship over-confident and want refitting on your data before you trust them; spending
2e-2 of that budget to save six milliseconds is the wrong trade. It is one environment variable
away if your workload disagrees.

**`ARBITER_MODE`** — `eager` (default) or `graphs`, which needs CUDA and says so on any other
device rather than quietly running eager. Graphs mode captures the forward as a CUDA
graph per padded shape bucket and replays it, which removes the cost of launching several
hundred small kernels. It is faster where there is little work to amortise padding over and
slower where there is plenty:

| | 1 question | 5 | 10 | 50 | 8 concurrent |
|---|---:|---:|---:|---:|---:|
| eager | 20.9 ms | 31.1 ms | **40.0 ms** | **152.4 ms** | **287 q/s** |
| graphs | **16.8 ms** | **29.0 ms** | 44.6 ms | 229.4 ms | 248 q/s |

Eager is the default because it wins every shape above five questions and every concurrency
above one. Turn graphs on for a single-caller, small-call deployment — a guardrail in front of a
chat endpoint, say — where it is about 20% better.

Graphs mode also carries a 1.4e-3 deviation from the reference where eager carries none, which
is inside the gate but not nothing.

### Everything you can set

| variable | default | |
|---|---|---|
| `PORT` / `HOST` | `8010` / `0.0.0.0` | |
| `ARBITER_DEVICE` | detected | `cuda`, else `mps`, else `cpu`; set it to pin one, and `/readyz` reports what was taken |
| `ARBITER_ENGINE` | `laya` | which package under [`engines/`](engines) serves the requests; `laya_mlx` is the Apple-silicon MLX engine, opt-in |
| `ARBITER_MODELS` | all three | comma-separated; a subset saves memory, and the router falls back to what is loaded |
| `ARBITER_MODE` | `eager` | or `graphs`, which is CUDA-only and refuses to run anywhere else |
| `ARBITER_DTYPE` | `autocast` | or `bf16` on CUDA; `fp16` and `bf16` on MPS, where autocast means fp32; `fp32` (default), `fp16` or `bf16` under `laya_mlx` |
| `ARBITER_API_KEY` | unset | set it to require `Authorization: Bearer` |
| `ARBITER_BATCH_WAIT_MS` | `2` | how long a batch waits for company |
| `ARBITER_MAX_BATCH` | `64` | question rows per forward |
| `ARBITER_MAX_QUEUE` | `256` | rows in flight before `529` |
| `ARBITER_GRAPH_MAX_MARKERS` | `32` | questions with more options than this take the eager path |
| `ARBITER_MLX_CACHE_MB` | `1024` | `laya_mlx` only: how much freed GPU memory MLX may keep for reuse; `0` is its own unbounded default |
| `MODELS_DIR` | `./models` | |

## Runs next to an LLM

![The cascade next to an LLM](docs/assets/cascade-next-to-an-llm.svg)

Measured with an unrelated LLM already holding 63,871 MiB on the same GPU:

| | |
|---|---|
| Laya, all three checkpoints, `autocast` | **5,055 MiB** of GPU memory |
| the same in `ARBITER_DTYPE=bf16` | about 2,900 MiB |
| host RSS | 3.6 GB |
| host memory before / after starting it | 74 GiB / 82 GiB used of 121 |

The LLM was serving throughout and was unaffected. Sharing the card is not free in the other
direction, though, and the cost is worth knowing before you plan around it: while the LLM is
actually decoding at 92 % GPU utilisation, a call takes roughly ten to fifteen times longer at
the small question counts — 326 ms for one question instead of 20.9 — and throughput roughly
halves, to 88 questions/s at eight callers against 287 idle, with no errors. Both sets of tables
are in [bench/results.md](bench/results.md).

### Before it shares a card: the admission gate

Headroom at the moment you start is not the same fact as headroom you own. A configuration was
run on a 16,311 MiB card in which a second server was meant to come up after the first released
it; every metric recorded during the experiment looked fine, and in use it was not. The failure
mode is a race, not an arithmetic error: `nvidia-smi` reports free memory, and free memory
includes whatever the process that is still tearing down has not handed back yet.

[`wait-vram.sh`](wait-vram.sh) is the answer, wired in front of the server as
`ExecStartPre=`: it polls one named card until that card reports a floor of free memory, then
admits, and on timeout it **exits non-zero**. The non-zero exit is the whole design. The unit
carries `Restart=on-failure`, so a refusal becomes a retry twenty seconds later instead of an
oversubscription with a warning in the log. A gate that only prints is a gate that loses.

Measured on the box it runs on: it admitted at 15,885 MiB free against a 6,144 MiB floor the
moment the resident 35B lane released the card, and refused a run at 980 MiB free while that lane
still held 14.9 GiB, printing `FATAL` and the card it would not oversubscribe. Steady state,
server plus gate: 4,978 MiB used, 10,912 MiB free. [`tests/test_gpu_admission.py`](tests/test_gpu_admission.py)
pins the arithmetic, the per-card lookup (reading the neighbour's headroom is the bug), and the
wire itself, because a unit that ships without the `ExecStartPre=` line is the same bug wearing a
different file. It fakes `nvidia-smi` on `PATH` rather than assuming it is absent, so it passes
on a driver box and a headless one alike.

[`systemd/arbiter.service`](systemd/arbiter.service) is the system unit those measurements were
taken with, byte for byte as deployed, and it is a different animal from
[`deploy/arbiter.service`](deploy/arbiter.service), which remains the unprivileged user-unit
template for someone installing into a home directory. Three differences matter:
`Environment=CUDA_VISIBLE_DEVICES=1` so the process cannot drift onto a neighbour's card,
`Environment=ARBITER_DEVICE=cuda` rather than `auto` so a missing CUDA context aborts instead of
serving quietly on the CPU at a tenth the speed, and the gate. The two are meant to be read as
what each one is for: `deploy/` is the template you edit, `systemd/` is the record of a unit
that ran.

[`build-venv.sh`](build-venv.sh) is the same kind of record for the other half of a rebuild. It
installs the package set the measurements above belong to, pinned (`laya==0.3.4`, `mcp>=2`, torch
from the CUDA 13.0 index, which carries both `aarch64` and `x86_64` builds), and it deliberately
omits the checkpoint download that `./run.sh setup` performs: on a LAN of machines that already
have the weights, re-pulling 2.3 GB from the hub is a slower and less reproducible way to obtain
a file you already hold and can hash. It ends by importing `server.app` and printing the device
capability of every card it can see, so a build that produced a CPU-only torch fails the build
step rather than the deployment.

Three checkpoints is 1.16B parameters, which
is small enough that the decision is about whether you want all three rather than about whether
they fit; `ARBITER_MODELS=english` alone is about 1.9 GB.

The reason all three stay resident is that the SDK's `Router` defaults to keeping one. A server
that alternates between English and German requests would then reload a checkpoint from disk on
every request — seconds, against microseconds to decide. Preloading is the only sensible server
setting, and the router is handed the already-built models so nothing is loaded twice.

## Honest limits

These are the model's, not the server's, and they are quoted from the model card because they
matter more than anything this recipe does.

- **The base checkpoints are near chance on typed decisions, zero-shot** — 0.362 for English and
  0.352 for multilingual, against 0.318 for random and 0.461 for always guessing the majority
  class. The 0.766 figure belongs to the checkpoint fine-tuned on that benchmark's own training
  split. Laya is a fast base to specialise, not a zero-shot decision engine.
- **High-cardinality choice questions run out of tokens.** A sequence is split into an option
  budget (`head_max_len`) and what is left for the state. English gets 512 total with 192 for
  options; multilingual and typed-decisions get 1024 with 256. A 77-option question like
  Banking77 therefore gets about 3–4 tokens per label and accuracy collapses — 0.425, against
  0.870 for Jev. Above about 125 options the English checkpoint cannot fit the markers at all,
  and this server returns `422` rather than silently answering a truncated question. If you need
  50+ options, either raise `head_max_len` and `max_len` in the checkpoint config, or split the
  question into a coarse-to-fine hierarchy.
- **Ordinal `score` questions are the weakest primitive** — 0.372 on SST-5.
- **It ships over-confident.** Refitting one temperature per (question type, option count) moves
  mean ECE from **0.466 to 0.081** for English and **0.314 to 0.106** for multilingual. Do that
  on your own data before you trust the probabilities. It is also why this recipe will not
  spend 2e-2 of probability error on a faster dtype.
- **The root checkpoint is English only.** Use multilingual for anything else — the router does
  this for you, and the reason it is aggressive about it is in the routing section above.

## Documentation

- [docs/use-cases.md](docs/use-cases.md) — the nine examples, their question sets, and the
  output captured from a live server for each one
- [docs/patterns.md](docs/patterns.md) — the five shapes the examples are built from, with a
  pasteable sketch of each
- [docs/integrations.md](docs/integrations.md) — the MCP server, the Claude Code plugin, the
  per-agent configuration and the CI job
- [docs/serving-options.md](docs/serving-options.md) — what was considered, what shipped, and
  what was ruled out with the measurement that ruled it out
- [recipes/nvidia](recipes/nvidia/README.md) and [recipes/apple](recipes/apple/README.md) — one
  install-and-serve recipe per machine, with that machine's numbers and its own chosen dtype
- [engines/README.md](engines/README.md) — the four methods a model backend implements
- [showcase/README.md](showcase/README.md) — the six games, what the model decides in each, and
  the measured scores against the random and hand-written baselines
- [bench/results.md](bench/results.md) — every figure and how it was taken
- [CHANGELOG.md](CHANGELOG.md) — which defaults changed when
- [CREDITS.md](CREDITS.md) — the model, the encoders, and the API shape are other people's work
- [My cat woke me at five on a Sunday, so I built a local Jev](https://blog.0xbakeer.com/my-cat-woke-me-at-five-on-a-sunday-so-i-built-a-local-jev-1350583cddc6?sharedUserId=0xbakeer) — the
  article this came out of: why a typed-decision model, and what it is for

MIT licensed. The model weights are Apache-2.0 from Convai Innovations and carry their own terms.
