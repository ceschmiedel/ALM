<h1 align="center">ALM · Agent Language Model</h1>

<p align="center">
  <strong>A governed federation of domain models, orchestrated by a Context Graph.</strong><br>
  Give every Expert Agent its own model instead of sharing one monolithic LLM — and move the
  intelligence from the model into the orchestration.
</p>

<p align="center">
  <a href="#-why">Why</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick start</a> •
  <a href="#-domain-packs">Domain packs</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-cli">CLI</a> •
  <a href="#-rest-api">REST API</a> •
  <a href="#-documentation">Docs</a>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%E2%89%A53.11-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/draiven-io/agentic-bus"><img src="https://img.shields.io/badge/protocol-LIP-orange" alt="Liquid Interface Protocol"></a>
</p>

> 🇧🇷 [Leia em português](README.pt-BR.md)

---

## 🧭 Why

The dominant agent architecture puts one frontier LLM at the centre: every agent calls the same
model, varying only the prompt. It works, and it hits four structural limits in regulated,
high-volume enterprise settings — aggregate token cost, external latency and single points of
failure, data residency, and generic behaviour that never internalises a client's ontology.

**ALM's thesis:** in a domain, a well-routed federation of specialists beats a generalist — but
that superiority is *not automatic*. It depends almost entirely on the quality of the context and
routing layers, not on the individual models. So this project invests where the leverage is:

| The models | The orchestration |
|---|---|
| Replaceable | Where the value accumulates |
| A LoRA adapter, an SLM, an API | The Context Graph that routes them, the memory that connects them, the governance that makes them auditable |

ALM does not replace the LLM. It repositions it: from omnipresent executor to occasional
orchestrator and teacher of the domain models. Where sovereignty, latency and volume matter,
the local specialist takes over; where open-ended reasoning is essential, the LLM stays in the loop.

**The fallback to the LLM is by design, not a failure.** It guarantees the federation is never
worse than the monolithic baseline: in the worst case it hands the task back to the LLM. In the
common case it answers locally, cheaply, and inside the client's perimeter.

---

## 🏗️ Architecture

Five functional layers plus two cross-cutting ones. Each maps to a Python package.

```
                    ┌──────────────────────────────────────────┐
   intent  ───────► │ L1  Interface & ingestion      alm.protocol
                    │     Liquid Interface Protocol (LIP)      │
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ L2  Cognitive router           alm.router │
                    │     classify · decompose · match · confidence
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ L4  Dependency orchestration   alm.orchestration
                    │     plan ──► DAG · parallel where independent
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ L3  Expert federation          alm.experts
                    │     own model (ALM) + own retrieval (CMRAG)
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ L5  Arbitration & synthesis    alm.arbitration
                    │     verifier · confidence · authority · judge
                    └───────────────────┬──────────────────────┘
                                        ▼
                                    answer + full provenance trace

   CX  Context & memory   alm.graph + alm.cmrag   ── feeds routing, execution, synthesis
   GV  Governance (IBAC)  alm.governance          ── no context access happens outside it
```

### What is actually new here

Mixture-of-Experts is not new; the difference is **where the routing lives**. Inside an MoE LLM
the router is internal, implicit, learned and unauditable. In ALM the routing is **external,
explicit and governable**: the Context Graph decides which expert is engaged, in what order, and
how the results are combined — and it records why.

That makes explicit what a monolith does implicitly. It is simultaneously the largest engineering
cost of the federation and its largest advantage: a monolith cannot show how it decided; a
federation can. In sectors where auditability is mandatory, that stops being a differentiator and
becomes a prerequisite.

### Layer map

| Layer | Package | Responsibility |
|---|---|---|
| **L1** | `alm.protocol` | LIP-compatible envelopes; turns intent into a structured `TaskEnvelope` |
| **L2** | `alm.router` | Intent classification, task decomposition, capability matching, confidence & fallback |
| **L3** | `alm.experts`, `alm.models` | Expert Agents, model tiers, multi-adapter LoRA serving |
| **L4** | `alm.orchestration` | Execution DAG: parallel where independent, sequential where dependent |
| **L5** | `alm.arbitration` | Conflict resolution and synthesis with a traceable decision record |
| **CX** | `alm.graph`, `alm.cmrag` | Context Graph (ontology, capabilities, authority, blackboard) + domain-scoped retrieval |
| **GV** | `alm.governance` | IBAC over every context read and action, with an append-only audit log |
| — | `alm.evaluation` | The falsifiability harness: federation vs monolithic baseline |
| — | `alm.distillation`, `alm.mlops` | Teacher→student pipeline, versioning, eval-gated promotion, drift |
| — | `alm.bridge` | Publish the federation as a provider agent on an Agentic Bus |

---

## 🚀 Quick start

### Install

```bash
git clone https://github.com/draiven-io/alm.git
cd alm
pip install -e ".[dev]"
```

### Run the demo — no API key, no GPU

The bundled `packs/demo-enterprise` pack runs on deterministic local backends so the whole
pipeline is observable end to end before you connect a single model:

```bash
alm init                              # create the database
alm pack install packs/demo-enterprise
alm ask "Clause 7.2 caps liability at 12 months of fees — is that consistent with the payment terms, and what is the exposure?"
```

You get the answer, the plan that produced it, which expert contributed what, how the conflict
between the legal and finance experts was arbitrated, and the cost and latency of every call.

```bash
alm ask "…" --trace          # full layer-by-layer trail
alm ask "…" --json           # machine-readable result
```

### Connect real models

```bash
alm install                   # interactive wizard, writes .env
alm model add finance-slm --tier slm --backend ollama --model qwen2.5:3b
alm model add router --tier micro_slm --backend ollama --model qwen2.5:0.5b
alm model add orchestrator --tier orchestrator --backend openai --model gpt-4o
alm model list
```

Multi-adapter serving — many experts, one loaded base, one GPU:

```bash
alm model add legal-expert \
  --tier slm --backend vllm \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --adapter legal-lora-v3
```

### Embed it

```python
import asyncio
from alm import FederationRuntime

async def main():
    runtime = FederationRuntime.from_pack("packs/demo-enterprise")
    result = await runtime.run("What is our exposure under the Acme MSA?")
    print(result.answer)
    print(result.trace.render())
    print(result.metrics.model_dump())

asyncio.run(main())
```

---

## 📦 Domain packs

A **domain pack** is a directory that declares an entire vertical: its ontology, its Expert
Agents, their models, their corpus, the governance policies and the evaluation set. Installing a
pack registers all of it in the Context Graph — no code.

```
packs/my-domain/
├── pack.yaml              # metadata, models, defaults
├── ontology/
│   └── legal.yaml         # entity types, relations, sensitivity
├── experts/
│   ├── contracts.yaml     # capabilities, prompt, model binding, retrieval scope
│   └── risk.yaml
├── corpus/                # documents CMRAG indexes, per domain
│   └── legal/*.md
├── policies/
│   └── ibac.yaml          # what each expert may read and do
└── eval/
    └── legal.jsonl        # the falsifiability set for this domain
```

```bash
alm pack validate packs/my-domain     # catches ontology and routing problems before install
alm pack install packs/my-domain
alm graph show --domain legal
```

See [docs/domain-packs.md](docs/domain-packs.md) for the full schema.

---

## 🎯 Choosing a model per expert

Do not start by training models. The right question is never "which model do I train" but
"does this agent actually need its own model, or is retrieval over a shared model enough?"

| Tier | Size | Use for |
|---|---|---|
| `micro_slm` | < 1B | Classification, extraction, routing, normalisation. CPU is enough. |
| `slm` | 1–4B | The standard domain expert: in-domain reasoning, structured generation. One GPU. |
| `small` | 4–8B | Longer reasoning or richer generation, still on-premise. |
| `orchestrator` | frontier | Planning, ambiguous cases, hard arbitration, generating training data. Used sparingly. |

And climb the specialisation ladder only when the rung below stops working — measured, not
assumed:

| Technique | Effort | Fixes |
|---|---|---|
| RAG over a shared model | Low | Factual knowledge that changes often. **Always the first choice.** |
| LoRA adapter | Medium | Format, tone, vocabulary, in-domain reasoning patterns. Cheap to train *and cheap to serve*. |
| Full SFT | Medium-high | Deep specialisation when the adapter is not enough. |
| Continued pretraining | High | Domains whose language the base model simply does not know. |

`alm expert promote-check <expert>` evaluates the four business triggers from the architecture —
sovereignty, volume, latency, accuracy — against real data and tells you whether a dedicated model
is justified yet. Most experts stop at LoRA.

---

## 📊 Evaluation

> The ALM thesis is falsifiable and is treated as such. Without an evaluation harness, ALM is
> faith; with one, it is engineering.

The harness runs the same battery of tasks in two arms — **(a)** the monolithic LLM with a good
prompt and RAG, and **(b)** the federation — and compares them on the dimensions that matter
together, not one at a time:

```bash
alm eval run --pack packs/demo-enterprise --compare
```

This is the **actual output** on the bundled demo pack, not an illustration:

```
DOMAIN all · 28 cases · federation vs baseline

  metric                      federation      baseline       delta
  ──────────────────────────────────────────────────────────────
  domain accuracy                  0.821         0.893      -0.072  ✗
  cost per query               $0.000000     $0.000000           —  ✓
  latency p95 (ms)                    38             7     +434.3%  ✗
  fallback rate                    0.536             —           —
  consistency                      0.253         0.325      -0.072  ✗
  auditability                     1.000         0.000      +1.000  ✓

  VERDICT  baseline
  The federation does not hold accuracy against the baseline. Keep these agents as
  RAG over the shared model — ALM applies where it wins, not everywhere on principle.
```

**Read that carefully: on this demo, the federation loses.** That is the correct result and it is
worth understanding rather than hiding.

The demo runs both arms on the deterministic *extractive* backend, which removes exactly what a
federation buys — in-domain reasoning by a specialised model — while keeping all of its overhead
(decomposition, several retrieval passes, arbitration). Both arms are reading the same corpus with
the same non-model responder, so the baseline's single unscoped retrieval is simply a shorter path
to the same sentences. Cost is $0 on both sides because no priced model is registered, and the
latency comparison is between two sub-40ms local runs.

What *does* show through is structural: **auditability 1.00 against 0.00**. The federation can say
who contributed what and why one line prevailed; the monolith cannot, by construction.

Point the pack at real models (`--backend ollama` or `vllm`) and the accuracy and cost columns
start measuring what they are meant to measure. Until then, treat this as proof that **the harness
is honest**, not as evidence for the thesis. A tool that only ever reported "the federation wins"
would be worth nothing.

If the federation does not win **on the set**, the honest decision is to keep that agent as RAG
over the shared model — and `alm eval run` says so explicitly rather than letting you read the
number you prefer.

---

## 🔐 Governance and sovereignty

Where ALM really shines is not cost, it is sovereignty. With the experts and their models running
on-premise or in the client's VPC, sensitive data never leaves the perimeter — which turns NIS2,
EU AI Act and LGPD requirements from an obstacle into an architectural feature.

IBAC governs each *agent*, not only each user: an expert in one domain cannot read another
domain's context without explicit authorisation, and every access is recorded.

```yaml
# policies/ibac.yaml
policies:
  - id: finance-no-hr
    description: The finance expert must not read HR records.
    effect: deny
    subjects: [finance-expert]
    resources: ["domain:hr", "entity:EmploymentContract"]

  - id: restricted-needs-claim
    effect: deny
    resources: ["sensitivity:restricted"]
    unless_claims: [legal.privileged]
```

```bash
alm audit tail --session ses_a1b2c3
alm audit export --format jsonl > audit.jsonl
```

---

## 🔄 Distillation

The real bottleneck of ALM is not training, it is having quality domain data. The pipeline uses
the frontier LLM as a teacher:

```bash
alm distill seed     --pack packs/demo-enterprise --domain legal   # from the real corpus
alm distill generate --domain legal --teacher orchestrator          # teacher Q&A + reasoning
alm distill filter   --domain legal                                 # validate, dedupe, drop bad data
alm distill export   --domain legal --out .alm/datasets/legal.jsonl # train/eval split
alm distill feedback --domain legal                                 # production failures back in
```

Bad data teaches errors persistently, so the filter stage is not optional and the eval split is
never touched during training. Step 5 is where the federation improves with use: every case an
expert failed or escalated is captured and fed back.

---

## 💻 CLI

```
alm install                      Interactive setup wizard
alm init                         Create/migrate the database
alm serve                        Start the REST + WebSocket API

alm ask <text>                   Run one intent through the federation
alm ask <text> --trace --json    …with the full trail / machine-readable

alm pack list|validate|install|remove
alm graph show|search|stats|export
alm expert list|show|test|promote-check
alm model list|add|remove|versions|promote|rollback
alm cmrag index|search|stats
alm eval run|list|show|compare
alm distill seed|generate|filter|export|feedback
alm audit tail|export
alm bridge serve                 Publish the federation on an Agentic Bus
alm config show
```

Every command supports `--json`.

---

## 🔌 REST API

```bash
alm serve      # http://localhost:8800  ·  Swagger at /docs
```

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/ask` | Run an intent through the federation |
| `GET` | `/v1/sessions/{id}` | Full result, plan, answers and trace |
| `WS` | `/v1/stream` | Live trace events while a run executes |
| `GET` | `/v1/graph/nodes` · `/v1/graph/search` | Inspect and query the Context Graph |
| `GET` | `/v1/experts` · `POST` `/v1/experts/{id}/test` | Expert inventory and probes |
| `GET/POST` | `/v1/models` | Model registry |
| `POST` | `/v1/cmrag/search` | Domain-scoped retrieval |
| `POST` | `/v1/eval/run` · `GET` `/v1/eval/runs` | Evaluation harness |
| `GET` | `/v1/audit` | IBAC decision log |
| `GET` | `/v1/health` · `/v1/stats` | Health and inventory |

---

## 🔗 Liquid Interface Protocol

L1 *is* LIP. The envelope is wire-compatible with
[Agentic Bus](https://github.com/draiven-io/agentic-bus), so an ALM federation can join an
existing bus as a provider agent — it registers a capability per domain, offers on matching
intents, and returns `complete` messages whose artifacts carry the provenance chain.

```bash
alm bridge serve --coordinator ws://localhost:8765 --agent-id alm-federation
```

---

## ⚠️ Known limits

An honest architecture states where it can fail.

| Risk | Mitigation in this repo |
|---|---|
| **MLOps load** — many models to train, version, monitor | Multi-adapter serving (fewer artifacts), `alm.mlops` versioning + eval-gated promotion + drift detection from day one |
| **Knowledge fragmentation** — isolated models lose synergy | Memory and routing stay centralised in the Context Graph: models execute, the graph integrates |
| **Inconsistency between agents** | Shared base model, common prompt guidelines, and L5 synthesis normalising the final output |
| **Small models drifting on open reasoning** | Designed fallback to the orchestrator: the federation is never worse than the baseline |
| **Falling LLM prices erode the cost case** | The value proposition is anchored in sovereignty, latency, auditability and defensibility — none of which fall with the token price |

ALM is not the default architecture for everything. It is a premium layer for regulated,
high-volume verticals, adopted incrementally: the LLM orchestrates and covers the long tail;
domain models take over where sovereignty, latency and volume demand it.

---

## 📚 Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | The layered architecture in depth |
| [docs/domain-packs.md](docs/domain-packs.md) | Pack schema and authoring guide |
| [docs/routing.md](docs/routing.md) | Decomposition, matching, confidence, fallback |
| [docs/arbitration.md](docs/arbitration.md) | Conflict resolution strategies and synthesis |
| [docs/serving.md](docs/serving.md) | Model tiers, backends and multi-adapter LoRA serving |
| [docs/evaluation.md](docs/evaluation.md) | The controlled experiment and its metrics |
| [docs/distillation.md](docs/distillation.md) | Teacher→student pipeline and the feedback loop |
| [docs/governance.md](docs/governance.md) | IBAC, sovereignty and the audit trail |
| [docs/deployment.md](docs/deployment.md) | Docker, PostgreSQL, vLLM, on-premise |
| [docs/adoption.md](docs/adoption.md) | The three-wave adoption path |

---

## 📖 Foundations

ALM is not speculative; each pillar rests on established literature.

- **SLMs per agent are viable and preferable** — Belcak et al. (2025), *Small Language Models are the Future of Agentic AI*, [arXiv:2506.02153](https://arxiv.org/abs/2506.02153)
- **Intelligence migrates to the system, not the model** — Zaharia et al. (2024), *The Shift from Models to Compound AI Systems*, BAIR; Jain et al. (2024), [arXiv:2412.01868](https://arxiv.org/abs/2412.01868)
- **A sum of specialists can beat the generalist** — Jacobs et al. (1991), *Adaptive Mixtures of Local Experts*; Shazeer et al. (2017), [arXiv:1701.06538](https://arxiv.org/abs/1701.06538); Jiang et al. (2023), *LLM-Blender*
- **Routing with fallback cuts cost** — Chen, Zaharia & Zou (2023), *FrugalGPT*, [arXiv:2305.05176](https://arxiv.org/abs/2305.05176)
- **How to build the domain models** — Lewis et al. (2020), *RAG*, [arXiv:2005.11401](https://arxiv.org/abs/2005.11401); Hu et al. (2021), *LoRA*, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685); Hinton et al. (2015), *Distillation*, [arXiv:1503.02531](https://arxiv.org/abs/1503.02531); Gururangan et al. (2020), *Don't Stop Pretraining*, [arXiv:2004.10964](https://arxiv.org/abs/2004.10964)
- **The economics of serving the federation** — Sheng et al. (2023), *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*, [arXiv:2311.03285](https://arxiv.org/abs/2311.03285)

---

## 🤝 Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

<p align="center">Made with ❤️ by <a href="https://draiven.io">Draiven</a></p>
