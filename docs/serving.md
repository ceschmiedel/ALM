# Model tiers and serving

## Choosing a tier

| Tier | Size | Use for | Runs on |
|---|---|---|---|
| `micro_slm` | < 1B | Classification, extraction, routing, normalisation | CPU |
| `slm` | 1–4B | The standard domain expert | One GPU |
| `small` | 4–8B | Longer in-domain reasoning | Still on-premise |
| `orchestrator` | frontier | Planning, hard arbitration, teaching | Hosted API or a large self-hosted model |

Routing by tier is what keeps each request on the cheapest hardware that can answer it. The
orchestrator is used *sparingly* — planning only when a request is composite, judging only when
cheaper strategies fail, and answering only on fallback.

## Choosing how to specialise

Climb the ladder only when the rung below stops working, measured rather than assumed.

| Technique | Effort | Fixes |
|---|---|---|
| RAG over a shared model | Low | Knowledge that changes often. **Always first.** |
| LoRA adapter | Medium | Format, tone, vocabulary, reasoning patterns. Cheap to train **and to serve**. |
| Full SFT | Medium-high | Deep specialisation when an adapter is not enough |
| Continued pretraining | High | Domains whose language the base model does not know |

```bash
alm expert promote-check contracts-expert --sovereignty
```

evaluates the four business triggers — sovereignty, volume, latency, accuracy — against real data.
Most experts stop at LoRA.

## Multi-adapter serving: the economics

The conceptual error is imagining "one model per agent" as "one GPU per agent". With multi-adapter
serving, one base loaded on the GPU serves many experts, each bringing only its own adapter (a few
megabytes), swapped per request. Cost becomes *base + adapters*, not N full models.

```bash
alm model add legal-expert \
  --tier slm --backend vllm \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --adapter legal-lora-v3 \
  --endpoint http://localhost:8000/v1

alm model list      # shows shared bases, adapters, and full loads saved
```

On an OpenAI-compatible server hosting LoRA adapters, each adapter is addressed by its own name
while sharing base weights — so `adapter` *is* the model name on the wire.

For embedded single-process use (and for evaluating a freshly trained adapter before publishing it),
`--backend transformers` loads base weights once per `base_model` and attaches adapters via PEFT.
Requires `pip install "draiven-alm[local]"`.

## Cost accounting

Prices are **not** baked in. A vendor price table in source code is wrong the week after it is
written and says nothing about self-hosted serving, which is where most of a federation's cost sits.

```bash
alm model cost --gpu-hourly 2.40 --tokens-per-second 1800 --utilisation 0.7
```

converts a GPU-hour rate into cost per 1k tokens, so on-premise and hosted models are comparable in
one number. Comparing a federation to a hosted LLM on token price alone is the mistake this avoids.

Models with no price set report $0, and `alm model add` warns when that happens.

## Backends

| Backend | Use |
|---|---|
| `ollama` | Local, sovereign, easiest start |
| `vllm` | On-premise, multi-adapter, production |
| `openai` / `azure` / `anthropic` / `google` | Hosted orchestrator tier |
| `transformers` | In-process, PEFT adapters |
| `heuristic` | Deterministic extractive stand-in — **not a language model** |
| `scripted` | Canned responses for tests |

`alm doctor` reports which models are reachable and warns when an expert is bound to a placeholder
backend.
