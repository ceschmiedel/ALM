# Adoption path

ALM does not replace the LLM overnight, and it should not. The recommended architecture is hybrid
and the path is incremental: the LLM orchestrates and covers the long tail; SLMs take over the
domains where sovereignty, latency or volume justify the investment.

## When to promote an agent to a dedicated model

An Expert Agent moves from "RAG over a shared model" to "dedicated model" when at least one trigger
is confirmed **by evaluation**:

| Trigger | Confirmed when |
|---|---|
| **Sovereignty** | The domain handles data that cannot leave the perimeter |
| **Volume** | Aggregate LLM cost exceeds the cost of training and serving a dedicated model |
| **Latency** | The task needs real-time response an external call cannot guarantee |
| **Accuracy** | RAG has hit its ceiling and still errs systematically on domain behaviour |

```bash
alm expert promote-check finance-expert \
  --calls 120000 --latency-requirement 800 --observed-p95 2400 \
  --rag-accuracy 0.71 --accuracy-target 0.85
```

## The three waves

**Wave 1 · Foundation.** Formalise decomposition (L2) and arbitration (L5) over the current
federation, still on a shared model. Build the per-domain evaluation harness. **No new models yet.**

```bash
alm pack install packs/my-domain     # ontology, capabilities, policies
alm eval run --pack packs/my-domain --compare
alm eval calibrate --pack packs/my-domain
```

**Wave 2 · First dedicated model.** Pick one domain with a clear trigger — a client with a
sovereignty requirement, say — and build the first expert with a LoRA adapter. Measure against the
baseline. Prove the distillation pipeline and multi-adapter serving.

**Wave 3 · Federation.** Expand to several dedicated agents over the shared base, with versioning
and drift monitoring in operation. The federation carries most of the volume; the LLM becomes
fallback.

## The principle

> Do not start by training models. Start by formalising routing and arbitration and building the
> evaluation. Dedicated models come afterwards, guided by a business trigger and validated by
> measurement, one domain at a time.
