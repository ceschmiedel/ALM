# Distillation

The real bottleneck of ALM is not training — it is having quality domain data.

## The five stages

```bash
alm distill seed     --domain legal                 # 1 · from the real corpus
alm distill generate --domain legal                 # 2 · teacher writes Q/A + reasoning
alm distill filter   --domain legal                 # 3 · validate, dedupe, drop bad data
alm distill export   --domain legal --out data.jsonl # 4 · train/eval split
alm distill feedback --domain legal                 # 5 · production failures back in
```

**1 · Seed with the real thing.** Start from the domain's own documents, anchored to the ontology.
Synthetic data unmoored from the corpus teaches a domain that does not exist.

**2 · Generate with the teacher.** The frontier model produces question/answer pairs and reasoning
chains over that material, covering the cases the agent will see in production.

**3 · Filter and validate.** Bad data teaches errors *persistently*, so this stage is not optional.
Every rejection is recorded with its reason rather than deleted — a teacher producing 80% rejects is
telling you something about the corpus.

**4 · Train the student.** The eval split is content-hash based and deterministic, so re-exporting
after adding examples keeps previously-eval cases in eval. A case drifting into the training set
between runs silently invalidates every comparison made against it.

**5 · Close the loop.** Every case an expert failed or escalated is captured automatically during
normal operation and becomes the next cycle's seed material. This is where the federation improves
with use rather than decaying.

This cycle is also where defensibility comes from: over time each Expert Agent accumulates a model
tuned to *that client's* domain. A competitor plugs in a generic API; you have a specialist that
actually learned the business.

## Training an adapter

```bash
pip install "draiven-alm[train]"
```

```python
from alm.distillation import TrainingConfig, train_adapter

result = train_adapter(TrainingConfig(
    base_model="meta-llama/Llama-3.2-3B-Instruct",
    dataset_path=".alm/datasets/legal.jsonl",
    output_dir=".alm/adapters",
    adapter_name="legal-lora-v1",
    lora_rank=16,
))
```

Then register it and gate its promotion on evaluation:

```bash
alm model add legal-expert --backend vllm \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --adapter legal-lora-v1 --adapter-uri .alm/adapters/legal-lora-v1

alm eval run --pack packs/my-domain --domain legal --compare
alm model promote legal-expert v1        # refused without a recorded evaluation
alm model rollback legal-expert          # undo the last promotion
```

`alm model promote` refuses a candidate with no evaluation record, and refuses one that would
regress the deciding metric. `--force` exists but makes accepting the risk explicit.
