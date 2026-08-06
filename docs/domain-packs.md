# Domain packs

A pack declares an entire vertical as data: ontology, Expert Agents, models, corpus, governance
policies and the evaluation set that decides whether any of it beats the baseline.

```
packs/my-domain/
├── pack.yaml
├── ontology/legal.yaml
├── experts/contracts.yaml
├── corpus/legal/*.md
├── policies/ibac.yaml
└── eval/legal.jsonl
```

```bash
alm pack validate packs/my-domain    # strict; run before installing
alm pack install  packs/my-domain
```

## pack.yaml

```yaml
name: my-domain
version: 0.1.0
description: What this federation covers.
tenant: default

models:
  legal-slm:
    tier: slm                 # micro_slm | slm | small | orchestrator
    backend: ollama           # ollama | vllm | openai | azure | anthropic | google | transformers
    model: qwen2.5:3b
    cost_per_1k_input: 0.0
    cost_per_1k_output: 0.0

defaults:
  router_model: router-micro
  expert_model: legal-slm
  orchestrator_model: gpt-4o

ontologies: [ontology/legal.yaml]
experts:    [experts/contracts.yaml]
policies:   [policies/ibac.yaml]
corpus:     corpus
eval:       [eval/legal.jsonl]
```

## Ontology

The ontology does three jobs at once, which is why it is one artifact: it defines the vocabulary the
domain model must master, it structures training data, and it is the schema CMRAG retrieves against.
Modelling the ontology and training the model are the same task from two angles.

```yaml
domain: legal
label: Legal & Contracts
entities:
  - name: Clause
    parent: Contract
    description: A numbered provision carrying an obligation or right.
    keywords: [clause, provision, section, article]
    attributes: [number, heading]
    sensitivity: confidential      # public | internal | confidential | restricted
relations:
  - name: contains_clause
    source: Contract
    target: Clause
```

**`sensitivity` is not decoration** — IBAC reads it, and a chunk inherits the strictest level among
the entities of *its own domain* that it mentions.

> **Keywords must discriminate.** A keyword that fires on ordinary prose will tag half the corpus.
> `breach` appears in every contract; if the risk domain's `Incident` entity claims it, contract
> clauses get reclassified as restricted and the legal expert is locked out of its own corpus. Use
> `data breach`, not `breach`.

## Expert

```yaml
id: contracts-expert
domain: legal
model: legal-slm
tier: slm

authority:               # weight of this expert's voice per domain, used by L5
  legal: 1.0
  finance: 0.35

capabilities:
  - id: analyze_clause
    description: Analyse a contractual clause and its liability allocation.
    keywords: [clause, liability, cap, termination, data protection, ...]
    examples:            # the strongest matching signal available
      - Is clause 7.2 enforceable?
      - Where may customer personal data be stored?
    operates_on: [Clause, Contract, LiabilityCap]
    produces:    [ClauseAnalysis]
    outputs:
      - name: finding
        type: string
      - name: severity
        type: string
        enum: [low, medium, high, critical]   # closed vocabulary → arbitrable
      - name: confidence
        type: number

retrieval:
  domains: [legal]       # defaults to the expert's own domain
  top_k: 6

verifier_rules:
  - id: must-cite
    type: requires_citation
    severity: error
```

### Declaring scope well

`keywords` and `examples` are the router's matching surface. Cover the vocabulary of the **corpus**,
not just the headline topic. Examples matter most because they are phrased the way real requests are.

`produces` and `operates_on` are how execution order is derived: an expert whose capability
`operates_on: [FinancialExposure]` automatically runs *after* the one that `produces` it. Nobody
writes the order down.

### Verifier rules

| Type | Checks |
|---|---|
| `required_fields` | Named output fields are present and non-empty |
| `requires_citation` | The answer cites the domain corpus |
| `numeric_range` | A field sits within `minimum`/`maximum` |
| `numeric_consistency` | The first field equals the sum of the rest |
| `forbidden_terms` | Named terms are absent |
| `regex_match` | A field matches a pattern (enforce a vocabulary) |
| `min_confidence` | Calibrated confidence clears a floor |

## Corpus

`corpus/<domain>/*.md` — the first path segment names the domain. Indexing is content-hash based, so
re-running on an unchanged corpus is a no-op and safe on every deploy.

## What validation catches

`alm pack validate` fails on: an ontology with unknown parents, cycles or dangling relations; an
expert in a domain no ontology defines; a capability operating on an undefined entity type; a
duplicate capability id across experts; an expert bound to an unregistered model; a capability with
neither description nor examples; retrieval from a domain the pack does not define.
