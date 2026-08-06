# Architecture

ALM organises a federation into five functional layers plus two cross-cutting ones. Each maps to a
Python package, and each boundary is a real interface rather than a naming convention.

```
   intent ──► L1 alm.protocol ──► L2 alm.router ──► L4 alm.orchestration
                                                          │
                                                          ▼
   answer ◄── L5 alm.arbitration ◄────────────── L3 alm.experts + alm.models

   CX  alm.graph + alm.cmrag   feeds routing, execution and synthesis
   GV  alm.governance          no context access happens outside it
```

## L1 · Interface and ingestion — `alm.protocol`

Layer 1 *is* the [Liquid Interface Protocol](https://github.com/draiven-io/agentic-bus). The
envelope in `alm/protocol/envelope.py` is field-for-field compatible with the Agentic Bus, so a
federation can join an existing bus without a translation layer.

`TaskEnvelope` is what the rest of the system consumes: intent text, context, ontology entity
references, and the governance identity (`principal`, `claims`, `tenant_id`) that IBAC evaluates
every later access against.

## L2 · Cognitive router — `alm.router`

The router answers three questions: what does this decompose into, which Expert Agent resolves each
part, and with what confidence. An error here contaminates everything downstream — a subtask sent to
the wrong specialist cannot be rescued by a good specialist.

Routing is **declarative**. Capabilities live as nodes in the Context Graph with edges to the
ontology entity types they operate on, and the router queries them. Adding an Expert Agent is adding
a node; the router is never rewritten. That property is what makes routing an evolving asset rather
than a thicket of conditionals.

Three complementary signals are blended, because each fails differently:

| Signal | Finds | Blind to |
|---|---|---|
| semantic | paraphrase | identifiers, clause numbers |
| lexical | domain jargon, exact references | synonyms |
| entity | what the request is *about* | requests that name no entity |

Only the signals actually available are blended — a request that names no ontology entity has that
signal normalised away rather than scored as zero, because treating silence as evidence *against*
every capability would push the whole federation into fallback. Corroboration is then rewarded: two
independent signals agreeing is stronger than one signal shouting.

See [routing.md](routing.md) for the escalation logic.

## L3 · Expert federation — `alm.experts`, `alm.models`

An Expert Agent is configuration, not code: a YAML file declaring its capabilities, its model
binding, its retrieval scope, its output schema and its verifier rules. Executing a subtask is five
steps in a fixed order — governance, retrieval, generation, structuring and calibration,
verification.

Model tiers and multi-adapter serving are in [serving.md](serving.md).

## L4 · Dependency orchestration — `alm.orchestration`

The plan becomes a DAG where nodes are expert invocations and edges are data dependencies.
Independent nodes run concurrently; chained nodes wait, and the predecessor's **structured** output
becomes the successor's input.

Execution order is never hard-coded. It is derived from the graph: explicit `depends_on` edges, plus
inference from `produces`/`operates_on` overlap — a capability reading an entity type another
capability writes must run after it.

A failed step does not fail the plan. Its dependents are skipped with a recorded reason, independent
branches continue, and arbitration receives a partial but fully attributed set of answers.

## L5 · Arbitration and synthesis — `alm.arbitration`

In a monolith, contradictions resolve implicitly inside a latent space. In a federation two experts
can reach incompatible conclusions and someone must decide explicitly. See
[arbitration.md](arbitration.md).

## CX · Context and memory — `alm.graph`, `alm.cmrag`

The Context Graph is the federation's cortex *and* its whiteboard: it holds the ontology, capability
declarations, authority weights, and the intermediate state of a running plan. Nothing is hidden in
process memory, so any step can read what earlier steps produced and the audit trail is complete
without a separate logging pass.

CMRAG is retrieval with three constraints applied before similarity is considered: domain scope,
ontology scope, and governance. See [governance.md](governance.md).

## GV · Governance — `alm.governance`

IBAC governs each *agent*, not only each user. Evaluation is deterministic because a context read
happens hundreds of times per session and a policy engine needing an LLM call per fragment would be
neither affordable nor reproducible.

## Where the value accumulates

The models are replaceable. The Context Graph that routes them, the memory that connects them and
the governance that makes them auditable are not. If routing and synthesis were generic, the
federation would not survive contact with an ever-cheaper LLM; because they are proprietary and
tuned to a client's domain, they compound.
