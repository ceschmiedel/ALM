# Governance, sovereignty and the audit trail

Where ALM really shines is not cost, it is sovereignty. With experts and their models running
on-premise or in the client's VPC, sensitive data does not leave the environment — which turns NIS2,
EU AI Act and LGPD requirements from an obstacle into an architectural feature.

## IBAC governs agents, not only users

In a federation, access control is per-*agent*. An expert in one domain does not see another
domain's data without an explicit grant, and every access is evaluated and recorded. That contains
the blast radius and produces the trail auditors ask for.

Evaluation is **deterministic**. A context read happens hundreds of times per session; a policy
engine needing an LLM call per fragment would be neither affordable nor reproducible. Identical
inputs always produce an identical decision — which is what an auditor is really asking about.

## Policies

```yaml
policies:
  - id: legal-no-finance-corpus
    description: The contracts expert may not read finance-owned context.
    effect: deny            # deny | allow | require_approval
    priority: 200
    evaluation_points: [context_read]
    subjects: [contracts-expert]
    resources: ["domain:finance"]
    actions: [read]

  - id: restricted-needs-claim
    effect: deny
    priority: 300
    resources: ["sensitivity:restricted"]
    unless_claims: [risk.incident_review]
```

### Selectors

| Selector | Matches |
|---|---|
| `*` | Anything |
| `expert:<id>` / `<id>` | A specific Expert Agent |
| `domain:<name>` | Everything belonging to a domain |
| `entity:<Type>` | Anything annotated with that ontology entity |
| `sensitivity:<level>` | That level **and everything stricter** |
| `capability:<id>` · `model:<id>` | A specific capability or model |

Deny wins. Within the same effect, higher `priority` wins. That ordering is what makes a broad allow
safe to write next to a narrow deny.

### Evaluation points

`intent_admission` · `routing` · `context_read` · `expert_execution` · `arbitration` ·
`artifact_emission`

The last one matters more than it looks: without an emission gate, an expert correctly denied a
fragment could still have its conclusions summarised into an answer that reaches an unauthorised
principal.

## The audit log

Every decision is recorded, allow and deny alike. Recording only denials would make the log useless
for the question auditors actually ask — not "what was blocked?" but "what did this agent see, and
under what authority?"

```bash
alm audit tail --session ses_a1b2c3
alm audit summary --hours 24
alm audit export > audit.jsonl        # JSON Lines, for an external SIEM
```

The log is append-only by contract: there is no update or delete path in the module. Audit writes
never raise — governance must not be able to break a run.

## Sensitivity classification

A chunk inherits the strictest sensitivity among the entities **of its own domain** that it
mentions. Cross-domain entity tags still help retrieval but never reclassify a document — otherwise
one keyword collision between domains would silently lock an expert out of its own corpus.
