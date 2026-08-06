# Routing, confidence and fallback

## The pipeline

1. **Classify** — map the request to domains. The graph path is free and deterministic; it escalates
   to a micro-SLM only when the free signal is genuinely ambiguous.
2. **Decompose** — capability-driven (no model call) for the common case; planner-driven
   (orchestrator) for composite or ambiguous requests.
3. **Match** — every proposed subtask is matched back onto a declared capability. The model may
   suggest structure; the graph decides who executes.
4. **Decide** — route, or escalate.

## Confidence

Plan confidence is **weakest-link dominated**: one misrouted step is enough to make the synthesised
answer wrong, so a plan is only as trustworthy as its worst assignment.

Classification enters as *corroboration* — a saturating boost that can raise a decent plan but never
rescue a bad one:

```
core        = 0.6 · weakest + 0.4 · mean
confidence  = core + (1 − core) · 0.35 · classification_confidence
penalty     = 1 − 0.5 · unmatched / total
```

Classification confidence itself is `best · (0.65 + 0.35 · margin)`. The margin **scales** the best
score rather than being added to it — an uncontested weak match is not evidence, it is the absence
of evidence, and it must escalate.

## Which experts are invited

Only capabilities within `RELATIVE_MATCH_BAND` (0.55) of the best match join the plan. A marginal
third specialist adds cost and noise, and — because confidence is weakest-link — would drag an
otherwise sound plan below the threshold. *Not inviting* an expert is different from failing to
find one.

## Fallback is design, not concession

Below `ALM_ROUTER_CONFIDENCE_THRESHOLD`, or when nothing matches a declared capability, the request
escalates to the orchestrator. This guarantees the federation is **never worse than the monolithic
baseline**: at worst it hands the task back to the same frontier model the baseline would have used.

The fallback rate is a **health metric**:

| Fallback rate | Reading |
|---|---|
| Too high | Insufficient expert coverage — capabilities under-declare their scope, or a domain is missing |
| Too low | The router may be gambling on cases it should escalate |

`alm explain "<question>"` shows every signal without executing anything, so a bad route is
diagnosable instead of mysterious.

## Diagnosing a bad route

A capability that under-declares its own scope routes to fallback and *looks* like a model failure
when it is really a declaration failure. Check, in order:

1. `alm explain` — are the entity types detected at all? If not, the ontology keywords are too narrow.
2. Is the lexical score near zero? The capability's `keywords`/`examples` do not cover the corpus's
   vocabulary. Examples matter most: they are written in the phrasing real requests use.
3. Is the semantic score flat across capabilities? No embedder is configured — `alm doctor` will say so.
