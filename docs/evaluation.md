# Evaluation

> The ALM thesis is falsifiable and should be treated as such. Without an evaluation harness, ALM is
> faith; with one, it is engineering.

## The controlled experiment

Two arms, same battery of tasks:

- **(a) baseline** — the monolithic frontier model with a good prompt and RAG over the whole corpus;
- **(b) federation** — routing, per-expert scoped retrieval, arbitration, synthesis.

The baseline is deliberately a *good* one. A straw man would make the comparison worthless.

```bash
alm eval run --pack packs/my-domain --domain legal --compare
```

## The six metrics

| Metric | Why it exists |
|---|---|
| **domain accuracy** | Quality on the real task against a gold standard. The sovereign metric. |
| **cost per query** | Total infrastructure cost per answer, not a token price. |
| **latency p95** | The tail is what users feel; the mean hides it. |
| **fallback rate** | Expert coverage. Reported, not scored — the baseline has no fallback. |
| **consistency** | Behavioural variation. A federation-specific risk a monolith does not have. |
| **auditability** | Trail completeness. Where the federation wins by construction — and still measured. |

The federation must win **on the set**, not on one number. `compare()` will return a verdict of
`baseline` or `inconclusive`, and the recommendation says plainly to keep those agents as RAG over
the shared model. A harness that only ever reported "the federation wins" would be worth nothing.

## Datasets

One JSON object per line, so a domain expert can author and review the set in a text editor and diff
it in version control.

```json
{"id": "legal-001", "domain": "legal", "question": "What does clause 7.2 cap liability at?",
 "expected": "Total fees paid in the twelve months preceding the claim.",
 "grader": "keywords", "keywords": ["twelve", "fees"], "min_keyword_ratio": 0.5}
```

| Grader | Behaviour |
|---|---|
| `keywords` | Fraction of required keywords present; accent- and case-insensitive |
| `numeric` | Closest number within relative `tolerance` |
| `exact` / `contains` | Normalised string comparison |
| `rubric` | LLM-graded; degrades to `keywords` with a stated reason if no orchestrator is configured |

Deterministic graders are preferred: they cost nothing, they are reproducible, and a regression is
attributable to the system rather than to a judging model that drifted.

## Calibration

Evaluation produces labelled outcomes, which is exactly what confidence calibration needs:

```bash
alm eval calibrate --pack packs/my-domain
```

fits Platt scaling per expert and reports the Brier score before and after. A fit that makes
predictions *worse*, or a degenerate set (all correct, all wrong), leaves the expert uncalibrated —
`fitted` is explicit, never inferred from sample count.

Until an expert is calibrated, arbitration discounts its confidence toward the prior rather than
trusting a number that has never been checked.

## Reading the demo pack's numbers

The bundled demo runs both arms on the deterministic extractive backend. That removes exactly what a
federation buys — in-domain reasoning — while keeping its overhead, so the federation loses on
accuracy there. Auditability (1.00 vs 0.00) is the structural difference that still shows through.
Point the pack at real models before drawing conclusions about the thesis.
