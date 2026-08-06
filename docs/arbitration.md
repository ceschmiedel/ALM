# Arbitration and synthesis

Making explicit what a monolith does implicitly is simultaneously the largest engineering cost of a
federation and its largest advantage. A monolith cannot show how it decided; a federation can.

## Detecting a conflict

Detection is deliberately conservative — it flags disagreements it can point at concretely:

- the same structured field with incompatible values (numeric beyond tolerance, opposite booleans);
- opposite verdicts on a closed-vocabulary field (`severity`, `enforceable`, `compliant`);
- opposite positions stated in prose when no schema was used.

Differently-worded narrative fields are **not** a conflict. A false conflict costs a judge call and
puts a fabricated disagreement into the audit trail.

> A number is only a number when the value *is* one. "Clause 7.2 caps liability" and "clause 9.1
> governs termination" are not a numeric disagreement about 7.2 versus 9.1.

## The strategy chain

Applied in order, cheapest and most deterministic first.

| Strategy | Mechanism | When it decides |
|---|---|---|
| **verifier** | Deterministic domain rules | One side fails its own domain's checks |
| **confidence** | Calibrated confidence | The margin exceeds `ALM_CONFLICT_CONFIDENCE_MARGIN` |
| **authority** | `has_authority_over` weights from the graph | The primary domain's expert outweighs the other |
| **judge** | Orchestrator LLM with both answers and their evidence | Nothing above resolved it |

**Uncalibrated confidence never wins an argument.** A model that is confident and wrong is worse
than one that is uncertain and honest, so `ConfidenceStrategy` declines rather than acting on a
number nobody validated. Run `alm eval calibrate` to fit Platt scaling from labelled outcomes; until
then an expert's score is shrunk toward the prior and reported as `uncalibrated` in the trace.

Authority weights live in the graph, not in code, so they are inspectable and tunable per tenant.

## Synthesis

Two paths, both producing the same provenance:

- **model** — the orchestrator weaves contributions into one coherent answer;
- **deterministic** — contributions ordered by arbitration weight and assembled into sections.

The deterministic path always exists, which means the federation still answers when the frontier
model is unreachable.

A single accepted answer is passed through unchanged: running it through a model would only risk
paraphrasing away the specialist's precision.

## The record

```
contracts-expert (legal) contributed with confidence 0.82, weight 0.82
finance-expert (finance) contributed with confidence 0.71, weight 0.25
  — superseded by contracts-expert: authority: 'contracts-expert' holds
    authority 1.00 over domain 'legal' against 0.30
```

Unresolved conflicts are reported, not hidden, and they reduce the confidence of the synthesised
answer. An answer built on an unsettled disagreement genuinely deserves less trust, and saying so is
more useful than a reassuring number.
