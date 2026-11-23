# Typed decisions

Several layers of the federation do not need prose from a model. They need a judgment that code
will consume: which domain a request belongs to, which expert prevails in a conflict, how good an
answer is, whether an intent violates a policy. `alm.decision` gives those judgments one contract.

## The contract

A `DecisionRequest` keeps the facts and the judgments apart: `state` is what is being judged,
`questions` is what is asked about it. Several independent questions go out in one call.

| Primitive | Asks | Answer |
|---|---|---|
| `Choice` | Which option? One label from a closed set, or a ranked subset with `multi=True` | `choice`, `selected`, `confidence` |
| `Score` | Where on an ordered scale? | `score` (0 … levels − 1), `normalised` (0 … 1) |
| `Noul` | Is this true? | `probability` (0 … 1) |

```python
from alm.decision import Choice, DecisionEngine, DecisionRequest, Noul

result = await DecisionEngine(serving).decide(
    DecisionRequest(
        state={"ticket": "Refund not received after 30 days", "plan": "enterprise"},
        questions={
            "queue": Choice(
                instructions="Which queue handles this ticket?",
                options={"billing": "payments and refunds", "support": "product issues"},
            ),
            "urgent": Noul(instructions="The ticket needs a same-day response."),
        },
    ),
    tier="micro_slm",
)
result.choice("queue").choice      # "billing"
result.noul("urgent").probability  # 0.8
```

An answer the backend did not give, or gave with a label outside the closed set, comes back as the
empty default (`choice == ""`, `confidence == 0.0`) rather than as an exception. A confidence with
no valid label attached is discarded: confident about *what*?

## Where the federation uses it

| Layer | Judgment | Primitive |
|---|---|---|
| L2 `router.classifier` | Which domains does the request belong to? | multi `Choice` |
| L5 `arbitration.judge` | Which expert's conclusion prevails? | `Choice` |
| `eval.rubric` | How good is the answer, and is it correct? | `Score` + `Noul` |
| GV `ibac.semantic` | Does the intent violate a DENY policy, and which? | `Noul` + multi `Choice` |

The semantic IBAC pass keeps its original guarantee: it can only add a denial on top of the
deterministic decision, never overturn one. A probabilistic judgment is not a sole access control.

## Backends

The engine hands a request to a `DecisionBackend`. One ships today:

- **`generative`** — any registered chat model, asked for JSON and read back tolerantly. This is
  the fallback every installation already has. Its confidences are the model *reporting* a number,
  so results carry `calibrated=False`.

The indirection is the point. A backend that reads a real probability distribution from a local
model, or one that calls a hosted decision model, implements the same `decide()` and every
judgment above moves to it at once — measurable against the generative fallback with
`alm eval run`.
