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

The engine hands a request to a backend. `ALM_DECISION_BACKEND` picks the default.

### `generative` (default)

Any registered chat model, asked for JSON and read back tolerantly. It works with every serving
backend. Its confidences are the model *reporting* a number, it returns no distribution, and it
pays for every output token.

### `readout`

The model is never asked to write. The options are laid out behind single-token labels (`A`, `B`,
… for a choice; `0`, `1`, … for a score; `Yes`/`No`), one forward pass runs, and the probability
of each label is read from the next-token distribution. Every answer therefore carries a real
distribution — `probabilities` — and no `reason`: a readout does not explain itself. The audit
trail is the question, the state and the distribution.

Two corrections are applied:

- **Order debiasing.** Models favour positions. Each choice is shown in up to
  `ALM_DECISION_PERMUTATIONS` cyclic shifts and the per-option log-probabilities are averaged,
  which cancels a position bias that is additive in logit space. A yes/no question is asked in
  both phrasings. Each shift is one more forward pass, so this trades latency for robustness.
- **Temperature.** One scalar per question type (`ALM_DECISION_TEMPERATURE_*`) rescales the
  distribution. It is meant to be fitted on labelled data; at the default of 1.0 the result is
  marked `calibrated=False`.

Readout needs to see the next-token distribution, which depends on how the model is served:

| Serving backend | Readout | How |
|---|---|---|
| `transformers` / `local` | exact | Full distribution, in process. Shares the base-model pool, so a LoRA decision adapter attaches to an already-loaded base |
| `vllm`, `openai_compat`, `openai` | top 20 | `logprobs` on chat completions. A label outside the listed candidates gets the lowest listed value |
| `ollama`, `anthropic`, `google` | no | The API returns text only |

Limits: a single choice covers at most 26 options and a score at most 10 levels, because each
needs its own single-token label. A `multi` choice has no such limit — each option becomes its own
yes/no judgment, at two forward passes per option.

Where readout cannot serve a request — a text-only model, too many options — the engine falls back
to `generative`, so turning readout on never removes a judgment the federation could make before.

`ALM_DECISION_MODEL` names the dedicated decision model: when set, every readout decision goes to
it, whatever tier the caller asked for.

```bash
declm model add decider --tier slm --backend transformers --model Qwen/Qwen2.5-1.5B-Instruct
export ALM_DECISION_BACKEND=readout ALM_DECISION_MODEL=decider
```

## Serving decisions: `POST /v1/systemone`

`declm serve` answers the request shape hosted decision models use, so clients and benchmarks written
for it can point at an DecLM server:

```bash
curl -s localhost:8800/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Refund not received after 30 days",
  "questions": {
    "queue":  {"type": "choice", "instructions": "Which queue handles this?",
               "criteria": {"billing": "payments and refunds", "support": "product issues"}},
    "urgent": {"type": "noul",   "instructions": "It needs a same-day response."},
    "impact": {"type": "score",  "instructions": "Customer impact?",
               "criteria": ["none", "minor", "major"]}
  }
}'
```

`criteria` is label → description for a choice, an ordered list of levels for a score, and an
optional `{"true": …, "false": …}` for a yes/no. The response carries one answer per question id
(`choice` + `probabilities`, `score` + `probabilities` + `legend`, or `noul`), plus `model`,
`method`, `calibrated` and token `usage`.

## Measuring: `declm decide bench`

A decision model is only as useful as its probabilities are honest, so the benchmark reports
calibration next to accuracy.

```bash
declm decide bench decisions.jsonl --model decider --backend readout
```

| Metric | Meaning |
|---|---|
| accuracy | The most probable label is the expected one. An item the backend failed on counts as wrong |
| majority | Accuracy of always answering the most common label — the floor to beat |
| Brier | Squared error of the whole distribution, summed over labels. 0 is perfect; a coin flip over two labels is 0.5 |
| ECE | Gap between stated confidence and accuracy, over ten equal-width confidence bins. 0.10 means confidence is off by ten points on average |

Items use the JevBench record shape, one JSON object per line:

```json
{"id": "refund-01", "state": "…", "expected": "no",
 "question": {"type": "noul", "instructions": "Is the refund permitted?"}}
```

`expected` is a label for a choice, a level index for a score, and `yes`/`no` for a yes/no.
Brier and ECE need a distribution, so they are only reported for the readout backend.

Items may carry a `family` (reported separately) and a `split`: `dev` items are for fitting
temperatures and thresholds, `test` items for reporting. `--split test` keeps the two apart.

### A benchmark in Portuguese

The public decision-model benchmarks are English-only. `packs/meridiano-financeiro/decisions/fpa.jsonl`
is a Brazilian Portuguese set of 265 decisions over the Grupo Meridiano financial model, in ten
families: year-over-year comparison, the leading unit on a metric, growth bands, sales-target
attainment, a discount policy and its approval ladder, cash-conversion cycle, quarterly trend,
scenario counts and routing a question to an analysis area.

```bash
declm decide bench packs/meridiano-financeiro/decisions/fpa.jsonl --split test --model decider --backend readout
```

Every expected answer is **computed from the workbook**, not judged by a model, and every family is
balanced across its labels, so always giving the commonest answer scores at chance. The state holds
the inputs to a judgment — two revenues, a price and a cost — never the derived figure the question
turns on. Regenerate it with `python packs/meridiano-financeiro/tools/build_decisions.py`.

## Training data: `declm decide dataset build`

A decision model is trained on labelled decisions. `declm decide dataset build` assembles them into
`train.jsonl`, `dev.jsonl` and a `manifest.json`:

```bash
declm decide dataset sources
declm decide dataset build --out .alm/datasets/decisions-v0 \
  --n-per-source 1000 --n-synthetic 3000 \
  --exclude-bench packs/meridiano-financeiro/decisions/fpa.jsonl
```

Two kinds of source, and **no model-labelled data** — in particular, no output of a hosted decision
model is ever used as a label:

- **Public datasets whose human labels already are decisions**, recast as typed questions. The
  first set is Portuguese: MASSIVE intents, and the pt-BR ExtraGLUE translations of BoolQ, QNLI,
  COPA, WNLI, RTE and SST-2. Each is pinned to a commit and downloaded once into
  `$ALM_HOME/datasets-cache`; only the *train* split is read.
- **Rule-based generators** in Brazilian Portuguese (`alm.decision.synth_pt`): comparison, change
  bands, stock reorder, expense approval ladder, SLA deadlines, multi-condition eligibility,
  counting, extremes, trends and routing. Each draws random facts, states a rule, and computes the
  label. They deliberately use other domains than the Meridiano benchmark — logistics, HR, support
  — so the benchmark measures whether the *skill* transfers, not whether the data was seen.

Choice questions are augmented: options are shuffled, and sometimes a "none of the above" option is
added — as the right answer with the true option removed, and as a wrong alternative with it kept —
so that wording cannot become a shortcut.

### Licences

Each source carries the licence class of its underlying data, and the build takes a policy:

| Class | Sources | Default |
|---|---|---|
| permissive | MASSIVE (CC BY 4.0), COPA (BSD), WNLI (CC BY 4.0), the generators | included |
| share-alike | BoolQ (CC BY-SA 3.0), QNLI from SQuAD (CC BY-SA 4.0) | included; `--only-permissive` leaves them out |
| unstated | RTE, SST-2 | left out; `--allow-unstated` includes them |

Whether share-alike terms reach a model trained on the data is a legal question this tool does not
settle, and the classes above come from the dataset cards and the original releases — have them
reviewed before shipping a model commercially. The manifest records every source, revision and
licence that went into a build.

### Record format

A record is the `/v1/systemone` request with a `label` on each question — the option name for a
choice, the level index for a score, `true`/`false` for a yes/no:

```json
{"state": "A cobrança de novembro veio com valor errado.",
 "questions": {"fila": {"type": "choice", "instructions": "Qual equipe deve tratar esta mensagem?",
                        "criteria": {"financeiro": "Cobranças, boletos…", "logistica": "Entregas…"},
                        "label": "financeiro"}},
 "_meta": {"source": "synth_pt/roteamento", "lang": "pt-BR", "licence_class": "permissive"}}
```

This is the labelled-request shape the open decision-model trainers read, and `declm decide bench`
accepts the same files, so the dev split doubles as an in-distribution benchmark.

