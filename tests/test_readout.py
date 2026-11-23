"""Readout decisions: probabilities read from the next token, debiased and scored."""

from __future__ import annotations

import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from alm.api.app import create_app, set_runtime
from alm.decision import Choice, DecisionEngine, DecisionRequest, Noul, Score
from alm.decision.bench import (
    BenchItem,
    brier,
    expected_calibration_error,
    load_items,
    run_bench,
)
from alm.decision.providers import OpenAICompatLogprobs
from alm.decision.readout import (
    LETTERS,
    LogprobBatch,
    ReadoutConfig,
    ReadoutDecisionBackend,
    ReadoutPrompt,
    ReadoutUnsupportedError,
    cyclic_shifts,
    marginalise,
    normalise,
)
from alm.decision.wire import request_from_wire
from alm.models.serving import ServingRouter
from alm.models.spec import ModelSpec
from alm.models.tiers import ModelTier


class FakeProvider:
    """Scores labels by looking the option text up in ``content``, plus a position bias.

    This is the additive model the debiasing assumes: logit = content + bias.
    """

    def __init__(self, content: dict[str, float], *, position_bias: float = 0.0) -> None:
        self.content = content
        self.position_bias = position_bias
        self.prompts: list[ReadoutPrompt] = []

    def _score(self, prompt: ReadoutPrompt, position: int, label: str) -> float:
        bias = -self.position_bias * position
        if label in {"Yes", "No"}:
            key = next((k for k in self.content if k in prompt.user), None)
            value = self.content.get(key, 0.0) if key else 0.0
            return (value if label == "Yes" else -value) + bias
        line = next(ln for ln in prompt.user.splitlines() if ln.startswith(f"{label}. "))
        return self.content.get(line.split(". ", 1)[1].split(" — ")[0], 0.0) + bias

    async def next_token_logprobs(self, prompts):
        self.prompts.extend(prompts)
        return LogprobBatch(
            logprobs=[
                [self._score(p, i, label) for i, label in enumerate(p.labels)] for p in prompts
            ],
            prompt_tokens=10 * len(prompts),
        )


def _serving(registry, backend="scripted") -> ServingRouter:
    registry.register(ModelSpec(model_id="decider", tier=ModelTier.SLM, backend=backend))
    return ServingRouter(registry)


def _backend(registry, provider, **config) -> ReadoutDecisionBackend:
    return ReadoutDecisionBackend(
        _serving(registry), provider=provider, config=ReadoutConfig(**config)
    )


_QUEUES = Choice(
    instructions="Which queue?",
    options={"billing": "payments", "support": "product issues", "sales": "new deals"},
)


# -- distributions -----------------------------------------------------------


def test_normalise_is_a_softmax_and_temperature_flattens_it():
    sharp = normalise([0.0, math.log(3.0)])
    assert sharp == pytest.approx([0.25, 0.75])
    flat = normalise([0.0, math.log(3.0)], temperature=4.0)
    assert 0.5 < flat[1] < sharp[1]


def test_cyclic_shifts_put_each_option_in_spread_positions():
    assert cyclic_shifts(3, 4) == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    assert len(cyclic_shifts(8, 4)) == 4
    assert cyclic_shifts(5, 1) == [[0, 1, 2, 3, 4]]


def test_marginalising_over_all_shifts_removes_an_additive_position_bias():
    content = [0.2, 1.0, -0.5]
    bias = [0.0, -1.5, -3.0]
    perms = cyclic_shifts(3, 3)
    by_position = [normalise([content[o] + bias[j] for j, o in enumerate(perm)]) for perm in perms]
    assert marginalise(by_position, perms) == pytest.approx(normalise(content))


# -- answers -----------------------------------------------------------------


async def test_choice_returns_a_full_distribution_over_the_options(registry):
    provider = FakeProvider({"billing": 2.0, "support": 0.0, "sales": 0.0})
    result = await _backend(registry, provider).decide(
        DecisionRequest(state="refund please", questions={"q": _QUEUES})
    )
    answer = result.choice("q")
    assert answer.choice == "billing"
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert answer.confidence == answer.probabilities["billing"]
    assert result.method == "readout"
    assert result.prompt_tokens == 30


async def test_position_bias_flips_a_single_pass_but_not_the_debiased_answer(registry):
    # "sales" is the right answer, but it is listed last and the model favours early positions.
    content = {"billing": 0.0, "support": 0.0, "sales": 1.0}
    biased = FakeProvider(content, position_bias=1.0)

    single = await _backend(registry, biased, max_permutations=1).decide(
        DecisionRequest(state="s", questions={"q": _QUEUES})
    )
    assert single.choice("q").choice == "billing"

    debiased = await ReadoutDecisionBackend(
        ServingRouter(registry), provider=biased, config=ReadoutConfig(max_permutations=3)
    ).decide(DecisionRequest(state="s", questions={"q": _QUEUES}), model_id="decider")
    assert debiased.choice("q").choice == "sales"


async def test_noul_is_asked_in_both_phrasings_and_returns_p_yes(registry):
    provider = FakeProvider({"urgent": 1.0})
    result = await _backend(registry, provider).decide(
        DecisionRequest(state="urgent ticket", questions={"q": Noul(instructions="Is it urgent?")})
    )
    assert [p.user.splitlines()[-1] for p in provider.prompts] == [
        "Answer Yes or No.",
        "Answer No or Yes.",
    ]
    assert result.noul("q").probability > 0.5
    assert result.noul("q").value is True


async def test_score_is_the_expectation_over_levels(registry):
    class Flat:
        async def next_token_logprobs(self, prompts):
            assert prompts[0].labels == ("0", "1", "2")
            return LogprobBatch(logprobs=[[math.log(0.1), math.log(0.2), math.log(0.7)]])

    result = await _backend(registry, Flat()).decide(
        DecisionRequest(
            state="s", questions={"q": Score(instructions="How bad?", levels=["low", "mid", "high"])}
        )
    )
    answer = result.score("q")
    assert answer.score == pytest.approx(1.6)
    assert answer.normalised == pytest.approx(0.8)
    assert answer.probabilities == pytest.approx([0.1, 0.2, 0.7])
    assert answer.confidence == pytest.approx(0.7)


async def test_multi_choice_judges_each_option_on_its_own(registry):
    provider = FakeProvider({"billing": 2.0, "support": 1.0, "sales": -2.0})
    multi = Choice(instructions="Which apply?", options=_QUEUES.options, multi=True)
    result = await _backend(registry, provider).decide(
        DecisionRequest(state="s", questions={"q": multi})
    )
    answer = result.choice("q")
    assert answer.selected == ["billing", "support"]
    assert answer.probabilities["sales"] < 0.5
    assert len(provider.prompts) == 6


async def test_temperature_changes_confidence_and_marks_the_result_calibrated(registry):
    provider = FakeProvider({"billing": 2.0, "support": 0.0, "sales": 0.0})
    request = DecisionRequest(state="s", questions={"q": _QUEUES})
    raw = await _backend(registry, provider).decide(request)
    assert raw.calibrated is False

    cooled = await ReadoutDecisionBackend(
        ServingRouter(registry), provider=provider, config=ReadoutConfig(temperature={"choice": 3.0})
    ).decide(request, model_id="decider")
    assert cooled.calibrated is True
    assert cooled.choice("q").choice == "billing"
    assert cooled.choice("q").confidence < raw.choice("q").confidence


async def test_too_many_options_is_unsupported(registry):
    wide = Choice(instructions="?", options={f"opt{i}": "" for i in range(len(LETTERS) + 1)})
    with pytest.raises(ReadoutUnsupportedError):
        await _backend(registry, FakeProvider({})).decide(
            DecisionRequest(state="s", questions={"q": wide})
        )


# -- engine ------------------------------------------------------------------


async def test_readout_mode_falls_back_to_generative_for_a_text_only_backend(registry):
    registry.register(
        ModelSpec(
            model_id="scripted",
            tier=ModelTier.SLM,
            backend="scripted",
            params={"responses": [{"match": ".*", "response": {"q": {"choice": "support"}}}]},
        )
    )
    engine = DecisionEngine(ServingRouter(registry), mode="readout")
    result = await engine.decide(DecisionRequest(state="s", questions={"q": _QUEUES}), tier="slm")
    assert result.method == "generative"
    assert result.choice("q").choice == "support"


# -- OpenAI-compatible provider ----------------------------------------------


def _logprob_response(candidates: dict[str, float]) -> dict:
    return {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": next(iter(candidates)),
                            "top_logprobs": [
                                {"token": t, "logprob": lp} for t, lp in candidates.items()
                            ],
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 42},
    }


def _http_provider(handler) -> OpenAICompatLogprobs:
    provider = OpenAICompatLogprobs(
        ModelSpec(model_id="m", backend="vllm", model_name="served", endpoint="http://x/v1")
    )
    provider.backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


async def test_http_provider_asks_for_one_token_with_logprobs_and_merges_token_variants():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json=_logprob_response({"A": math.log(0.5), " A": math.log(0.2), "B": math.log(0.1)}),
        )

    batch = await _http_provider(handler).next_token_logprobs(
        [ReadoutPrompt("sys", "user", ("A", "B", "C"))]
    )
    assert seen["max_tokens"] == 1
    assert seen["logprobs"] is True
    assert seen["top_logprobs"] == 20
    assert seen["model"] == "served"
    a, b, c = batch.logprobs[0]
    assert a == pytest.approx(math.log(0.7))
    assert b == pytest.approx(math.log(0.1))
    # "C" was not listed: it gets the lowest listed value, an upper bound.
    assert c == pytest.approx(math.log(0.1))
    assert batch.prompt_tokens == 42


async def test_http_provider_refuses_when_no_label_is_among_the_top_tokens():
    from alm.core.errors import BackendError

    provider = _http_provider(
        lambda request: httpx.Response(200, json=_logprob_response({"The": -0.1, "I": -2.0}))
    )
    with pytest.raises(BackendError):
        await provider.next_token_logprobs([ReadoutPrompt("sys", "user", ("A", "B"))])


async def test_http_provider_reports_an_endpoint_without_logprobs_as_unsupported():
    provider = _http_provider(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})
    )
    with pytest.raises(ReadoutUnsupportedError):
        await provider.next_token_logprobs([ReadoutPrompt("sys", "user", ("A", "B"))])


# -- wire format and benchmark -----------------------------------------------


def test_wire_request_maps_the_three_question_types():
    request = request_from_wire(
        {
            "state": {"ticket": 1},
            "questions": {
                "c": {"type": "choice", "instructions": "?", "criteria": {"a": "first", "b": ""}},
                "s": {"type": "score", "instructions": "?", "criteria": ["low", "high"]},
                "n": {"type": "noul", "instructions": "?", "criteria": {"true": "yes it is"}},
            },
        }
    )
    assert request.questions["c"].options == {"a": "first", "b": ""}
    assert request.questions["s"].levels == ["low", "high"]
    assert request.questions["n"].when_true == "yes it is"


@pytest.mark.parametrize(
    "payload",
    [
        {"state": "s", "questions": {}},
        {"state": "s", "questions": {"q": {"type": "choice", "instructions": "?"}}},
        {"state": "s", "questions": {"q": {"type": "ranking", "instructions": "?"}}},
    ],
)
def test_wire_request_rejects_malformed_questions(payload):
    with pytest.raises(ValueError):
        request_from_wire(payload)


def test_systemone_endpoint_answers_in_the_wire_shape(runtime, monkeypatch):
    provider = FakeProvider({"billing": 2.0, "urgent": 1.0})
    backend = ReadoutDecisionBackend(runtime.serving, provider=provider)
    monkeypatch.setattr("alm.decision.engine._readout_backend", lambda serving, settings: backend)
    monkeypatch.setenv("ALM_DECISION_BACKEND", "readout")
    from alm.config import reset_settings_cache

    reset_settings_cache()
    set_runtime(runtime)
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "state": "urgent: refund my billing error",
                "questions": {
                    "queue": {
                        "type": "choice",
                        "instructions": "Which queue?",
                        "criteria": {"billing": "payments", "support": "product"},
                    },
                    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
                    "impact": {"type": "score", "instructions": "Impact?", "criteria": ["a", "b"]},
                },
            },
        )
        bad = client.post("/v1/systemone", json={"state": "s", "questions": {}})
    set_runtime(None)

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "readout"
    assert body["answers"]["queue"]["choice"] == "billing"
    assert set(body["answers"]["queue"]["probabilities"]) == {"billing", "support"}
    assert 0.5 < body["answers"]["urgent"]["noul"] <= 1.0
    assert set(body["answers"]["impact"]["probabilities"]) == {"0", "1"}
    assert body["answers"]["impact"]["legend"] == {"0": "a", "1": "b"}
    assert body["usage"]["input_tokens"] > 0
    assert bad.status_code == 422


def test_brier_and_ece_match_hand_computed_values():
    assert brier({"yes": 0.8, "no": 0.2}, "yes", ["no", "yes"]) == pytest.approx(0.08)
    assert brier({"a": 1.0}, "b", ["a", "b", "c"]) == pytest.approx(2.0)
    # Ten predictions at 0.9 confidence, seven right: off by 0.2.
    pairs = [(0.9, True)] * 7 + [(0.9, False)] * 3
    assert expected_calibration_error(pairs) == pytest.approx(0.2)
    assert expected_calibration_error([]) == 0.0


async def test_bench_scores_accuracy_and_calibration(registry, tmp_path):
    path = tmp_path / "bench.jsonl"
    records = [
        {
            "id": "c1",
            "state": "refund",
            "expected": "billing",
            "question": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": {"billing": "payments", "support": "product"},
            },
        },
        {
            "id": "n1",
            "state": "urgent thing",
            "expected": "no",
            "question": {"type": "noul", "instructions": "Is it urgent?"},
        },
        {
            "id": "s1",
            "state": "x",
            "expected": 0,
            "question": {"type": "score", "instructions": "Impact?", "criteria": ["low", "high"]},
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in records))
    items = load_items(path)
    assert [i.labels for i in items] == [["billing", "support"], ["no", "yes"], ["0", "1"]]
    assert isinstance(items[0], BenchItem)

    provider = FakeProvider({"billing": 2.0, "urgent": 1.0, "low": 1.0})
    engine = DecisionEngine(ServingRouter(registry), backend=_backend(registry, provider))
    report = await run_bench(items, engine, model_id="decider")

    assert report.method == "readout"
    assert report.overall.n == 3
    # The choice and the score are right; the model says "urgent" where the truth is "no".
    assert report.overall.accuracy == pytest.approx(2 / 3, abs=1e-4)
    assert report.by_type["noul"].accuracy == 0.0
    assert report.overall.brier is not None
    assert report.overall.ece is not None
    assert report.overall.errors == 0


# -- in-process provider helpers ---------------------------------------------


class _Tokenizer:
    """Single characters and a few whole words are one token; anything else splits."""

    chat_template = None
    whole = {"Yes", "No", " Yes", " No", " A", " B"}

    def encode(self, text, add_special_tokens=False):
        if text in self.whole or len(text) == 1:
            return [sum(map(ord, text))]
        return [ord(c) for c in text]


def test_labels_must_map_to_distinct_single_tokens():
    from alm.decision.providers import label_token_ids

    tokenizer = _Tokenizer()
    assert len(set(label_token_ids(tokenizer, ("A", "B"), leading_space=False))) == 2
    # After a plain "Answer:" the label follows a space, so that variant is preferred.
    spaced = label_token_ids(tokenizer, ("A", "B"), leading_space=True)
    assert spaced == [tokenizer.encode(" A")[0], tokenizer.encode(" B")[0]]
    with pytest.raises(ReadoutUnsupportedError):
        label_token_ids(tokenizer, ("Maybe",), leading_space=False)


def test_prompts_fall_back_to_plain_text_without_a_chat_template():
    from alm.decision.providers import render_chat

    text, templated = render_chat(_Tokenizer(), ReadoutPrompt("sys", "user", ("A",)))
    assert templated is False
    assert text.endswith("Answer:")


# -- the Portuguese decision benchmark ---------------------------------------


def test_meridiano_decision_benchmark_is_well_formed_and_balanced():
    from collections import Counter
    from pathlib import Path

    from alm.decision.bench import question_from_wire

    path = Path(__file__).resolve().parent.parent / "packs/meridiano-financeiro/decisions/fpa.jsonl"
    items = load_items(path)
    assert len(items) >= 250
    assert len({item.id for item in items}) == len(items)
    assert {item.split for item in items} == {"dev", "test"}

    by_family: dict[str, list] = {}
    for item in items:
        question_from_wire(item.question)  # every question must be servable
        assert item.expected in item.labels
        by_family.setdefault(item.family, []).append(item)

    assert len(by_family) == 10
    for family, members in by_family.items():
        commonest = Counter(m.expected for m in members).most_common(1)[0][1]
        # No family can be passed by always giving its commonest answer.
        assert commonest / len(members) <= 0.5, family
