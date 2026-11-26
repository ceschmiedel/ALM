"""Readout backend — read the judgment from the model's next-token distribution.

A generative model asked "which option?" already holds the answer as a
probability over its next token.  Generating text and parsing it throws that
distribution away and pays for every output token.  The readout backend stops
one step earlier: it lays the options out behind single-token labels, runs one
forward pass, and reads the probability of each label.

Three things make the raw numbers usable:

* **Order debiasing.**  Models favour positions ("A" over "D").  Each choice is
  shown in several cyclic shifts and the per-option log-probabilities are
  averaged; where the bias is additive in logit space, that removes it exactly
  (Zheng et al., ICLR 2024).  Yes/no questions are asked in both phrasings.
* **A real distribution.**  Answers carry the probability of every option, so
  confidence gating and calibration have something to work with.
* **Temperature.**  One scalar per question type, fitted offline on labelled
  data, rescales the distribution.  It defaults to 1 — unfitted — and results
  say so through ``calibrated``.

What a readout cannot do is explain itself: answers carry no ``reason``.  The
audit trail is the question, the state and the distribution.

The technique follows the open decision-model work, in particular AnyJev
(Apache-2.0); the implementation here is DecLM's own.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from alm.core.errors import ALMError
from alm.core.jsonutil import dumps
from alm.core.telemetry import CallRecord, UsageMeter
from alm.decision.spec import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResult,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)
from alm.models.serving import ServingRouter
from alm.models.spec import ModelSpec
from alm.models.tiers import ModelTier

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
YES, NO = "Yes", "No"

#: Largest closed set a single readout can cover: one letter per option.
MAX_OPTIONS = len(LETTERS)
#: Largest scale: one digit per level.
MAX_LEVELS = 10

_SYSTEM = (
    "You are a decision function. You are given a state and one question. "
    "Reply with the answer label only: no words, no punctuation, no explanation."
)


class ReadoutUnsupportedError(ALMError):
    """This request cannot be served by readout; the caller should fall back."""


@dataclass(frozen=True)
class ReadoutPrompt:
    """One forward pass: a prompt and the labels whose next-token probability is read."""

    system: str
    user: str
    labels: tuple[str, ...]


@dataclass
class LogprobBatch:
    """Per prompt, the log-probability of each of its labels, in label order."""

    logprobs: list[list[float]]
    prompt_tokens: int = 0


class LogprobProvider(Protocol):
    """A model that exposes next-token log-probabilities for chosen labels.

    Values need not be normalised over the labels; the readout renormalises.
    """

    async def next_token_logprobs(self, prompts: Sequence[ReadoutPrompt]) -> LogprobBatch: ...


@dataclass(frozen=True)
class ReadoutConfig:
    #: Cyclic shifts shown per choice.  1 disables order debiasing.
    max_permutations: int = 4
    #: Softmax temperature per question type; 1.0 means "not fitted".
    temperature: dict[str, float] = field(default_factory=dict)

    def temperature_for(self, question_type: str) -> float:
        return max(float(self.temperature.get(question_type, 1.0)), 1e-3)

    @property
    def fitted(self) -> bool:
        return any(abs(t - 1.0) > 1e-9 for t in self.temperature.values())


# -- distributions -----------------------------------------------------------


def normalise(logprobs: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Softmax of ``logprobs / temperature`` over the labels."""
    scaled = [lp / temperature for lp in logprobs]
    top = max(scaled)
    weights = [math.exp(s - top) for s in scaled]
    total = sum(weights)
    return [w / total for w in weights]


def cyclic_shifts(k: int, limit: int) -> list[list[int]]:
    """Evenly spread cyclic shifts; ``perm[j]`` is the option shown at position ``j``."""
    count = max(1, min(k, limit))
    shifts = sorted({(i * k) // count for i in range(count)})
    return [[(j + s) % k for j in range(k)] for s in shifts]


def marginalise(by_position: Sequence[Sequence[float]], perms: Sequence[Sequence[int]]) -> list[float]:
    """Combine per-shift distributions into one distribution per option.

    Log-probabilities are averaged (a geometric mean) and renormalised: if the
    model's logit is ``content[option] + bias[position]``, the bias cancels.
    """
    k = len(perms[0])
    sums = [0.0] * k
    for distribution, perm in zip(by_position, perms, strict=True):
        for position, option in enumerate(perm):
            sums[option] += math.log(max(distribution[position], 1e-12))
    return normalise([s / len(perms) for s in sums])


# -- prompts -----------------------------------------------------------------


def _state_text(state: object) -> str:
    if isinstance(state, str):
        return state or "(empty)"
    return dumps(state, indent=2)


def _head(request: DecisionRequest, instructions: str) -> tuple[str, str]:
    system = f"{request.preamble.strip()}\n\n{_SYSTEM}" if request.preamble.strip() else _SYSTEM
    user = f"State:\n{_state_text(request.state)}\n\nQuestion: {instructions.strip()}\n"
    return system, user


def _option_text(label: str, description: str) -> str:
    return f"{label} — {description}" if description else label


def _noul_prompts(system: str, head: str, when_true: str = "", when_false: str = "") -> list[ReadoutPrompt]:
    meaning = ""
    if when_true:
        meaning += f"{YES} means: {when_true}\n"
    if when_false:
        meaning += f"{NO} means: {when_false}\n"
    return [
        ReadoutPrompt(system, f"{head}{meaning}Answer {a} or {b}.", (YES, NO))
        for a, b in ((YES, NO), (NO, YES))
    ]


@dataclass
class _Plan:
    """The forward passes one question needs, and how to fold them into an answer."""

    question: Choice | Score | Noul
    prompts: list[ReadoutPrompt]
    perms: list[list[int]] = field(default_factory=list)


def plan_question(request: DecisionRequest, question: Choice | Score | Noul, config: ReadoutConfig) -> _Plan:
    system, head = _head(request, question.instructions)

    if isinstance(question, Noul):
        return _Plan(question, _noul_prompts(system, head, question.when_true, question.when_false))

    if isinstance(question, Score):
        if len(question.levels) > MAX_LEVELS:
            raise ReadoutUnsupportedError(
                f"a readout score supports at most {MAX_LEVELS} levels, got {len(question.levels)}"
            )
        labels = tuple(str(i) for i in range(len(question.levels)))
        lines = "\n".join(f"{i}. {level}" for i, level in enumerate(question.levels))
        user = f"{head}Levels, lowest first:\n{lines}\nAnswer with the number only."
        return _Plan(question, [ReadoutPrompt(system, user, labels)])

    options = list(question.options.items())
    if question.multi:
        # Each option is its own yes/no judgment: a subset has no single softmax.
        prompts: list[ReadoutPrompt] = []
        for label, description in options:
            candidate = f"{head}Candidate: {_option_text(label, description)}\nDoes this candidate apply?\n"
            prompts.extend(_noul_prompts(system, candidate))
        return _Plan(question, prompts)

    if len(options) > MAX_OPTIONS:
        raise ReadoutUnsupportedError(
            f"a readout choice supports at most {MAX_OPTIONS} options, got {len(options)}"
        )
    perms = cyclic_shifts(len(options), config.max_permutations)
    labels = tuple(LETTERS[: len(options)])
    prompts = []
    for perm in perms:
        lines = "\n".join(
            f"{labels[position]}. {_option_text(*options[option])}"
            for position, option in enumerate(perm)
        )
        prompts.append(
            ReadoutPrompt(system, f"{head}Options:\n{lines}\nAnswer with the letter only.", labels)
        )
    return _Plan(question, prompts, perms)


# -- answers -----------------------------------------------------------------


def _p_yes(pair: Sequence[Sequence[float]], temperature: float) -> float:
    """P(yes) from the two phrasings of one yes/no question."""
    identity = [[0, 1], [0, 1]]
    return marginalise([normalise(lp, temperature) for lp in pair], identity)[0]


def fold(plan: _Plan, logprobs: Sequence[Sequence[float]], config: ReadoutConfig) -> Answer:
    """Turn the log-probabilities of a plan's prompts into its typed answer."""
    question = plan.question

    if isinstance(question, Noul):
        probability = _p_yes(logprobs, config.temperature_for("noul"))
        return NoulAnswer(probability=round(probability, 6))

    if isinstance(question, Score):
        distribution = normalise(logprobs[0], config.temperature_for("score"))
        expected = sum(level * p for level, p in enumerate(distribution))
        return ScoreAnswer(
            score=round(expected, 4),
            normalised=round(expected / question.top, 4),
            probabilities=[round(p, 6) for p in distribution],
            confidence=round(max(distribution), 6),
        )

    labels = list(question.options)
    if question.multi:
        temperature = config.temperature_for("noul")
        probabilities = {
            label: round(_p_yes(logprobs[2 * i : 2 * i + 2], temperature), 6)
            for i, label in enumerate(labels)
        }
        selected = sorted(
            (label for label, p in probabilities.items() if p >= 0.5),
            key=lambda label: probabilities[label],
            reverse=True,
        )
        return ChoiceAnswer(
            choice=selected[0] if selected else "",
            selected=selected,
            probabilities=probabilities,
            confidence=probabilities[selected[0]] if selected else 0.0,
        )

    temperature = config.temperature_for("choice")
    distribution = marginalise([normalise(lp, temperature) for lp in logprobs], plan.perms)
    best = max(range(len(labels)), key=distribution.__getitem__)
    return ChoiceAnswer(
        choice=labels[best],
        selected=[labels[best]],
        probabilities={label: round(p, 6) for label, p in zip(labels, distribution, strict=True)},
        confidence=round(distribution[best], 6),
    )


# -- backend -----------------------------------------------------------------


def provider_for(spec: ModelSpec) -> LogprobProvider:
    """The log-probability provider for a model, by its serving backend."""
    backend = spec.backend.strip().lower()
    if backend in {"transformers", "local"}:
        from alm.decision.providers import TransformersLogprobs

        return TransformersLogprobs(spec)
    if backend in {"vllm", "openai_compat", "openai"}:
        from alm.decision.providers import OpenAICompatLogprobs

        return OpenAICompatLogprobs(spec)
    raise ReadoutUnsupportedError(
        f"backend {spec.backend!r} of model {spec.model_id!r} does not expose "
        "next-token probabilities; readout needs transformers, vllm or an "
        "OpenAI-compatible server with logprobs"
    )


class ReadoutDecisionBackend:
    """Serves :class:`DecisionRequest` from next-token probabilities."""

    method = "readout"

    def __init__(
        self,
        serving: ServingRouter,
        *,
        config: ReadoutConfig | None = None,
        provider: LogprobProvider | None = None,
    ) -> None:
        self.serving = serving
        self.config = config or ReadoutConfig()
        self._provider = provider
        self._providers: dict[str, LogprobProvider] = {}

    def _provider_for(self, spec: ModelSpec) -> LogprobProvider:
        if self._provider is not None:
            return self._provider
        if spec.model_id not in self._providers:
            self._providers[spec.model_id] = provider_for(spec)
        return self._providers[spec.model_id]

    async def decide(
        self,
        request: DecisionRequest,
        *,
        tier: ModelTier | str | None = None,
        model_id: str = "",
        meter: UsageMeter | None = None,
        component: str = "",
    ) -> DecisionResult:
        spec = self.serving.registry.require(self.serving.resolve(model_id=model_id, tier=tier))
        provider = self._provider_for(spec)
        plans = {
            qid: plan_question(request, question, self.config)
            for qid, question in request.questions.items()
        }

        # Every forward pass of every question goes out as one batch.
        prompts = [prompt for plan in plans.values() for prompt in plan.prompts]
        started = time.perf_counter()
        batch = await provider.next_token_logprobs(prompts)
        latency_ms = (time.perf_counter() - started) * 1000.0

        answers: dict[str, Answer] = {}
        cursor = 0
        for qid, plan in plans.items():
            span = batch.logprobs[cursor : cursor + len(plan.prompts)]
            cursor += len(plan.prompts)
            answers[qid] = fold(plan, span, self.config)

        cost = spec.cost_for(batch.prompt_tokens, 0)
        if meter is not None:
            meter.record(
                CallRecord(
                    model_id=spec.model_id,
                    tier=str(spec.tier),
                    prompt_tokens=batch.prompt_tokens,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                    component=component,
                )
            )
        return DecisionResult(
            answers=answers,
            parsed=True,
            method=self.method,
            calibrated=self.config.fitted,
            model_id=spec.model_id,
            prompt_tokens=batch.prompt_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=cost,
        )
