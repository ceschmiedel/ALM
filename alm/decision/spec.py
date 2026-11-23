"""The decision contract — typed questions in, typed judgments out.

Several layers of the federation do not need prose from a model; they need a
judgment that code will consume: which domain, which expert prevails, how good
an answer is, whether an intent violates a policy.  Asking for free text and
parsing it back is the expensive, unauditable way to get that.

This module names the three shapes such a judgment takes:

* :class:`Choice` — which option?  One label (or, with ``multi``, a ranked
  subset) from a closed set.
* :class:`Score`  — where on an ordered scale?
* :class:`Noul`   — is this true?  A probability between 0 and 1.

A :class:`DecisionRequest` carries the facts in ``state`` and the judgments in
``questions``; the two are kept apart so that the same state can be asked
several independent questions in one call.

The contract is backend-neutral on purpose.  Today it is served by a generative
model asked for JSON (:mod:`alm.decision.generative`); the same request can be
served by a dedicated decision model — hosted or local — without any caller
changing.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


class Choice(BaseModel):
    """Pick from a closed set of labels."""

    type: Literal["choice"] = "choice"
    instructions: str
    #: label → description.  The labels are the only valid answers.
    options: dict[str, str]
    #: When set, the answer is a ranked subset (possibly empty) instead of one label.
    multi: bool = False

    @model_validator(mode="after")
    def _has_options(self) -> Choice:
        if not self.options:
            raise ValueError("a choice question needs at least one option")
        return self


class Score(BaseModel):
    """Place the state on an ordered scale, lowest level first."""

    type: Literal["score"] = "score"
    instructions: str
    levels: list[str]

    @model_validator(mode="after")
    def _has_levels(self) -> Score:
        if len(self.levels) < 2:
            raise ValueError("a score question needs at least two levels")
        return self

    @property
    def top(self) -> int:
        return len(self.levels) - 1


class Noul(BaseModel):
    """A yes/no judgment, answered as the probability of "yes"."""

    type: Literal["noul"] = "noul"
    instructions: str


Question = Annotated[Choice | Score | Noul, Field(discriminator="type")]


class DecisionRequest(BaseModel):
    """Facts in ``state``, judgments in ``questions``."""

    #: What is being judged: a string, or any JSON-serialisable structure.
    state: Any
    #: Caller-chosen id → question.  The ids key the answers.
    questions: dict[str, Question]
    #: Role framing for backends that take a prompt.  Decision-native backends ignore it.
    preamble: str = ""
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _has_questions(self) -> DecisionRequest:
        if not self.questions:
            raise ValueError("a decision request needs at least one question")
        return self


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    #: The winning label; empty when the backend named none or an unknown one.
    choice: str = ""
    #: Ranked valid labels.  For a single choice this is ``[choice]`` or empty.
    selected: list[str] = Field(default_factory=list)
    #: Distribution over the options.  Empty when the backend cannot supply one.
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    #: Position on the scale, 0 … ``len(levels) - 1``; may fall between levels.
    score: float = 0.0
    #: The same position mapped onto 0 … 1, comparable across scales.
    normalised: float = 0.0
    probabilities: list[float] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    #: Probability that the statement holds.
    probability: float = 0.0
    reason: str = ""

    @property
    def value(self) -> bool:
        return self.probability >= 0.5


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


class DecisionResult(BaseModel):
    """Answers keyed like the questions, with accounting attached."""

    answers: dict[str, Answer] = Field(default_factory=dict)
    #: False when the backend returned nothing usable; every answer is then the empty default.
    parsed: bool = True
    #: How the judgments were produced, e.g. ``"generative"``.
    method: str = ""
    #: Whether the reported probabilities are calibrated rather than self-reported.
    calibrated: bool = False

    model_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    def choice(self, question_id: str) -> ChoiceAnswer:
        answer = self.answers.get(question_id)
        return answer if isinstance(answer, ChoiceAnswer) else ChoiceAnswer()

    def score(self, question_id: str) -> ScoreAnswer:
        answer = self.answers.get(question_id)
        return answer if isinstance(answer, ScoreAnswer) else ScoreAnswer()

    def noul(self, question_id: str) -> NoulAnswer:
        answer = self.answers.get(question_id)
        return answer if isinstance(answer, NoulAnswer) else NoulAnswer()


def empty_answer(question: Choice | Score | Noul) -> Answer:
    """The answer that means "the backend said nothing about this question"."""
    if isinstance(question, Choice):
        return ChoiceAnswer()
    if isinstance(question, Score):
        return ScoreAnswer()
    return NoulAnswer()
