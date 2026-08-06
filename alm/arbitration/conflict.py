"""Conflict detection between expert answers.

In a monolithic model, when two lines of reasoning contradict each other the
resolution happens implicitly inside the latent space — invisible and
unauditable.  In a federation two Expert Agents can reach incompatible
conclusions, and someone has to decide explicitly which one prevails.  What was
a black box becomes a designed decision.

The first half of that is *noticing*.  Detection here is deliberately
conservative: it flags disagreements it can point at concretely — the same
structured field with incompatible values, opposite verdicts on the same
question, a failed domain check — rather than inferring contradiction from
prose. A false conflict costs a judge call and muddies the audit trail; a
missed one is caught downstream by the verifier or by synthesis.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from alm.protocol.answer import ExpertAnswer

#: Fields whose disagreement is a substantive conflict rather than phrasing.
VERDICT_FIELDS = {
    "verdict", "compliant", "enforceable", "approved", "valid", "permitted",
    "recommendation", "decision", "risk_level", "severity", "status", "outcome",
    "conclusion", "eligible", "acceptable",
}

_FULL_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?%?")
_CURRENCY_RE = re.compile(r"[€$£R\s]|EUR|USD|GBP|BRL", re.IGNORECASE)

_NEGATION_PAIRS = [
    ("compliant", "non-compliant"),
    ("enforceable", "unenforceable"),
    ("valid", "invalid"),
    ("permitted", "prohibited"),
    ("approved", "rejected"),
    ("sim", "não"),
    ("yes", "no"),
    ("true", "false"),
]


class ConflictKind(StrEnum):
    FIELD_DISAGREEMENT = "field_disagreement"
    VERDICT_DISAGREEMENT = "verdict_disagreement"
    NUMERIC_DISAGREEMENT = "numeric_disagreement"
    VERIFICATION_FAILURE = "verification_failure"


class Conflict(BaseModel):
    """Two or more answers that cannot both be right."""

    kind: ConflictKind
    field: str = ""
    domain: str = ""
    participants: list[str] = Field(
        default_factory=list, description="answer_id of each side."
    )
    experts: list[str] = Field(default_factory=list)
    values: dict[str, Any] = Field(
        default_factory=dict, description="expert_id → the value it asserted."
    )
    description: str = ""

    def summary(self) -> str:
        rendered = "; ".join(f"{k} says {v!r}" for k, v in self.values.items())
        return f"[{self.kind}] {self.field or 'conclusion'}: {rendered}"


class ConflictDetector:
    """Finds substantive disagreements across a set of answers."""

    def __init__(self, *, numeric_tolerance: float = 0.02) -> None:
        #: Relative difference below which two numbers are treated as agreeing.
        self.numeric_tolerance = numeric_tolerance

    def detect(self, answers: list[ExpertAnswer]) -> list[Conflict]:
        usable = [a for a in answers if a.ok]
        if len(usable) < 2:
            return []

        conflicts: list[Conflict] = []
        conflicts.extend(self._structured_conflicts(usable))
        conflicts.extend(self._narrative_verdict_conflicts(usable))
        return _deduplicate(conflicts)

    # -- structured --------------------------------------------------------

    def _structured_conflicts(self, answers: list[ExpertAnswer]) -> list[Conflict]:
        by_field: dict[str, list[tuple[ExpertAnswer, Any]]] = {}
        for answer in answers:
            for field, value in (answer.structured or {}).items():
                if field in {"confidence", "citations", "sources", "reasoning"}:
                    continue
                if value in (None, "", [], {}):
                    continue
                by_field.setdefault(field, []).append((answer, value))

        conflicts: list[Conflict] = []
        for field, entries in by_field.items():
            if len(entries) < 2:
                continue
            for index, (answer_a, value_a) in enumerate(entries):
                for answer_b, value_b in entries[index + 1 :]:
                    kind = self._incompatible(field, value_a, value_b)
                    if kind is None:
                        continue
                    conflicts.append(
                        Conflict(
                            kind=kind,
                            field=field,
                            domain=answer_a.domain,
                            participants=[answer_a.answer_id, answer_b.answer_id],
                            experts=[answer_a.expert_id, answer_b.expert_id],
                            values={
                                answer_a.expert_id: value_a,
                                answer_b.expert_id: value_b,
                            },
                            description=(
                                f"{answer_a.expert_id} and {answer_b.expert_id} report "
                                f"different values for '{field}'"
                            ),
                        )
                    )
        return conflicts

    def _incompatible(self, field: str, a: Any, b: Any) -> ConflictKind | None:
        """Whether two values for the same field genuinely disagree."""
        if isinstance(a, bool) or isinstance(b, bool):
            if bool(a) != bool(b):
                return ConflictKind.VERDICT_DISAGREEMENT
            return None

        number_a, number_b = _as_number(a), _as_number(b)
        if number_a is not None and number_b is not None:
            scale = max(abs(number_a), abs(number_b), 1e-9)
            if abs(number_a - number_b) / scale > self.numeric_tolerance:
                return ConflictKind.NUMERIC_DISAGREEMENT
            return None

        if isinstance(a, str) and isinstance(b, str):
            norm_a, norm_b = a.strip().lower(), b.strip().lower()
            if norm_a == norm_b:
                return None
            if field.lower() in VERDICT_FIELDS or _are_opposites(norm_a, norm_b):
                return ConflictKind.VERDICT_DISAGREEMENT
            # Free-text fields differ by phrasing far more often than by
            # substance; flagging every wording difference would be noise.
            return None

        if isinstance(a, list) and isinstance(b, list):
            return None  # complementary lists are a merge, not a conflict

        return None

    # -- narrative ---------------------------------------------------------

    def _narrative_verdict_conflicts(self, answers: list[ExpertAnswer]) -> list[Conflict]:
        """Catch opposite verdicts stated in prose when no schema was used."""
        stances: list[tuple[ExpertAnswer, str, str]] = []
        for answer in answers:
            if answer.structured:
                continue  # structured comparison already covered it
            for positive, negative in _NEGATION_PAIRS:
                text = answer.content.lower()
                has_negative = re.search(rf"\b{re.escape(negative)}\b", text)
                has_positive = re.search(rf"\b{re.escape(positive)}\b", text)
                if has_negative and not has_positive:
                    stances.append((answer, positive, "negative"))
                elif has_positive and not has_negative:
                    stances.append((answer, positive, "positive"))

        conflicts: list[Conflict] = []
        by_topic: dict[str, list[tuple[ExpertAnswer, str]]] = {}
        for answer, topic, stance in stances:
            by_topic.setdefault(topic, []).append((answer, stance))

        for topic, entries in by_topic.items():
            positives = [a for a, s in entries if s == "positive"]
            negatives = [a for a, s in entries if s == "negative"]
            if positives and negatives:
                conflicts.append(
                    Conflict(
                        kind=ConflictKind.VERDICT_DISAGREEMENT,
                        field=topic,
                        domain=positives[0].domain,
                        participants=[positives[0].answer_id, negatives[0].answer_id],
                        experts=[positives[0].expert_id, negatives[0].expert_id],
                        values={
                            positives[0].expert_id: topic,
                            negatives[0].expert_id: f"not {topic}",
                        },
                        description=(
                            f"{positives[0].expert_id} and {negatives[0].expert_id} take "
                            f"opposite positions on '{topic}'"
                        ),
                    )
                )
        return conflicts


def _as_number(value: Any) -> float | None:
    """Parse a value that *is* a number — never one that merely contains one.

    Searching for a number inside free text is how a narrative field turns into
    a phantom conflict: two experts writing prose that happens to mention
    "clause 7.2" and "clause 9.1" are not disagreeing about a quantity, and
    reporting them as a numeric conflict would spend a judge call and put a
    fabricated disagreement into the audit trail.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = _CURRENCY_RE.sub("", value).replace(",", "").strip()
        if not cleaned or not _FULL_NUMBER_RE.fullmatch(cleaned):
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _are_opposites(a: str, b: str) -> bool:
    for positive, negative in _NEGATION_PAIRS:
        if {a, b} == {positive, negative}:
            return True
        if a.startswith("not ") and a[4:].strip() == b:
            return True
        if b.startswith("not ") and b[4:].strip() == a:
            return True
    return False


def _deduplicate(conflicts: list[Conflict]) -> list[Conflict]:
    seen: set[tuple[str, str, frozenset[str]]] = set()
    unique: list[Conflict] = []
    for conflict in conflicts:
        key = (str(conflict.kind), conflict.field, frozenset(conflict.participants))
        if key in seen:
            continue
        seen.add(key)
        unique.append(conflict)
    return unique
