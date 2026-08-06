"""Deterministic verification of expert answers.

The first arbitration strategy, and the one worth reaching for whenever the
domain rules are clear: cheap, deterministic, and it explains itself.  A clause
cannot contradict another; a total must equal the sum of its parts; a
confidential finding must carry a citation.  Checking that with rules costs
nothing and produces a reason a person can act on — asking a model to judge it
costs a call and produces prose.

The LLM judge exists for genuinely ambiguous conflicts.  It should not be
spending tokens on arithmetic.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from alm.experts.spec import VerifierRule
from alm.protocol.answer import ExpertAnswer


class Violation(BaseModel):
    """One failed check."""

    rule_id: str
    rule_type: str
    severity: str = "error"
    message: str = ""
    field: str = ""


class VerificationResult(BaseModel):
    """Outcome of verifying one answer."""

    expert_id: str = ""
    answer_id: str = ""
    passed: bool = True
    violations: list[Violation] = Field(default_factory=list)
    checked_rules: list[str] = Field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warning"]

    def summary(self) -> str:
        if self.passed:
            return f"{len(self.checked_rules)} rule(s) passed"
        return "; ".join(f"[{v.rule_id}] {v.message}" for v in self.errors) or "failed"


class AnswerVerifier:
    """Applies an expert's :class:`VerifierRule` list to its answers."""

    def __init__(self, rules: list[VerifierRule] | None = None) -> None:
        self.rules = rules or []

    def verify(self, answer: ExpertAnswer) -> VerificationResult:
        result = VerificationResult(
            expert_id=answer.expert_id,
            answer_id=answer.answer_id,
            checked_rules=[r.id for r in self.rules],
        )
        if answer.status != "success":
            result.passed = False
            result.violations.append(
                Violation(
                    rule_id="__execution__",
                    rule_type="execution",
                    message=answer.error or "expert did not produce an answer",
                )
            )
            return result

        for rule in self.rules:
            violation = self._check(rule, answer)
            if violation is not None:
                result.violations.append(violation)

        result.passed = not result.errors
        return result

    # -- individual checks -------------------------------------------------

    def _check(self, rule: VerifierRule, answer: ExpertAnswer) -> Violation | None:
        handler = {
            "required_fields": self._required_fields,
            "requires_citation": self._requires_citation,
            "numeric_range": self._numeric_range,
            "numeric_consistency": self._numeric_consistency,
            "forbidden_terms": self._forbidden_terms,
            "regex_match": self._regex_match,
            "min_confidence": self._min_confidence,
        }.get(rule.type)
        if handler is None:
            return Violation(
                rule_id=rule.id,
                rule_type=rule.type,
                severity="warning",
                message=f"unknown verifier rule type {rule.type!r}",
            )
        return handler(rule, answer)

    @staticmethod
    def _required_fields(rule: VerifierRule, answer: ExpertAnswer) -> Violation | None:
        missing = [
            field
            for field in rule.fields
            if field not in answer.structured
            or answer.structured[field] in (None, "", [], {})
        ]
        if not missing:
            return None
        return Violation(
            rule_id=rule.id,
            rule_type=rule.type,
            severity=rule.severity,
            message=f"missing required output field(s): {', '.join(missing)}",
            field=missing[0],
        )

    @staticmethod
    def _requires_citation(rule: VerifierRule, answer: ExpertAnswer) -> Violation | None:
        if answer.citations:
            return None
        return Violation(
            rule_id=rule.id,
            rule_type=rule.type,
            severity=rule.severity,
            message="answer cites no source from the domain corpus",
        )

    @staticmethod
    def _numeric_range(rule: VerifierRule, answer: ExpertAnswer) -> Violation | None:
        for field in rule.fields:
            value = answer.structured.get(field)
            number = _as_number(value)
            if number is None:
                continue
            if rule.minimum is not None and number < rule.minimum:
                return Violation(
                    rule_id=rule.id,
                    rule_type=rule.type,
                    severity=rule.severity,
                    message=f"{field}={number} is below the allowed minimum {rule.minimum}",
                    field=field,
                )
            if rule.maximum is not None and number > rule.maximum:
                return Violation(
                    rule_id=rule.id,
                    rule_type=rule.type,
                    severity=rule.severity,
                    message=f"{field}={number} exceeds the allowed maximum {rule.maximum}",
                    field=field,
                )
        return None

    @staticmethod
    def _numeric_consistency(rule: VerifierRule, answer: ExpertAnswer) -> Violation | None:
        """First field must equal the sum of the rest, within tolerance."""
        if len(rule.fields) < 2:
            return None
        total = _as_number(answer.structured.get(rule.fields[0]))
        parts = [_as_number(answer.structured.get(f)) for f in rule.fields[1:]]
        if total is None or any(p is None for p in parts):
            return None
        expected = sum(p for p in parts if p is not None)
        if abs(total - expected) <= max(rule.tolerance, abs(expected) * rule.tolerance):
            return None
        return Violation(
            rule_id=rule.id,
            rule_type=rule.type,
            severity=rule.severity,
            message=(
                f"{rule.fields[0]}={total} does not equal the sum of "
                f"{', '.join(rule.fields[1:])} ({expected})"
            ),
            field=rule.fields[0],
        )

    @staticmethod
    def _forbidden_terms(rule: VerifierRule, answer: ExpertAnswer) -> Violation | None:
        haystack = (answer.content + " " + str(answer.structured)).lower()
        hits = [term for term in rule.terms if term.lower() in haystack]
        if not hits:
            return None
        return Violation(
            rule_id=rule.id,
            rule_type=rule.type,
            severity=rule.severity,
            message=f"answer contains forbidden term(s): {', '.join(hits)}",
        )

    @staticmethod
    def _regex_match(rule: VerifierRule, answer: ExpertAnswer) -> Violation | None:
        if not rule.pattern:
            return None
        try:
            pattern = re.compile(rule.pattern, re.IGNORECASE)
        except re.error as exc:
            return Violation(
                rule_id=rule.id,
                rule_type=rule.type,
                severity="warning",
                message=f"invalid regex in rule: {exc}",
            )
        targets = (
            [str(answer.structured.get(f, "")) for f in rule.fields]
            if rule.fields
            else [answer.content]
        )
        if any(pattern.search(t) for t in targets):
            return None
        return Violation(
            rule_id=rule.id,
            rule_type=rule.type,
            severity=rule.severity,
            message=f"answer does not match the required pattern {rule.pattern!r}",
            field=rule.fields[0] if rule.fields else "",
        )

    @staticmethod
    def _min_confidence(rule: VerifierRule, answer: ExpertAnswer) -> Violation | None:
        threshold = rule.minimum if rule.minimum is not None else 0.4
        if answer.confidence >= threshold:
            return None
        return Violation(
            rule_id=rule.id,
            rule_type=rule.type,
            severity=rule.severity,
            message=(
                f"calibrated confidence {answer.confidence:.2f} is below the "
                f"required {threshold:.2f}"
            ),
        )


_FULL_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?%?")
_CURRENCY_RE = re.compile(r"[€$£R\s]|EUR|USD|GBP|BRL", re.IGNORECASE)


def _as_number(value: Any) -> float | None:
    """Parse a value that *is* a number, tolerating currency and separators.

    Stripping every non-digit from arbitrary text would turn "clause 7.2 of the
    2025 agreement" into 7.22025 and range-check it — so a string only counts as
    a number when what remains after removing currency marks is entirely
    numeric.
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
            return float(cleaned.rstrip("%"))
        except ValueError:
            return None
    return None
