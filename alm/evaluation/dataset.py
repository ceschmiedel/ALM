"""Evaluation datasets.

One case per line of JSONL, so a domain expert can author and review the set in
a text editor and diff it in version control.

The split matters more than the format.  The evaluation set must never be seen
during training — the distillation pipeline writes to ``train`` and this module
reads ``eval``, and the two are kept apart deliberately, because a domain model
scored on data it was trained on tells you nothing about whether the federation
beats the baseline.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from alm.core.errors import EvaluationError


class EvalCase(BaseModel):
    """One question with a known-good answer and how to grade it."""

    id: str = ""
    domain: str = ""
    question: str
    expected: str = ""

    grader: str = Field(
        default="keywords",
        description="keywords | numeric | exact | contains | rubric",
    )
    keywords: list[str] = Field(default_factory=list)
    min_keyword_ratio: float = Field(
        default=0.6, description="Fraction of keywords required to count as correct."
    )
    expected_number: float | None = None
    tolerance: float = 0.02
    rubric: str = ""

    # Governance context the case should run under.
    claims: list[str] = Field(default_factory=list)
    principal: str = ""

    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("eval case question must not be empty")
        return value.strip()


class EvalDataset(BaseModel):
    """A named collection of cases."""

    name: str = ""
    domain: str = ""
    cases: list[EvalCase] = Field(default_factory=list)
    source: str = ""

    @classmethod
    def from_jsonl(cls, path: str | Path, *, domain: str = "") -> EvalDataset:
        file_path = Path(path)
        if not file_path.exists():
            raise EvaluationError(
                f"evaluation dataset not found: {file_path}", path=str(file_path)
            )

        cases: list[EvalCase] = []
        for line_number, line in enumerate(
            file_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"{file_path}:{line_number} is not valid JSON: {exc}",
                    path=str(file_path),
                    line=line_number,
                ) from exc
            try:
                case = EvalCase.model_validate(payload)
            except Exception as exc:
                raise EvaluationError(
                    f"{file_path}:{line_number} is not a valid eval case: {exc}",
                    path=str(file_path),
                    line=line_number,
                ) from exc
            if not case.id:
                case.id = f"{file_path.stem}-{line_number:03d}"
            if not case.domain:
                case.domain = domain or file_path.stem
            cases.append(case)

        if not cases:
            raise EvaluationError(
                f"evaluation dataset {file_path} contains no cases", path=str(file_path)
            )

        return cls(
            name=file_path.stem,
            domain=domain or (cases[0].domain if cases else ""),
            cases=cases,
            source=str(file_path),
        )

    @classmethod
    def merge(cls, datasets: Sequence[EvalDataset], *, name: str = "merged") -> EvalDataset:
        cases: list[EvalCase] = []
        for dataset in datasets:
            cases.extend(dataset.cases)
        domains = {d.domain for d in datasets if d.domain}
        return cls(
            name=name,
            domain=next(iter(domains)) if len(domains) == 1 else "",
            cases=cases,
            source=", ".join(d.source for d in datasets),
        )

    def filter_domain(self, domain: str) -> EvalDataset:
        return EvalDataset(
            name=f"{self.name}:{domain}",
            domain=domain,
            cases=[c for c in self.cases if c.domain == domain],
            source=self.source,
        )

    def domains(self) -> list[str]:
        return sorted({c.domain for c in self.cases if c.domain})

    def to_jsonl(self, path: str | Path) -> int:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            for case in self.cases:
                handle.write(
                    json.dumps(case.model_dump(mode="json", exclude_defaults=True))
                    + "\n"
                )
        return len(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvalCase]:  # type: ignore[override]
        return iter(self.cases)


def load_datasets(paths: Sequence[str | Path]) -> EvalDataset:
    """Load and merge several JSONL files."""
    return EvalDataset.merge([EvalDataset.from_jsonl(p) for p in paths])
