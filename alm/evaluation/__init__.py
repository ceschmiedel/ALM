"""Evaluation — the controlled experiment that makes the ALM thesis falsifiable."""

from alm.evaluation.baseline import BaselineResult, MonolithicBaseline
from alm.evaluation.dataset import EvalCase, EvalDataset, load_datasets
from alm.evaluation.graders import GradeResult, grade
from alm.evaluation.harness import (
    EvaluationHarness,
    EvaluationRun,
    list_runs,
    load_pack_datasets,
)
from alm.evaluation.metrics import (
    ArmMetrics,
    CaseResult,
    Comparison,
    aggregate,
    compare,
    percentile,
    render_comparison,
)

__all__ = [
    "ArmMetrics",
    "BaselineResult",
    "CaseResult",
    "Comparison",
    "EvalCase",
    "EvalDataset",
    "EvaluationHarness",
    "EvaluationRun",
    "GradeResult",
    "MonolithicBaseline",
    "aggregate",
    "compare",
    "grade",
    "list_runs",
    "load_datasets",
    "load_pack_datasets",
    "percentile",
    "render_comparison",
]
