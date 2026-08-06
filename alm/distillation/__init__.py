"""Distillation — teacher→student pipeline and the production feedback loop."""

from alm.distillation.pipeline import DistillationPipeline, TrainingExample
from alm.distillation.training import (
    TrainingConfig,
    TrainingResult,
    load_jsonl_dataset,
    train_adapter,
    training_available,
)

__all__ = [
    "DistillationPipeline",
    "TrainingConfig",
    "TrainingExample",
    "TrainingResult",
    "load_jsonl_dataset",
    "train_adapter",
    "training_available",
]
