"""LoRA training for expert adapters.

Stage 4 of the pipeline: fine-tune the student on the distilled dataset.

LoRA is the default for a reason that is economic rather than technical. It
trains a small number of parameters over a frozen base, adds no inference
latency, and — decisively — is cheap to *serve*: many adapters share one loaded
base, so the federation's GPU cost stops scaling with the number of experts.

Training needs the optional extra::

    pip install "alm-federation[train]"

The module is importable without it; only :func:`train_adapter` requires torch,
so `alm distill` and the rest of the CLI work on a machine with no GPU.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from alm.core.errors import ConfigurationError, DistillationError

logger = logging.getLogger(__name__)

_IMPORT_HINT = 'LoRA training needs the optional extra: pip install "alm-federation[train]"'


class TrainingConfig(BaseModel):
    """Hyper-parameters for one adapter training run."""

    base_model: str
    dataset_path: str
    output_dir: str

    adapter_name: str = "adapter"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    epochs: float = 3.0
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    seed: int = 17

    bf16: bool = True
    gradient_checkpointing: bool = True


class TrainingResult(BaseModel):
    """What a training run produced."""

    adapter_path: str = ""
    base_model: str = ""
    adapter_name: str = ""
    examples: int = 0
    epochs: float = 0.0
    final_loss: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


def load_jsonl_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read a distilled dataset written by :meth:`DistillationPipeline.export`."""
    file_path = Path(path)
    if not file_path.exists():
        raise DistillationError(f"training dataset not found: {file_path}", path=str(file_path))
    records: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed line in %s", file_path)
    if not records:
        raise DistillationError(f"training dataset {file_path} is empty", path=str(file_path))
    return records


def train_adapter(config: TrainingConfig) -> TrainingResult:
    """Train a LoRA adapter and write it to ``output_dir``.

    The produced directory is what :class:`~alm.models.spec.ModelSpec`'s
    ``adapter_uri`` points at, and what a vLLM server registers to serve the
    adapter alongside its siblings on one base.
    """
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ConfigurationError(_IMPORT_HINT) from exc

    records = load_jsonl_dataset(config.dataset_path)
    logger.info(
        "Training adapter %s on %d example(s) over base %s",
        config.adapter_name,
        len(records),
        config.base_model,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(record: dict[str, Any]) -> str:
        messages = record.get("messages") or []
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )
        # Causal LM: labels are the inputs; the collator shifts them.
        encoded["labels"] = [list(ids) for ids in encoded["input_ids"]]
        return encoded

    dataset = Dataset.from_dict({"text": [to_text(r) for r in records]})
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(config.base_model)
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable, total = _parameter_counts(model)
    logger.info(
        "Trainable parameters: %s of %s (%.3f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=config.seed,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=tokenized)
    train_output = trainer.train()

    adapter_path = output_dir / config.adapter_name
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    logger.info("Adapter written to %s", adapter_path)
    return TrainingResult(
        adapter_path=str(adapter_path),
        base_model=config.base_model,
        adapter_name=config.adapter_name,
        examples=len(records),
        epochs=config.epochs,
        final_loss=float(getattr(train_output, "training_loss", 0.0) or 0.0),
        metrics={
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": round(trainable / max(total, 1), 6),
        },
    )


def _parameter_counts(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def training_available() -> bool:
    """Whether the optional training stack is installed."""
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False
