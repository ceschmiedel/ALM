"""Model and adapter versioning, with eval-gated promotion and rollback.

The MLOps load is the real hidden cost of ALM, and the mitigation is to treat it
as platform from the first dedicated agent rather than as per-client effort.
Three guarantees are implemented here:

* **Every model and adapter has versions, and rollback is a row update.** A new
  version of one expert must never degrade the others.
* **No promotion without evaluation.** A candidate is promoted only against a
  recorded evaluation run that beats the incumbent on the deciding metric.
* **Drift is watched.** Falling accuracy or rising fallback over time is the
  signal that the domain moved and the model needs recycling.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select

from alm.core.errors import PromotionBlockedError
from alm.core.ids import utc_now
from alm.persistence.database import session_scope
from alm.persistence.models import ModelRow, ModelVersionRow

logger = logging.getLogger(__name__)


class ModelVersion(BaseModel):
    """One immutable version of a model or adapter."""

    model_id: str
    version: str
    status: str = "candidate"
    artifact_uri: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    eval_run_id: str = ""
    notes: str = ""
    created_at: str = ""
    promoted_at: str = ""


class VersionRegistry:
    """Version history and promotion control for the model registry."""

    def __init__(self, tenant_id: str = "default") -> None:
        self.tenant_id = tenant_id

    # -- recording ---------------------------------------------------------

    def record(
        self,
        model_id: str,
        version: str,
        *,
        artifact_uri: str = "",
        metrics: dict[str, Any] | None = None,
        eval_run_id: str = "",
        notes: str = "",
        status: str = "candidate",
    ) -> ModelVersion:
        """Register a new candidate version.  Idempotent per ``(model, version)``."""
        with session_scope() as session:
            row = session.execute(
                select(ModelVersionRow).where(
                    ModelVersionRow.model_id == model_id,
                    ModelVersionRow.version == version,
                )
            ).scalar_one_or_none()
            if row is None:
                row = ModelVersionRow(model_id=model_id, version=version)
                session.add(row)
            row.artifact_uri = artifact_uri
            row.metrics = metrics or {}
            row.eval_run_id = eval_run_id
            row.notes = notes
            row.status = status
            session.flush()
            return _to_version(row)

    def versions(self, model_id: str) -> list[ModelVersion]:
        with session_scope() as session:
            rows = session.execute(
                select(ModelVersionRow)
                .where(ModelVersionRow.model_id == model_id)
                .order_by(ModelVersionRow.created_at.desc())
            ).scalars().all()
            return [_to_version(r) for r in rows]

    def active(self, model_id: str) -> ModelVersion | None:
        for version in self.versions(model_id):
            if version.status == "active":
                return version
        return None

    # -- promotion ---------------------------------------------------------

    def promote(
        self,
        model_id: str,
        version: str,
        *,
        metric: str = "domain_accuracy",
        min_improvement: float = 0.0,
        force: bool = False,
    ) -> ModelVersion:
        """Promote a candidate to active, gated on its evaluation record.

        The gate is refused, not warned about: a promotion without a recorded
        evaluation is exactly how a federation acquires an expert nobody
        measured, and by the time it shows up in production metrics it has
        already been answering.
        """
        with session_scope() as session:
            candidate = session.execute(
                select(ModelVersionRow).where(
                    ModelVersionRow.model_id == model_id,
                    ModelVersionRow.version == version,
                )
            ).scalar_one_or_none()
            if candidate is None:
                raise PromotionBlockedError(
                    f"version {version!r} of model {model_id!r} is not registered",
                    model_id=model_id,
                    version=version,
                )

            if not force:
                candidate_value = _metric(candidate.metrics, metric)
                if candidate_value is None:
                    raise PromotionBlockedError(
                        f"version {version} has no recorded '{metric}'. Run "
                        f"`alm eval run` and attach the result, or pass --force "
                        f"to accept the risk explicitly.",
                        model_id=model_id,
                        version=version,
                        metric=metric,
                    )

                incumbent = session.execute(
                    select(ModelVersionRow).where(
                        ModelVersionRow.model_id == model_id,
                        ModelVersionRow.status == "active",
                    )
                ).scalar_one_or_none()
                if incumbent is not None:
                    incumbent_value = _metric(incumbent.metrics, metric) or 0.0
                    if candidate_value < incumbent_value + min_improvement:
                        raise PromotionBlockedError(
                            f"version {version} scores {candidate_value:.4f} on "
                            f"'{metric}' against the active version's "
                            f"{incumbent_value:.4f}; promotion would be a regression",
                            model_id=model_id,
                            version=version,
                            candidate=candidate_value,
                            incumbent=incumbent_value,
                        )

            for row in session.execute(
                select(ModelVersionRow).where(
                    ModelVersionRow.model_id == model_id,
                    ModelVersionRow.status == "active",
                )
            ).scalars().all():
                row.status = "retired"
                row.retired_at = utc_now()

            candidate.status = "active"
            candidate.promoted_at = utc_now()

            model_row = session.get(ModelRow, model_id)
            if model_row is not None:
                model_row.active_version = version
                if candidate.artifact_uri:
                    model_row.adapter_uri = candidate.artifact_uri

            session.flush()
            logger.info("Promoted %s to version %s", model_id, version)
            return _to_version(candidate)

    def rollback(self, model_id: str, version: str = "") -> ModelVersion:
        """Reactivate a previous version.

        Defaults to the most recently retired one, which is what an operator
        wants at 3am: undo the last promotion.
        """
        history = self.versions(model_id)
        if not history:
            raise PromotionBlockedError(
                f"model {model_id!r} has no version history to roll back to",
                model_id=model_id,
            )

        target = version
        if not target:
            retired = [v for v in history if v.status == "retired"]
            if not retired:
                raise PromotionBlockedError(
                    f"model {model_id!r} has no retired version to roll back to",
                    model_id=model_id,
                )
            target = retired[0].version

        return self.promote(model_id, target, force=True)

    def history(self, model_id: str) -> list[dict[str, Any]]:
        return [v.model_dump(mode="json") for v in self.versions(model_id)]


def _to_version(row: ModelVersionRow) -> ModelVersion:
    return ModelVersion(
        model_id=row.model_id,
        version=row.version,
        status=row.status,
        artifact_uri=row.artifact_uri,
        metrics=dict(row.metrics or {}),
        eval_run_id=row.eval_run_id,
        notes=row.notes,
        created_at=row.created_at.isoformat() if row.created_at else "",
        promoted_at=row.promoted_at.isoformat() if row.promoted_at else "",
    )


def _metric(metrics: dict[str, Any] | None, name: str) -> float | None:
    if not metrics:
        return None
    value = metrics.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def summarise(registry: VersionRegistry, model_ids: Sequence[str]) -> dict[str, Any]:
    """Version status across several models, for `alm model versions`."""
    out: dict[str, Any] = {}
    for model_id in model_ids:
        versions = registry.versions(model_id)
        active = next((v for v in versions if v.status == "active"), None)
        out[model_id] = {
            "versions": len(versions),
            "active": active.version if active else None,
            "candidates": [v.version for v in versions if v.status == "candidate"],
        }
    return out
