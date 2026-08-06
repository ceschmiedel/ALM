"""Confidence calibration.

Arbitration by confidence is only meaningful if the numbers mean something.  A
model that is confident and wrong is worse than one that is uncertain and
honest, so raw self-reported confidence is never used directly to decide
between experts — it is mapped through a per-expert calibration fitted on
labelled evaluation data.

The method is Platt scaling on the logit: ``p' = σ(a · logit(p) + b)``, fitted
by gradient descent on log loss.  Two parameters, so it fits from the handful of
labelled cases a domain evaluation set realistically provides, without
overfitting the way a richer model would.

Until an expert has been calibrated, its answers are marked ``calibrated=False``
and arbitration discounts them toward the prior rather than trusting a number
that has never been checked.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
from sqlalchemy import select

from alm.persistence.database import session_scope
from alm.persistence.models import CalibrationRow

logger = logging.getLogger(__name__)

_EPS = 1e-6

#: Minimum labelled samples before a fit is trusted.
MIN_SAMPLES = 12

#: Confidence assigned when an uncalibrated expert reports nothing usable.
UNCALIBRATED_PRIOR = 0.5

#: How far an uncalibrated raw score may move away from the prior.
UNCALIBRATED_SHRINK = 0.6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def brier_score(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Mean squared error of probabilistic predictions.  Lower is better."""
    if not probabilities:
        return 0.0
    return round(
        float(
            np.mean(
                [
                    (p - (1.0 if y else 0.0)) ** 2
                    for p, y in zip(probabilities, outcomes, strict=False)
                ]
            )
        ),
        6,
    )


class Calibration:
    """Fitted parameters for one expert."""

    __slots__ = (
        "expert_id",
        "slope",
        "bias",
        "samples",
        "brier_before",
        "brier_after",
        "fitted",
    )

    def __init__(
        self,
        expert_id: str,
        slope: float = 1.0,
        bias: float = 0.0,
        samples: int = 0,
        brier_before: float = 0.0,
        brier_after: float = 0.0,
        fitted: bool = False,
    ) -> None:
        self.expert_id = expert_id
        self.slope = slope
        self.bias = bias
        self.samples = samples
        self.brier_before = brier_before
        self.brier_after = brier_after
        #: True only when a fit was actually performed *and accepted*. Having
        #: enough samples is not the same as having a usable calibration: a
        #: degenerate set (all correct, all wrong) and a fit that worsened the
        #: Brier score both leave the expert uncalibrated, and arbitration must
        #: keep discounting it rather than trusting a number nobody validated.
        self.fitted = fitted

    def apply(self, raw_confidence: float) -> float:
        """Map a raw score to a calibrated probability."""
        if not self.fitted:
            # Shrink toward the prior: an unvalidated number should not carry
            # the same weight in arbitration as a validated one.
            return round(
                UNCALIBRATED_PRIOR
                + (raw_confidence - UNCALIBRATED_PRIOR) * UNCALIBRATED_SHRINK,
                6,
            )
        return round(_sigmoid(self.slope * _logit(raw_confidence) + self.bias), 6)

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "expert_id": self.expert_id,
            "slope": round(self.slope, 6),
            "bias": round(self.bias, 6),
            "samples": self.samples,
            "fitted": self.fitted,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
        }


def fit_calibration(
    expert_id: str,
    raw_confidences: Sequence[float],
    outcomes: Sequence[bool],
    *,
    iterations: int = 800,
    learning_rate: float = 0.15,
) -> Calibration:
    """Fit Platt scaling by gradient descent on log loss."""
    pairs = [
        (float(c), bool(o))
        for c, o in zip(raw_confidences, outcomes, strict=False)
        if c is not None
    ]
    if len(pairs) < MIN_SAMPLES:
        return Calibration(expert_id, samples=len(pairs))

    x = np.array([_logit(c) for c, _ in pairs], dtype=np.float64)
    y = np.array([1.0 if o else 0.0 for _, o in pairs], dtype=np.float64)

    # A set that is all-correct or all-wrong carries no information about the
    # mapping; fitting it would drive the parameters to infinity.
    if y.min() == y.max():
        return Calibration(expert_id, samples=len(pairs))

    slope, bias = 1.0, 0.0
    n = len(x)
    for _ in range(iterations):
        z = slope * x + bias
        predictions = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        error = predictions - y
        grad_slope = float(np.dot(error, x) / n)
        grad_bias = float(np.sum(error) / n)
        slope -= learning_rate * grad_slope
        bias -= learning_rate * grad_bias

    before = brier_score([c for c, _ in pairs], [o for _, o in pairs])
    calibrated = [_sigmoid(slope * _logit(c) + bias) for c, _ in pairs]
    after = brier_score(calibrated, [o for _, o in pairs])

    if after > before:
        # The fit made predictions worse — keep the identity mapping and say so.
        logger.info(
            "Calibration for %s rejected: Brier %.4f → %.4f (worse); keeping raw scores",
            expert_id,
            before,
            after,
        )
        return Calibration(
            expert_id, samples=len(pairs), brier_before=before, brier_after=before
        )

    logger.info(
        "Calibrated %s on %d samples: Brier %.4f → %.4f (slope=%.3f bias=%.3f)",
        expert_id,
        len(pairs),
        before,
        after,
        slope,
        bias,
    )
    return Calibration(expert_id, slope, bias, len(pairs), before, after, fitted=True)


class CalibrationStore:
    """Persistence for fitted calibrations, cached per process."""

    def __init__(self, tenant_id: str = "default") -> None:
        self.tenant_id = tenant_id
        self._cache: dict[str, Calibration] = {}

    def get(self, expert_id: str) -> Calibration:
        cached = self._cache.get(expert_id)
        if cached is not None:
            return cached
        with session_scope() as session:
            row = session.execute(
                select(CalibrationRow).where(
                    CalibrationRow.tenant_id == self.tenant_id,
                    CalibrationRow.expert_id == expert_id,
                )
            ).scalar_one_or_none()
            calibration = (
                Calibration(
                    expert_id,
                    slope=row.temperature,
                    bias=row.bias,
                    samples=row.sample_count,
                    brier_before=row.brier_before,
                    brier_after=row.brier_after,
                    fitted=bool(row.fitted),
                )
                if row is not None
                else Calibration(expert_id)
            )
        self._cache[expert_id] = calibration
        return calibration

    def save(self, calibration: Calibration) -> None:
        with session_scope() as session:
            row = session.execute(
                select(CalibrationRow).where(
                    CalibrationRow.tenant_id == self.tenant_id,
                    CalibrationRow.expert_id == calibration.expert_id,
                )
            ).scalar_one_or_none()
            if row is None:
                row = CalibrationRow(
                    tenant_id=self.tenant_id, expert_id=calibration.expert_id
                )
                session.add(row)
            row.temperature = calibration.slope
            row.bias = calibration.bias
            row.sample_count = calibration.samples
            row.brier_before = calibration.brier_before
            row.brier_after = calibration.brier_after
            row.fitted = calibration.fitted
        self._cache[calibration.expert_id] = calibration

    def apply(self, expert_id: str, raw_confidence: float) -> tuple[float, bool]:
        """Return ``(calibrated_confidence, was_fitted)``."""
        calibration = self.get(expert_id)
        return calibration.apply(raw_confidence), calibration.fitted

    def all(self) -> list[Calibration]:
        with session_scope() as session:
            rows = session.execute(
                select(CalibrationRow).where(CalibrationRow.tenant_id == self.tenant_id)
            ).scalars().all()
            return [
                Calibration(
                    r.expert_id,
                    r.temperature,
                    r.bias,
                    r.sample_count,
                    r.brier_before,
                    r.brier_after,
                    fitted=bool(r.fitted),
                )
                for r in rows
            ]

    def invalidate(self, expert_id: str | None = None) -> None:
        if expert_id is None:
            self._cache.clear()
        else:
            self._cache.pop(expert_id, None)
