"""Exception hierarchy for the ALM runtime.

Every failure the federation can produce is a subclass of :class:`ALMError`, so
callers embedding the runtime can catch one type and still distinguish the
cause.  Errors carry structured ``details`` because they end up in the audit
trail, not only in a log line.
"""

from __future__ import annotations

from typing import Any


class ALMError(Exception):
    """Base class for every ALM runtime error."""

    code = "alm_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigurationError(ALMError):
    """Something the operator must fix before the runtime can work."""

    code = "configuration_error"


class ModelNotConfiguredError(ConfigurationError):
    """A required model tier has no backing model registered."""

    code = "model_not_configured"


class BackendError(ALMError):
    """An inference backend failed or is unreachable."""

    code = "backend_error"


class BackendUnavailableError(BackendError):
    """The backend is open-circuited or refused the connection."""

    code = "backend_unavailable"


class GraphError(ALMError):
    """Invalid Context Graph operation (unknown node, bad edge, cycle …)."""

    code = "graph_error"


class OntologyError(ALMError):
    """The ontology definition is malformed or internally inconsistent."""

    code = "ontology_error"


class PackError(ALMError):
    """A domain pack could not be loaded."""

    code = "pack_error"


class RoutingError(ALMError):
    """The cognitive router could not produce a usable plan."""

    code = "routing_error"


class NoCapabilityMatchError(RoutingError):
    """No declared capability covers a subtask and no fallback is available."""

    code = "no_capability_match"


class OrchestrationError(ALMError):
    """The execution DAG is invalid or could not be executed."""

    code = "orchestration_error"


class CyclicPlanError(OrchestrationError):
    """The generated plan contains a dependency cycle."""

    code = "cyclic_plan"


class ArbitrationError(ALMError):
    """Arbitration could not resolve a conflict."""

    code = "arbitration_error"


class GovernanceDeniedError(ALMError):
    """IBAC denied the operation."""

    code = "governance_denied"


class EvaluationError(ALMError):
    """The evaluation harness could not complete a run."""

    code = "evaluation_error"


class DistillationError(ALMError):
    """The distillation pipeline failed."""

    code = "distillation_error"


class PromotionBlockedError(ALMError):
    """An MLOps promotion gate rejected a candidate version."""

    code = "promotion_blocked"
