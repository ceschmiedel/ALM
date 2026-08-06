"""Structured task objects — the contract between L1, L2 and L4.

L1 turns a free-form intention into a :class:`TaskEnvelope`.  L2 decomposes it
into :class:`SubTask` nodes carrying explicit dependencies, producing an
:class:`ExecutionPlan`.  L4 turns that plan into a DAG.

Two properties matter for the whole architecture and are enforced here:

* **Entity references, not loose text.** When one expert's output feeds
  another, what travels is a structured reference to ontology entities plus the
  producing step, never an unlabelled paragraph.  That is what keeps the
  provenance chain intact across a multi-step plan.
* **Declared dependencies.** ``depends_on`` is the only place execution order
  is expressed.  L4 derives parallelism from it; nothing runs in a hard-coded
  sequence.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from alm.core.ids import new_id, utc_now_iso
from alm.core.ids import session_id as new_session_id


class EntityRef(BaseModel):
    """A reference to an entity of the domain ontology.

    ``entity_type`` is an ontology class name (e.g. ``Contract``, ``Clause``);
    ``key`` identifies the instance within that class.
    """

    entity_type: str
    key: str = ""
    label: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    source: str = Field(default="", description="Step or component that produced it.")

    def qualified(self) -> str:
        return f"{self.entity_type}:{self.key}" if self.key else self.entity_type


class TaskEnvelope(BaseModel):
    """A structured task, as produced by the L1 interface layer."""

    task_id: str = Field(default_factory=lambda: new_id("tsk"))
    session_id: str = Field(default_factory=new_session_id)
    intent_text: str
    context: dict[str, Any] = Field(default_factory=dict)
    requested_outputs: list[str] = Field(default_factory=list)
    entities: list[EntityRef] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)

    # Governance identity — IBAC evaluates every context read against these.
    tenant_id: str = "default"
    principal: str = Field(default="", description="OIDC subject of the requester.")
    claims: list[str] = Field(default_factory=list)

    source: str = Field(default="api", description="api | cli | lip | eval")
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("intent_text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("intent_text must not be empty")
        return value.strip()


class SubTask(BaseModel):
    """One unit of work assigned to exactly one Expert Agent."""

    subtask_id: str = Field(default_factory=lambda: new_id("sub"))
    description: str
    retrieval_query: str = Field(
        default="",
        description=(
            "What to search the corpus for, as distinct from what to do. The "
            "description frames the task for the expert and is often mostly "
            "boilerplate from the capability declaration; using it as a search "
            "query would drown the user's actual words. Falls back to "
            "``description`` when unset."
        ),
    )
    domain: str = ""
    expert_id: str = ""
    capability_id: str = ""

    depends_on: list[str] = Field(
        default_factory=list,
        description="subtask_ids whose outputs are required as input.",
    )
    inputs: dict[str, Any] = Field(default_factory=dict)
    entities: list[EntityRef] = Field(default_factory=list)
    requested_outputs: list[str] = Field(default_factory=list)

    # Routing metadata — how this assignment was arrived at, and how sure L2 is.
    confidence: float = 0.0
    rationale: str = ""
    tier_hint: str = ""
    is_fallback: bool = Field(
        default=False,
        description="True when the orchestrator LLM handles it because no expert matched.",
    )

    @field_validator("description")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("subtask description must not be empty")
        return value.strip()

    def query(self) -> str:
        """What this subtask is actually *asking*, stripped of task framing.

        Retrieval and any question-aware backend must use this rather than
        ``description``: the description carries the capability's boilerplate,
        which matches the whole corpus equally well and therefore ranks nothing.
        """
        return self.retrieval_query or self.description


class ExecutionPlan(BaseModel):
    """The output of L2: subtasks, their dependencies, and routing confidence."""

    plan_id: str = Field(default_factory=lambda: new_id("pln"))
    task_id: str = ""
    session_id: str = ""
    subtasks: list[SubTask] = Field(default_factory=list)

    router_confidence: float = 0.0
    strategy: str = Field(
        default="federated",
        description="federated | single_expert | fallback_orchestrator",
    )
    fallback_reason: str = ""
    domains: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)

    def by_id(self, subtask_id: str) -> SubTask | None:
        for st in self.subtasks:
            if st.subtask_id == subtask_id:
                return st
        return None

    @property
    def is_fallback(self) -> bool:
        return self.strategy == "fallback_orchestrator" or all(
            st.is_fallback for st in self.subtasks
        )

    @property
    def expert_ids(self) -> list[str]:
        seen: list[str] = []
        for st in self.subtasks:
            if st.expert_id and st.expert_id not in seen:
                seen.append(st.expert_id)
        return seen

    def validate_dependencies(self) -> list[str]:
        """Return a list of human-readable problems with the declared graph.

        An empty list means the plan is structurally sound.  Cycle detection
        itself lives in :mod:`alm.orchestration.dag`; this catches the cheaper
        classes of error (dangling and self references) before a DAG is built.
        """
        problems: list[str] = []
        known = {st.subtask_id for st in self.subtasks}
        for st in self.subtasks:
            for dep in st.depends_on:
                if dep == st.subtask_id:
                    problems.append(f"subtask {st.subtask_id} depends on itself")
                elif dep not in known:
                    problems.append(
                        f"subtask {st.subtask_id} depends on unknown subtask {dep}"
                    )
        return problems
