"""Context passing between steps of a plan.

When one expert's result becomes another's input, what crosses the boundary is
a **structured LIP payload**, not loose text: each contribution keeps its
producing expert, its domain, its confidence and its entity references.

Two things depend on that. Traceability — it stays knowable which agent
produced which piece — and quality: threading free text through many hops
degrades it, and a downstream specialist that cannot tell one source from
another cannot flag a conflict with its own analysis.

Everything written here also lands on the session blackboard in the Context
Graph, so intermediate state is never hidden inside a process.
"""

from __future__ import annotations

from typing import Any

from alm.graph.blackboard import SessionBlackboard
from alm.protocol.answer import ExpertAnswer
from alm.protocol.lip import upstream_context
from alm.protocol.task import SubTask


class ExecutionContext:
    """Carries answers between DAG nodes and mirrors them to the blackboard."""

    def __init__(
        self,
        session_id: str,
        *,
        blackboard: SessionBlackboard | None = None,
        tenant_id: str = "default",
    ) -> None:
        self.session_id = session_id
        self.blackboard = blackboard or SessionBlackboard(session_id, tenant_id)
        self._answers: dict[str, ExpertAnswer] = {}

    # -- writing -----------------------------------------------------------

    def record(self, answer: ExpertAnswer) -> None:
        """Store an answer and publish it to the shared blackboard."""
        self._answers[answer.subtask_id] = answer
        self.blackboard.put(
            f"answer:{answer.subtask_id}",
            {
                "expert_id": answer.expert_id,
                "domain": answer.domain,
                "capability_id": answer.capability_id,
                "content": answer.content,
                "structured": answer.structured,
                "confidence": answer.confidence,
                "status": answer.status,
                "citations": [c.model_dump(mode="json") for c in answer.citations],
            },
            produced_by=answer.expert_id,
            subtask_id=answer.subtask_id,
        )

    # -- reading -----------------------------------------------------------

    def answer_for(self, subtask_id: str) -> ExpertAnswer | None:
        return self._answers.get(subtask_id)

    def answers(self) -> list[ExpertAnswer]:
        return list(self._answers.values())

    def successful(self) -> list[ExpertAnswer]:
        return [a for a in self._answers.values() if a.ok]

    def inputs_for(self, subtask: SubTask) -> dict[str, Any]:
        """Build the structured upstream payload for one subtask.

        Only *declared* dependencies are passed. An expert seeing everything
        every other expert produced would defeat both the noise reduction that
        makes specialists accurate and the access containment IBAC provides.
        """
        upstream = [
            self._answers[dependency]
            for dependency in subtask.depends_on
            if dependency in self._answers
        ]
        payload = upstream_context(upstream)
        payload["inputs"] = dict(subtask.inputs)
        return payload

    def missing_dependencies(self, subtask: SubTask) -> list[str]:
        """Declared dependencies that produced no usable answer."""
        return [
            dependency
            for dependency in subtask.depends_on
            if dependency not in self._answers
            or not self._answers[dependency].ok
        ]

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return self.blackboard.snapshot()

    def clear(self) -> None:
        self._answers.clear()
        self.blackboard.clear()
