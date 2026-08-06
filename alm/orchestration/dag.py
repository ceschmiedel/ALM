"""Execution DAG.

The plan produced by L2 becomes a directed acyclic graph where each node is one
Expert Agent invocation and each edge is a data dependency.  Execution then
respects that structure naturally:

* nodes with no dependency between them run **in parallel**, cutting total latency;
* chained nodes **wait** for their predecessor, whose output becomes their input;
* **join points** gather several experts' outputs for a downstream step or for
  the final synthesis.

Nothing here is pre-declared.  The topology is derived from the plan on every
request, so the same federation answers a one-step question with one node and a
four-domain question with a four-node graph.
"""

from __future__ import annotations

from collections.abc import Iterator

from alm.core.errors import CyclicPlanError, OrchestrationError
from alm.protocol.task import ExecutionPlan, SubTask


class ExecutionDAG:
    """A validated dependency graph over the subtasks of a plan."""

    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan
        self.nodes: dict[str, SubTask] = {st.subtask_id: st for st in plan.subtasks}
        if not self.nodes:
            raise OrchestrationError(
                "cannot build an execution graph from an empty plan", plan_id=plan.plan_id
            )
        self.edges: dict[str, list[str]] = {
            st.subtask_id: [d for d in st.depends_on if d in self.nodes]
            for st in plan.subtasks
        }
        self._levels: list[list[str]] | None = None
        self.validate()

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Reject dangling references and cycles before anything executes."""
        for subtask_id, dependencies in self.edges.items():
            for dependency in dependencies:
                if dependency == subtask_id:
                    raise CyclicPlanError(
                        f"subtask {subtask_id} depends on itself", subtask_id=subtask_id
                    )
        cycle = self._find_cycle()
        if cycle:
            raise CyclicPlanError(
                "execution plan contains a dependency cycle: " + " → ".join(cycle),
                cycle=cycle,
                plan_id=self.plan.plan_id,
            )

    def _find_cycle(self) -> list[str]:
        WHITE, GREY, BLACK = 0, 1, 2
        colour = dict.fromkeys(self.nodes, WHITE)
        stack: list[str] = []

        def visit(node: str) -> list[str]:
            colour[node] = GREY
            stack.append(node)
            for dependency in self.edges.get(node, []):
                if colour[dependency] == GREY:
                    start = stack.index(dependency)
                    return [*stack[start:], dependency]
                if colour[dependency] == WHITE:
                    found = visit(dependency)
                    if found:
                        return found
            colour[node] = BLACK
            stack.pop()
            return []

        for node in self.nodes:
            if colour[node] == WHITE:
                found = visit(node)
                if found:
                    return found
        return []

    # -- topology ----------------------------------------------------------

    def levels(self) -> list[list[str]]:
        """Group subtasks into waves that can each run concurrently.

        Level 0 has no dependencies; level *n* depends only on levels below it.
        The number of levels is the critical path; the width of a level is the
        parallelism available at that step.
        """
        if self._levels is not None:
            return self._levels

        remaining = {node: set(deps) for node, deps in self.edges.items()}
        done: set[str] = set()
        levels: list[list[str]] = []

        while remaining:
            ready = sorted(
                node for node, deps in remaining.items() if not (deps - done)
            )
            if not ready:
                # validate() rules this out; belt and braces for hand-built DAGs.
                raise CyclicPlanError(
                    "cannot schedule the remaining subtasks; the graph is cyclic",
                    remaining=sorted(remaining),
                )
            levels.append(ready)
            done.update(ready)
            for node in ready:
                remaining.pop(node, None)

        self._levels = levels
        return levels

    def dependencies_of(self, subtask_id: str) -> list[SubTask]:
        return [self.nodes[d] for d in self.edges.get(subtask_id, []) if d in self.nodes]

    def dependents_of(self, subtask_id: str) -> list[SubTask]:
        return [
            self.nodes[node]
            for node, deps in self.edges.items()
            if subtask_id in deps
        ]

    def terminal_nodes(self) -> list[SubTask]:
        """Subtasks nothing else depends on — the plan's outputs."""
        depended_on = {d for deps in self.edges.values() for d in deps}
        return [st for sid, st in self.nodes.items() if sid not in depended_on]

    # -- metrics -----------------------------------------------------------

    @property
    def width(self) -> int:
        """Maximum parallelism the plan allows."""
        return max((len(level) for level in self.levels()), default=0)

    @property
    def depth(self) -> int:
        """Length of the critical path, in waves."""
        return len(self.levels())

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[SubTask]:
        for level in self.levels():
            for node in level:
                yield self.nodes[node]

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan.plan_id,
            "nodes": [
                {
                    "subtask_id": st.subtask_id,
                    "expert_id": st.expert_id or "(orchestrator fallback)",
                    "capability_id": st.capability_id,
                    "domain": st.domain,
                    "depends_on": st.depends_on,
                    "confidence": round(st.confidence, 4),
                }
                for st in self.nodes.values()
            ],
            "levels": self.levels(),
            "width": self.width,
            "depth": self.depth,
        }

    def render(self) -> str:
        """ASCII rendering of the execution waves, for the CLI."""
        lines: list[str] = []
        for index, level in enumerate(self.levels()):
            marker = "└─" if index == len(self.levels()) - 1 else "├─"
            names = ", ".join(
                self.nodes[node].expert_id or "orchestrator" for node in level
            )
            suffix = " (parallel)" if len(level) > 1 else ""
            lines.append(f" {marker} wave {index + 1}: {names}{suffix}")
        return "\n".join(lines)


def build_dag(plan: ExecutionPlan) -> ExecutionDAG:
    """Build and validate the execution graph for a plan."""
    return ExecutionDAG(plan)
