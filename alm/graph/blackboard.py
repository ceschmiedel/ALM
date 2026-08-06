"""Session blackboard — shared execution state, held in the graph.

While a DAG runs, intermediate results do not live in a process's memory.  They
are written here, so that:

* any later step in the plan can read what earlier steps produced,
* arbitration sees the entire chain of reasoning rather than only the leaves, and
* the audit trail is complete without a separate logging pass.

The graph is long-term memory and the current run's whiteboard at the same time.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from alm.persistence.database import session_scope
from alm.persistence.models import BlackboardRow


class SessionBlackboard:
    """Key/value scratch space scoped to one federation session."""

    def __init__(self, session_id: str, tenant_id: str = "default") -> None:
        self.session_id = session_id
        self.tenant_id = tenant_id

    def put(
        self,
        key: str,
        value: Any,
        *,
        produced_by: str = "",
        subtask_id: str = "",
    ) -> None:
        """Write (or overwrite) a value under ``key``."""
        payload = value if isinstance(value, (dict, list)) else {"value": value}
        with session_scope() as session:
            row = session.execute(
                select(BlackboardRow).where(
                    BlackboardRow.session_id == self.session_id,
                    BlackboardRow.key == key,
                )
            ).scalar_one_or_none()
            if row is None:
                row = BlackboardRow(
                    session_id=self.session_id,
                    tenant_id=self.tenant_id,
                    key=key,
                )
                session.add(row)
            row.value = payload
            row.produced_by = produced_by
            row.subtask_id = subtask_id

    def get(self, key: str, default: Any = None) -> Any:
        with session_scope() as session:
            row = session.execute(
                select(BlackboardRow).where(
                    BlackboardRow.session_id == self.session_id,
                    BlackboardRow.key == key,
                )
            ).scalar_one_or_none()
            if row is None:
                return default
            value = row.value
            if isinstance(value, dict) and set(value.keys()) == {"value"}:
                return value["value"]
            return value

    def has(self, key: str) -> bool:
        return self.get(key, _MISSING) is not _MISSING

    def keys(self) -> list[str]:
        with session_scope() as session:
            rows = session.execute(
                select(BlackboardRow.key)
                .where(BlackboardRow.session_id == self.session_id)
                .order_by(BlackboardRow.id)
            ).scalars().all()
            return list(rows)

    def snapshot(self) -> dict[str, Any]:
        """Return the whole board, with provenance, for the audit trail."""
        with session_scope() as session:
            rows = session.execute(
                select(BlackboardRow)
                .where(BlackboardRow.session_id == self.session_id)
                .order_by(BlackboardRow.id)
            ).scalars().all()
            return {
                row.key: {
                    "value": row.value,
                    "produced_by": row.produced_by,
                    "subtask_id": row.subtask_id,
                }
                for row in rows
            }

    def values(self) -> dict[str, Any]:
        """Return the board as a plain key → value mapping."""
        return {k: v["value"] for k, v in self.snapshot().items()}

    def clear(self) -> int:
        """Erase the board.  Returns how many entries were removed."""
        with session_scope() as session:
            rows = session.execute(
                select(BlackboardRow).where(BlackboardRow.session_id == self.session_id)
            ).scalars().all()
            for row in rows:
                session.delete(row)
            return len(rows)


class _Missing:
    __slots__ = ()


_MISSING = _Missing()
