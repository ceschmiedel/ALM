"""Append-only audit log.

Every IBAC decision is recorded, allow and deny alike.  Recording only denials
would make the log useless for the question auditors actually ask, which is not
"what was blocked?" but "what did this agent see, and under what authority?"

The log is append-only by contract: there is no update or delete path in this
module.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from alm.core.ids import utc_now
from alm.persistence.database import session_scope
from alm.persistence.models import AuditRow

logger = logging.getLogger(__name__)


class AuditRecord:
    """One decision, as returned by queries."""

    __slots__ = (
        "id",
        "timestamp",
        "session_id",
        "tenant_id",
        "evaluation_point",
        "principal",
        "expert_id",
        "resource",
        "action",
        "decision",
        "reason",
        "matched_policies",
        "detail",
    )

    def __init__(self, row: AuditRow) -> None:
        self.id = row.id
        self.timestamp = row.timestamp
        self.session_id = row.session_id
        self.tenant_id = row.tenant_id
        self.evaluation_point = row.evaluation_point
        self.principal = row.principal
        self.expert_id = row.expert_id
        self.resource = row.resource
        self.action = row.action
        self.decision = row.decision
        self.reason = row.reason
        self.matched_policies = list(row.matched_policies or [])
        self.detail = dict(row.detail or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else "",
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "evaluation_point": self.evaluation_point,
            "principal": self.principal,
            "expert_id": self.expert_id,
            "resource": self.resource,
            "action": self.action,
            "decision": self.decision,
            "reason": self.reason,
            "matched_policies": self.matched_policies,
            "detail": self.detail,
        }

    def line(self) -> str:
        stamp = self.timestamp.strftime("%H:%M:%S") if self.timestamp else "--:--:--"
        mark = "✓" if self.decision == "allow" else "✗"
        who = self.expert_id or self.principal or "-"
        return (
            f"{stamp} {mark} {self.decision:<6} {self.evaluation_point:<18} "
            f"{who:<22} {self.action:<8} {self.resource}"
        )


class AuditLog:
    """Writer and reader for the IBAC decision log."""

    def __init__(self, tenant_id: str = "default", enabled: bool = True) -> None:
        self.tenant_id = tenant_id
        self.enabled = enabled

    def record(
        self,
        *,
        evaluation_point: str,
        decision: str,
        session_id: str = "",
        principal: str = "",
        expert_id: str = "",
        resource: str = "",
        action: str = "",
        reason: str = "",
        matched_policies: Sequence[str] = (),
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Append one decision.  Never raises — governance must not break a run."""
        if not self.enabled:
            return
        try:
            with session_scope() as session:
                session.add(
                    AuditRow(
                        tenant_id=self.tenant_id,
                        session_id=session_id,
                        timestamp=utc_now(),
                        evaluation_point=evaluation_point,
                        principal=principal,
                        expert_id=expert_id,
                        resource=resource[:512],
                        action=action,
                        decision=decision,
                        reason=reason,
                        matched_policies=list(matched_policies),
                        detail=detail or {},
                    )
                )
        except Exception:  # pragma: no cover - audit must not break the run
            logger.warning("Failed to write audit record", exc_info=True)

    # -- queries -----------------------------------------------------------

    def query(
        self,
        *,
        session_id: str = "",
        expert_id: str = "",
        decision: str = "",
        evaluation_point: str = "",
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[AuditRecord]:
        with session_scope() as session:
            stmt = select(AuditRow).where(AuditRow.tenant_id == self.tenant_id)
            if session_id:
                stmt = stmt.where(AuditRow.session_id == session_id)
            if expert_id:
                stmt = stmt.where(AuditRow.expert_id == expert_id)
            if decision:
                stmt = stmt.where(AuditRow.decision == decision)
            if evaluation_point:
                stmt = stmt.where(AuditRow.evaluation_point == evaluation_point)
            if since is not None:
                stmt = stmt.where(AuditRow.timestamp >= since)
            rows = session.execute(
                stmt.order_by(AuditRow.timestamp.desc(), AuditRow.id.desc()).limit(limit)
            ).scalars().all()
            return [AuditRecord(r) for r in rows]

    def for_session(self, session_id: str, limit: int = 1000) -> list[AuditRecord]:
        records = self.query(session_id=session_id, limit=limit)
        records.reverse()  # chronological, as a trail should read
        return records

    def summary(self, *, hours: int = 24) -> dict[str, Any]:
        since = utc_now() - timedelta(hours=hours)
        records = self.query(since=since, limit=100_000)
        allowed = sum(1 for r in records if r.decision == "allow")
        denied = len(records) - allowed
        by_expert: dict[str, dict[str, int]] = {}
        for record in records:
            bucket = by_expert.setdefault(record.expert_id or "-", {"allow": 0, "deny": 0})
            bucket[record.decision if record.decision in bucket else "allow"] += 1
        return {
            "window_hours": hours,
            "total": len(records),
            "allowed": allowed,
            "denied": denied,
            "deny_rate": round(denied / len(records), 4) if records else 0.0,
            "by_expert": by_expert,
        }

    def export_jsonl(self, *, limit: int = 100_000, **filters: Any) -> Iterator[str]:
        """Stream the log as JSON Lines, for handing to an external SIEM."""
        for record in reversed(self.query(limit=limit, **filters)):
            yield json.dumps(record.to_dict(), ensure_ascii=False)

    def count(self) -> int:
        with session_scope() as session:
            rows = session.execute(
                select(AuditRow.id).where(AuditRow.tenant_id == self.tenant_id)
            ).scalars().all()
            return len(rows)
