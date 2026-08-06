"""Identifier and timestamp helpers.

The runtime uses prefixed, sortable-ish identifiers so a trace read by a human
is self-describing: ``ses_…`` is a session, ``stp_…`` an execution step,
``nod_…`` a Context Graph node.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime


def new_id(prefix: str) -> str:
    """Return a new prefixed identifier, e.g. ``ses_3f9c1a2b…``."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def session_id() -> str:
    return new_id("ses")


def step_id() -> str:
    return new_id("stp")


def node_id() -> str:
    return new_id("nod")


def edge_id() -> str:
    return new_id("edg")


def run_id() -> str:
    return new_id("run")


def utc_now() -> datetime:
    """Timezone-aware current time (always UTC)."""
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp used across envelopes and audit records."""
    return utc_now().isoformat()


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Normalise free text into a stable slug usable as a graph key."""
    return _SLUG_RE.sub("-", text.strip().lower()).strip("-")
