"""Tolerant JSON handling for model output.

Small language models are exactly the population most likely to wrap JSON in
markdown fences, prepend a sentence, or emit a trailing comma.  Treating that
as a hard failure would inflate the fallback rate for reasons that have nothing
to do with domain competence, so the runtime repairs what is unambiguously
repairable and reports a clean failure otherwise.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def strip_fences(text: str) -> str:
    """Return the contents of the first fenced block, or the text unchanged."""
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _balanced_slice(text: str, open_ch: str, close_ch: str) -> str | None:
    """Extract the first balanced ``open_ch … close_ch`` region, string-aware."""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> Any | None:
    """Best-effort parse of a JSON value embedded in free-form model output.

    Returns ``None`` when nothing parseable is present — callers decide whether
    that is a retry, a fallback, or a hard error.
    """
    if not text:
        return None

    candidate = strip_fences(text)

    for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
        try:
            return json.loads(attempt)
        except (ValueError, TypeError):
            pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        sliced = _balanced_slice(candidate, open_ch, close_ch)
        if sliced is None:
            continue
        for attempt in (sliced, _TRAILING_COMMA_RE.sub(r"\1", sliced)):
            try:
                return json.loads(attempt)
            except (ValueError, TypeError):
                continue
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Like :func:`extract_json` but only accepts a JSON object."""
    value = extract_json(text)
    return value if isinstance(value, dict) else None


def to_jsonable(value: Any) -> Any:
    """Coerce arbitrary values into something ``json.dumps`` accepts."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(mode="json"))
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def dumps(value: Any, *, indent: int | None = None) -> str:
    """``json.dumps`` that never raises on unexpected types."""
    return json.dumps(to_jsonable(value), ensure_ascii=False, indent=indent)


def truncate(text: str, limit: int = 4000) -> str:
    """Shorten text for prompt inclusion, marking the elision."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} characters truncated]"
