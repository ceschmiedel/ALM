"""Scripted backend — canned responses for tests and reproducible fixtures.

Register rules on the model spec and every matching request returns the same
text.  Tests that need to assert on arbitration behaviour can therefore stage
an exact disagreement between two experts without invoking a model.

    alm model add legal-stub --backend scripted --param 'responses=[
        {"match": "liability", "response": "Clause 7.2 caps liability.", "confidence": 0.9}
    ]'
"""

from __future__ import annotations

import re
from typing import Any

from alm.core.jsonutil import dumps
from alm.models.backends.base import ChatBackend
from alm.models.spec import GenerationRequest, GenerationResult


class ScriptedBackend(ChatBackend):
    """Returns the first response whose pattern matches the prompt."""

    name = "scripted"

    async def _generate(self, request: GenerationRequest) -> GenerationResult:
        prompt = request.prompt_text()
        rules: list[dict[str, Any]] = list(self.spec.params.get("responses", []) or [])

        chosen: dict[str, Any] | None = None
        for rule in rules:
            pattern = rule.get("match")
            if pattern is None:
                chosen = rule
                break
            try:
                if re.search(str(pattern), prompt, re.IGNORECASE | re.DOTALL):
                    chosen = rule
                    break
            except re.error:
                if str(pattern).lower() in prompt.lower():
                    chosen = rule
                    break

        if chosen is None:
            default = self.spec.params.get("default_response")
            chosen = (
                {"response": default}
                if default is not None
                else {"response": "No scripted response matched this prompt."}
            )

        payload = chosen.get("response", "")
        if request.json_mode and not isinstance(payload, str):
            text = dumps(payload)
        elif request.json_mode and isinstance(payload, str):
            confidence = chosen.get("confidence")
            text = (
                payload
                if payload.lstrip().startswith(("{", "["))
                else dumps({"answer": payload, "confidence": confidence or 0.8})
            )
        elif isinstance(payload, str):
            text = payload
        else:
            text = dumps(payload)

        return GenerationResult(
            text=text,
            prompt_tokens=int(chosen.get("prompt_tokens", 0) or 0),
            completion_tokens=int(chosen.get("completion_tokens", 0) or 0),
            finish_reason="stop",
            raw={"rule": chosen.get("match", "<default>")},
        )

    async def health(self) -> bool:
        return True
