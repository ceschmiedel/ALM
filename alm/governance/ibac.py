"""IBAC — Intention-Based Access Control for a federation.

In a federation, access control is not only about the *user*.  Each Expert
Agent is an independent reader of context and an independent actor, so the
governed question is per-agent: what may **this expert** read, and what may it
do?  An expert in one domain does not see another domain's data without an
explicit grant, and every access is logged.  That contains the blast radius and
produces the trail audit and compliance require.

Evaluation is **deterministic**.  A context read happens tens to hundreds of
times per session, and a policy engine that needed an LLM call per fragment
would be neither affordable nor reproducible.  Policies are matched by
selectors, deny wins, and identical inputs always produce an identical decision
— which is the property an auditor is really asking about.

Semantic evaluation is available where it is affordable and useful: intent
admission runs once per session, so :meth:`IBACEngine.evaluate_intent_semantic`
can additionally ask the orchestrator model whether a request violates the
*spirit* of a natural-language policy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from alm.cmrag.store import RetrievedChunk
from alm.core.ids import utc_now_iso
from alm.core.jsonutil import extract_json_object
from alm.governance.audit import AuditLog
from alm.governance.policies import (
    Effect,
    EvaluationPoint,
    Policy,
    PolicySet,
)

logger = logging.getLogger(__name__)


class AccessRequest(BaseModel):
    """One thing an expert (or the runtime) wants to do."""

    evaluation_point: EvaluationPoint
    action: str = "read"

    # Who is asking
    expert_id: str = ""
    domain: str = ""
    principal: str = ""
    claims: list[str] = Field(default_factory=list)
    tenant_id: str = "default"

    # What is being asked for
    resource_id: str = ""
    resource_domain: str = ""
    entities: list[str] = Field(default_factory=list)
    sensitivity: str = "internal"
    capability_id: str = ""
    model_id: str = ""

    session_id: str = ""
    intent_text: str = ""
    context: dict[str, Any] = Field(default_factory=dict)

    def resource_label(self) -> str:
        parts = []
        if self.resource_domain:
            parts.append(f"domain:{self.resource_domain}")
        if self.capability_id:
            parts.append(f"capability:{self.capability_id}")
        if self.model_id:
            parts.append(f"model:{self.model_id}")
        if self.entities:
            parts.append("entity:" + ",".join(self.entities[:4]))
        if self.sensitivity:
            parts.append(f"sensitivity:{self.sensitivity}")
        if self.resource_id:
            parts.append(self.resource_id)
        return " ".join(parts) or "*"


class AccessDecision(BaseModel):
    """The outcome, with enough context to explain itself."""

    allowed: bool
    effect: Effect
    evaluation_point: EvaluationPoint
    reason: str = ""
    matched_policies: list[str] = Field(default_factory=list)
    requires_approval: bool = False
    decided_at: str = Field(default_factory=utc_now_iso)

    @property
    def decision(self) -> str:
        return "allow" if self.allowed else "deny"


_SEMANTIC_SYSTEM_PROMPT = """\
You are the IBAC governance engine of an ALM federation. Decide whether a
request should be ALLOWED or DENIED against the organisation's policies.

## Active policies
{policies}

## Rules
1. Judge whether the request semantically violates a DENY policy, even when the
   policy has no explicit selector for this case. Use its description.
2. A DENY that applies wins.
3. If no policy applies, the decision is ALLOW.
4. Never invent policies that are not listed.

Return ONLY a JSON object:
{{"decision": "allow" | "deny", "reason": "<one sentence>", "matching_policies": ["<id>"]}}
"""


class IBACEngine:
    """Evaluates access requests against a :class:`PolicySet`."""

    def __init__(
        self,
        policies: PolicySet | None = None,
        *,
        tenant_id: str = "default",
        audit: AuditLog | None = None,
        enabled: bool = True,
        default_effect: Effect | str = Effect.ALLOW,
    ) -> None:
        self.policies = policies or PolicySet()
        self.tenant_id = tenant_id
        self.audit = audit or AuditLog(tenant_id=tenant_id)
        self.enabled = enabled
        self.default_effect = Effect(str(default_effect))

    # -- policy management -------------------------------------------------

    def add_policy(self, policy: Policy) -> None:
        self.policies.add(policy)

    def load(self, policies: PolicySet) -> None:
        self.policies = self.policies.merge(policies)

    # -- evaluation --------------------------------------------------------

    def evaluate(self, request: AccessRequest, *, record: bool = True) -> AccessDecision:
        """Decide one request.  Deterministic and side-effect free apart from audit."""
        if not self.enabled:
            decision = AccessDecision(
                allowed=True,
                effect=Effect.ALLOW,
                evaluation_point=request.evaluation_point,
                reason="governance disabled",
            )
            return decision

        candidates = self.policies.enabled_for(request.evaluation_point, request.tenant_id)
        for policy in candidates:
            if not policy.matches_claims(request.claims):
                continue
            if not policy.matches_subject(
                expert_id=request.expert_id,
                domain=request.domain,
                principal=request.principal,
            ):
                continue
            if not policy.matches_action(request.action):
                continue
            if not policy.matches_resource(
                domain=request.resource_domain or request.domain,
                entities=request.entities,
                sensitivity=request.sensitivity,
                capability_id=request.capability_id,
                model_id=request.model_id,
                resource_id=request.resource_id,
            ):
                continue

            decision = AccessDecision(
                allowed=policy.effect != Effect.DENY,
                effect=policy.effect,
                evaluation_point=request.evaluation_point,
                reason=policy.reason or policy.description or f"policy {policy.id}",
                matched_policies=[policy.id],
                requires_approval=policy.effect == Effect.REQUIRE_APPROVAL,
            )
            if record:
                self._record(request, decision)
            return decision

        decision = AccessDecision(
            allowed=self.default_effect != Effect.DENY,
            effect=self.default_effect,
            evaluation_point=request.evaluation_point,
            reason="no policy matched; default applied",
        )
        if record:
            self._record(request, decision)
        return decision

    def _record(self, request: AccessRequest, decision: AccessDecision) -> None:
        self.audit.record(
            evaluation_point=str(request.evaluation_point),
            decision=decision.decision,
            session_id=request.session_id,
            principal=request.principal,
            expert_id=request.expert_id,
            resource=request.resource_label(),
            action=request.action,
            reason=decision.reason,
            matched_policies=decision.matched_policies,
            detail={
                "domain": request.domain,
                "resource_domain": request.resource_domain,
                "entities": request.entities,
                "sensitivity": request.sensitivity,
                "requires_approval": decision.requires_approval,
            },
        )

    # -- convenience wrappers ---------------------------------------------

    def may_read_context(
        self,
        *,
        expert_id: str,
        domain: str,
        chunk: RetrievedChunk,
        principal: str = "",
        claims: Sequence[str] = (),
        session_id: str = "",
        record: bool = True,
    ) -> AccessDecision:
        """Whether ``expert_id`` may see one retrieved fragment."""
        return self.evaluate(
            AccessRequest(
                evaluation_point=EvaluationPoint.CONTEXT_READ,
                action="read",
                expert_id=expert_id,
                domain=domain,
                principal=principal,
                claims=list(claims),
                tenant_id=self.tenant_id,
                resource_id=chunk.chunk_id,
                resource_domain=chunk.domain,
                entities=list(chunk.entities),
                sensitivity=chunk.sensitivity,
                session_id=session_id,
            ),
            record=record,
        )

    def context_filter(
        self,
        *,
        expert_id: str,
        domain: str,
        principal: str = "",
        claims: Sequence[str] = (),
        session_id: str = "",
    ):
        """Build the access filter CMRAG applies to every candidate fragment.

        Denials are recorded individually; that is the point of per-agent
        governance, and it is what lets an auditor reconstruct exactly which
        evidence an expert was and was not shown.
        """

        def _filter(chunk: RetrievedChunk) -> bool:
            decision = self.may_read_context(
                expert_id=expert_id,
                domain=domain,
                chunk=chunk,
                principal=principal,
                claims=claims,
                session_id=session_id,
                # Recording every allowed fragment would drown the log; denials
                # and the aggregate per-expert read are what carry information.
                record=False,
            )
            if not decision.allowed:
                self.audit.record(
                    evaluation_point=str(EvaluationPoint.CONTEXT_READ),
                    decision="deny",
                    session_id=session_id,
                    principal=principal,
                    expert_id=expert_id,
                    resource=f"chunk:{chunk.chunk_id} domain:{chunk.domain} "
                    f"sensitivity:{chunk.sensitivity}",
                    action="read",
                    reason=decision.reason,
                    matched_policies=decision.matched_policies,
                    detail={"entities": chunk.entities},
                )
            return decision.allowed

        return _filter

    def may_execute(
        self,
        *,
        expert_id: str,
        domain: str,
        capability_id: str = "",
        model_id: str = "",
        principal: str = "",
        claims: Sequence[str] = (),
        session_id: str = "",
        intent_text: str = "",
    ) -> AccessDecision:
        return self.evaluate(
            AccessRequest(
                evaluation_point=EvaluationPoint.EXPERT_EXECUTION,
                action="execute",
                expert_id=expert_id,
                domain=domain,
                capability_id=capability_id,
                model_id=model_id,
                principal=principal,
                claims=list(claims),
                tenant_id=self.tenant_id,
                session_id=session_id,
                intent_text=intent_text,
            )
        )

    def admit_intent(
        self,
        *,
        intent_text: str,
        principal: str = "",
        claims: Sequence[str] = (),
        session_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> AccessDecision:
        return self.evaluate(
            AccessRequest(
                evaluation_point=EvaluationPoint.INTENT_ADMISSION,
                action="submit",
                principal=principal,
                claims=list(claims),
                tenant_id=self.tenant_id,
                session_id=session_id,
                intent_text=intent_text,
                context=context or {},
            )
        )

    def may_emit(
        self,
        *,
        domains: Sequence[str],
        entities: Sequence[str],
        sensitivity: str = "internal",
        principal: str = "",
        claims: Sequence[str] = (),
        session_id: str = "",
    ) -> AccessDecision:
        """Last gate: may this answer leave the federation for this principal?"""
        return self.evaluate(
            AccessRequest(
                evaluation_point=EvaluationPoint.ARTIFACT_EMISSION,
                action="emit",
                principal=principal,
                claims=list(claims),
                tenant_id=self.tenant_id,
                resource_domain=domains[0] if domains else "",
                entities=list(entities),
                sensitivity=sensitivity,
                session_id=session_id,
            )
        )

    # -- optional semantic pass -------------------------------------------

    async def evaluate_intent_semantic(
        self,
        *,
        intent_text: str,
        serving: Any,
        principal: str = "",
        claims: Sequence[str] = (),
        session_id: str = "",
    ) -> AccessDecision:
        """Ask the orchestrator whether an intent violates a policy in spirit.

        Runs at most once per session, and only augments the deterministic pass:
        a deterministic deny is never overturned here.  When no orchestrator is
        configured the deterministic decision stands unchanged.
        """
        baseline = self.admit_intent(
            intent_text=intent_text,
            principal=principal,
            claims=claims,
            session_id=session_id,
        )
        if not baseline.allowed or not self.enabled:
            return baseline

        applicable = [
            p
            for p in self.policies.enabled_for(
                EvaluationPoint.INTENT_ADMISSION, self.tenant_id
            )
            if p.effect == Effect.DENY
        ]
        if not applicable:
            return baseline

        from alm.models.spec import GenerationRequest, Message
        from alm.models.tiers import ModelTier

        policies_block = "\n".join(
            f"- [{p.id}] ({p.effect}) {p.description or p.reason}" for p in applicable
        )
        request = GenerationRequest(
            messages=[
                Message(
                    role="system",
                    content=_SEMANTIC_SYSTEM_PROMPT.format(policies=policies_block),
                ),
                Message(
                    role="user",
                    content=f"Principal: {principal or 'anonymous'}\n"
                    f"Claims: {', '.join(claims) or 'none'}\n"
                    f"Intent: {intent_text}",
                ),
            ],
            json_mode=True,
            temperature=0.0,
        )
        try:
            result = await serving.generate(
                request,
                tier=ModelTier.ORCHESTRATOR,
                component="ibac.semantic",
                allow_escalation=False,
            )
        except Exception:
            logger.info("Semantic IBAC pass unavailable; deterministic decision stands")
            return baseline

        parsed = extract_json_object(result.text) or {}
        if str(parsed.get("decision", "allow")).lower() != "deny":
            return baseline

        decision = AccessDecision(
            allowed=False,
            effect=Effect.DENY,
            evaluation_point=EvaluationPoint.INTENT_ADMISSION,
            reason=str(parsed.get("reason", "semantically violates an active policy")),
            matched_policies=[str(p) for p in parsed.get("matching_policies", [])],
        )
        self.audit.record(
            evaluation_point=str(EvaluationPoint.INTENT_ADMISSION),
            decision="deny",
            session_id=session_id,
            principal=principal,
            resource="intent",
            action="submit",
            reason=decision.reason,
            matched_policies=decision.matched_policies,
            detail={"semantic": True, "intent_text": intent_text[:500]},
        )
        return decision
