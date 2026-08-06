"""IBAC policy model and selector matching.

Policies are declarative and live in a domain pack, not in code, because the
question "may the finance expert read HR records?" is a governance decision that
compliance owns and must be able to read, review and change without a deploy.

Selector grammar (used by ``subjects``, ``resources`` and ``actions``):

``*``
    Matches anything.
``expert:contracts-expert`` / ``contracts-expert``
    A specific Expert Agent.
``domain:finance``
    Everything belonging to a domain — the expert, its capabilities, its corpus.
``entity:EmploymentContract``
    Anything annotated with that ontology entity type.
``sensitivity:restricted``
    Anything classified at that level or above.
``capability:analyze_clause`` / ``model:legal-lora``
    A specific capability or model.

Deny wins.  Within the same effect, higher ``priority`` wins.  That ordering is
what makes a broad allow safe to write next to a narrow deny.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from alm.core.errors import ALMError

#: Ordered from least to most restrictive; a selector matches at or above its level.
SENSITIVITY_ORDER = ("public", "internal", "confidential", "restricted")


class PolicyError(ALMError):
    """Malformed policy definition."""

    code = "policy_error"


class Effect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class EvaluationPoint(StrEnum):
    """Where in the lifecycle a policy is consulted.

    These mirror the LIP evaluation points, extended with the two that only
    exist once there is a federation: reading another domain's context, and
    arbitrating between experts.
    """

    INTENT_ADMISSION = "intent_admission"
    ROUTING = "routing"
    CONTEXT_READ = "context_read"
    EXPERT_EXECUTION = "expert_execution"
    ARBITRATION = "arbitration"
    ARTIFACT_EMISSION = "artifact_emission"


def sensitivity_rank(level: str) -> int:
    try:
        return SENSITIVITY_ORDER.index((level or "internal").lower())
    except ValueError:
        return SENSITIVITY_ORDER.index("internal")


class Policy(BaseModel):
    """One declarative access rule."""

    id: str
    description: str = ""
    effect: Effect = Effect.DENY
    priority: int = 100

    evaluation_points: list[EvaluationPoint] = Field(
        default_factory=list,
        description="Empty means the policy applies at every evaluation point.",
    )
    subjects: list[str] = Field(default_factory=lambda: ["*"])
    resources: list[str] = Field(default_factory=lambda: ["*"])
    actions: list[str] = Field(default_factory=lambda: ["*"])

    when_claims: list[str] = Field(
        default_factory=list,
        description="Policy applies only if the principal holds all of these claims.",
    )
    unless_claims: list[str] = Field(
        default_factory=list,
        description="Policy is skipped if the principal holds any of these claims.",
    )
    tenants: list[str] = Field(default_factory=list)
    reason: str = ""
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("policy id must not be empty")
        return value.strip()

    # -- matching ----------------------------------------------------------

    def applies_at(self, point: EvaluationPoint) -> bool:
        return not self.evaluation_points or point in self.evaluation_points

    def matches_subject(self, *, expert_id: str, domain: str, principal: str) -> bool:
        return any(
            _match_subject(selector, expert_id=expert_id, domain=domain, principal=principal)
            for selector in self.subjects
        )

    def matches_resource(
        self,
        *,
        domain: str = "",
        entities: Sequence[str] = (),
        sensitivity: str = "",
        capability_id: str = "",
        model_id: str = "",
        resource_id: str = "",
    ) -> bool:
        return any(
            _match_resource(
                selector,
                domain=domain,
                entities=entities,
                sensitivity=sensitivity,
                capability_id=capability_id,
                model_id=model_id,
                resource_id=resource_id,
            )
            for selector in self.resources
        )

    def matches_action(self, action: str) -> bool:
        return any(s == "*" or s == action for s in self.actions)

    def matches_claims(self, claims: Sequence[str]) -> bool:
        held = set(claims)
        if self.unless_claims and held & set(self.unless_claims):
            return False
        if self.when_claims and not set(self.when_claims).issubset(held):
            return False
        return True

    def matches_tenant(self, tenant_id: str) -> bool:
        return not self.tenants or tenant_id in self.tenants


def _match_subject(selector: str, *, expert_id: str, domain: str, principal: str) -> bool:
    selector = selector.strip()
    if selector in {"*", ""}:
        return True
    if selector.startswith("expert:"):
        return selector[7:] == expert_id
    if selector.startswith("domain:"):
        return selector[7:] == domain
    if selector.startswith("principal:"):
        return selector[10:] == principal
    return selector == expert_id


def _match_resource(
    selector: str,
    *,
    domain: str,
    entities: Sequence[str],
    sensitivity: str,
    capability_id: str,
    model_id: str,
    resource_id: str,
) -> bool:
    selector = selector.strip()
    if selector in {"*", ""}:
        return True
    if selector.startswith("domain:"):
        return selector[7:] == domain
    if selector.startswith("entity:"):
        return selector[7:] in set(entities)
    if selector.startswith("sensitivity:"):
        # "sensitivity:confidential" also covers restricted — a policy guarding
        # a level must not leave the stricter levels unguarded.
        return sensitivity_rank(sensitivity) >= sensitivity_rank(selector[12:])
    if selector.startswith("capability:"):
        return selector[11:] == capability_id
    if selector.startswith("model:"):
        return selector[6:] == model_id
    return selector == resource_id


class PolicySet(BaseModel):
    """An ordered collection of policies."""

    policies: list[Policy] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicySet:
        if not isinstance(data, dict):
            raise PolicyError("policy document must be a mapping")
        raw = data.get("policies", data.get("rules", []))
        if not isinstance(raw, list):
            raise PolicyError("'policies' must be a list")
        try:
            return cls(policies=[Policy.model_validate(item) for item in raw])
        except Exception as exc:
            raise PolicyError(f"invalid policy definition: {exc}") from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicySet:
        file_path = Path(path)
        if not file_path.exists():
            raise PolicyError(f"policy file not found: {file_path}", path=str(file_path))
        try:
            data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise PolicyError(f"invalid YAML in {file_path}: {exc}") from exc
        return cls.from_dict(data)

    def enabled_for(self, point: EvaluationPoint, tenant_id: str) -> list[Policy]:
        """Policies applicable at a point, ordered so the winner comes first."""
        candidates = [
            p
            for p in self.policies
            if p.enabled and p.applies_at(point) and p.matches_tenant(tenant_id)
        ]
        # Deny before allow, then by descending priority: the first match wins.
        effect_rank = {Effect.DENY: 0, Effect.REQUIRE_APPROVAL: 1, Effect.ALLOW: 2}
        candidates.sort(key=lambda p: (effect_rank[p.effect], -p.priority, p.id))
        return candidates

    def by_id(self, policy_id: str) -> Policy | None:
        for policy in self.policies:
            if policy.id == policy_id:
                return policy
        return None

    def add(self, policy: Policy) -> None:
        self.policies = [p for p in self.policies if p.id != policy.id]
        self.policies.append(policy)

    def remove(self, policy_id: str) -> bool:
        before = len(self.policies)
        self.policies = [p for p in self.policies if p.id != policy_id]
        return len(self.policies) != before

    def merge(self, other: PolicySet) -> PolicySet:
        merged = PolicySet(policies=list(self.policies))
        for policy in other.policies:
            merged.add(policy)
        return merged

    def __len__(self) -> int:
        return len(self.policies)
