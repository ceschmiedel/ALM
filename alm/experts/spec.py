"""Declarative Expert Agent specification.

An expert is configuration, not code.  Everything the runtime needs — which
model backs it, what it declares it can do, which corpus it may read, how its
output is shaped, and what a verifier should check — is a YAML document inside
a domain pack.  Adding an expert to the federation is adding a file.

The specification deliberately couples four things that are usually scattered:
the *capability declaration* the router matches against, the *retrieval scope*
CMRAG enforces, the *output schema* arbitration reasons over, and the *verifier
rules* that catch domain-invalid answers cheaply.  They belong together because
they all describe the same boundary: what this expert is responsible for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from alm.core.errors import PackError
from alm.graph.models import CapabilityDeclaration
from alm.models.tiers import ModelTier, parse_tier

_TYPE_MAP = {
    "string": "string",
    "str": "string",
    "text": "string",
    "number": "number",
    "float": "number",
    "integer": "integer",
    "int": "integer",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
}


class OutputField(BaseModel):
    """Shorthand for one field of an expert's structured output."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    enum: list[str] = Field(
        default_factory=list,
        description=(
            "Closed vocabulary for this field. Constraining severities, verdicts "
            "and ratings is what makes them comparable across experts — and "
            "therefore arbitrable and aggregable."
        ),
    )

    def json_type(self) -> str:
        return _TYPE_MAP.get(self.type.strip().lower(), "string")


class CapabilitySpec(BaseModel):
    """What the expert declares it can do — the router's matching surface."""

    id: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    examples: list[str] = Field(
        default_factory=list,
        description="Representative requests. The strongest matching signal available.",
    )
    operates_on: list[str] = Field(
        default_factory=list, description="Ontology entity types this capability reads."
    )
    produces: list[str] = Field(default_factory=list)
    required_scopes: list[str] = Field(default_factory=list)
    outputs: list[OutputField] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    estimated_cost: float = 0.0
    estimated_latency: float = 0.0
    tier_hint: str = ""
    instructions: str = Field(
        default="", description="Extra guidance appended to the prompt for this capability."
    )

    @field_validator("id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("capability id must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def _derive_schema(self) -> CapabilitySpec:
        """Build a JSON Schema from the ``outputs`` shorthand when none is given."""
        if self.output_schema or not self.outputs:
            return self
        properties: dict[str, Any] = {}
        required: list[str] = []
        for field in self.outputs:
            entry: dict[str, Any] = {"type": field.json_type()}
            if field.description:
                entry["description"] = field.description
            if field.enum:
                entry["enum"] = list(field.enum)
            if entry["type"] == "array":
                entry["items"] = {"type": "string"}
            properties[field.name] = entry
            if field.required:
                required.append(field.name)
        self.output_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        return self

    def to_declaration(self, expert_id: str, domain: str) -> CapabilityDeclaration:
        return CapabilityDeclaration(
            capability_id=self.id,
            expert_id=expert_id,
            domain=domain,
            description=self.description,
            keywords=list(self.keywords),
            examples=list(self.examples),
            operates_on=list(self.operates_on),
            produces=list(self.produces),
            required_scopes=list(self.required_scopes),
            output_schema=self.output_schema,
            estimated_cost=self.estimated_cost,
            estimated_latency=self.estimated_latency,
            tier_hint=self.tier_hint,
        )


class RetrievalSpec(BaseModel):
    """The expert's CMRAG scope.

    ``domains`` defaults to the expert's own domain — cross-domain reads are
    opt-in *and* still subject to IBAC, so widening the scope here does not
    bypass governance.
    """

    domains: list[str] = Field(default_factory=list)
    top_k: int = 6
    max_context_chars: int = 6000
    use_capability_entities: bool = True
    extra_entities: list[str] = Field(default_factory=list)
    enabled: bool = True


class VerifierRule(BaseModel):
    """A cheap, deterministic domain check applied to this expert's answers.

    Where the rules are clear, a verifier is far better than asking a model to
    judge: it is deterministic, costs nothing, and produces a reason a person
    can act on.
    """

    id: str
    type: str = Field(
        default="required_fields",
        description=(
            "required_fields | requires_citation | numeric_range | numeric_consistency "
            "| forbidden_terms | regex_match | min_confidence"
        ),
    )
    description: str = ""
    fields: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)
    pattern: str = ""
    minimum: float | None = None
    maximum: float | None = None
    tolerance: float = 0.01
    severity: str = Field(default="error", description="error | warning")


class PromptSpec(BaseModel):
    """How the expert is addressed."""

    system: str = ""
    guidelines: list[str] = Field(default_factory=list)
    answer_language: str = ""
    require_citations: bool = True
    temperature: float = 0.0
    max_tokens: int = 900


class ExpertSpec(BaseModel):
    """A complete Expert Agent declaration."""

    id: str
    domain: str
    label: str = ""
    description: str = ""

    model: str = Field(default="", description="Model id in the registry.")
    tier: ModelTier = ModelTier.SLM
    fallback_model: str = ""

    capabilities: list[CapabilitySpec] = Field(default_factory=list)
    authority: dict[str, float] = Field(
        default_factory=dict,
        description="Weight of this expert per domain, used by L5 arbitration.",
    )
    prompt: PromptSpec = Field(default_factory=PromptSpec)
    retrieval: RetrievalSpec = Field(default_factory=RetrievalSpec)
    verifier_rules: list[VerifierRule] = Field(default_factory=list)

    required_scopes: list[str] = Field(default_factory=list)
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "domain")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("expert id and domain must not be empty")
        return value.strip()

    @field_validator("tier", mode="before")
    @classmethod
    def _coerce_tier(cls, value: Any) -> Any:
        return parse_tier(value) if value else ModelTier.SLM

    @model_validator(mode="after")
    def _defaults(self) -> ExpertSpec:
        if not self.label:
            self.label = self.id
        if not self.retrieval.domains:
            self.retrieval.domains = [self.domain]
        if not self.authority:
            self.authority = {self.domain: 1.0}
        elif self.domain not in self.authority:
            self.authority[self.domain] = 1.0
        return self

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpertSpec:
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise PackError(f"invalid expert specification: {exc}") from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExpertSpec:
        file_path = Path(path)
        if not file_path.exists():
            raise PackError(f"expert file not found: {file_path}", path=str(file_path))
        try:
            data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise PackError(f"invalid YAML in {file_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise PackError(f"expert file must contain a mapping: {file_path}")
        return cls.from_dict(data)

    # -- helpers -----------------------------------------------------------

    def declarations(self) -> list[CapabilityDeclaration]:
        return [c.to_declaration(self.id, self.domain) for c in self.capabilities]

    def capability(self, capability_id: str) -> CapabilitySpec | None:
        for cap in self.capabilities:
            if cap.id == capability_id:
                return cap
        return self.capabilities[0] if self.capabilities else None

    def entity_types(self) -> list[str]:
        seen: list[str] = []
        for cap in self.capabilities:
            for entity in cap.operates_on:
                if entity not in seen:
                    seen.append(entity)
        return seen

    def validate_against_ontology(self, entity_names: set[str]) -> list[str]:
        """Report capabilities referencing entity types the ontology does not define.

        Catching this at pack-install time matters: a capability that operates on
        a non-existent entity type will never match well, and the failure would
        otherwise surface as mysteriously poor routing.
        """
        problems: list[str] = []
        for cap in self.capabilities:
            for entity in cap.operates_on:
                if entity not in entity_names:
                    problems.append(
                        f"expert '{self.id}' capability '{cap.id}' operates on "
                        f"'{entity}', which the ontology does not define"
                    )
        return problems
