"""GV · IBAC policy matching, per-agent context filtering and the audit log."""

from __future__ import annotations

from alm.cmrag.store import RetrievedChunk
from alm.governance.audit import AuditLog
from alm.governance.ibac import AccessRequest, IBACEngine
from alm.governance.policies import (
    Effect,
    EvaluationPoint,
    Policy,
    PolicySet,
    sensitivity_rank,
)


def _chunk(**overrides) -> RetrievedChunk:
    defaults = dict(
        chunk_id="chk1",
        document_id="doc1",
        title="doc",
        text="text",
        domain="legal",
        entities=["Clause"],
        metadata={},
        sensitivity="internal",
    )
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


def _engine(*policies: Policy) -> IBACEngine:
    return IBACEngine(PolicySet(policies=list(policies)))


# -- selectors ---------------------------------------------------------------


def test_sensitivity_selector_covers_stricter_levels():
    policy = Policy(id="p", effect=Effect.DENY, resources=["sensitivity:confidential"])
    assert policy.matches_resource(sensitivity="confidential")
    assert policy.matches_resource(sensitivity="restricted"), (
        "guarding a level must also guard everything stricter"
    )
    assert not policy.matches_resource(sensitivity="internal")


def test_sensitivity_rank_ordering():
    assert sensitivity_rank("public") < sensitivity_rank("restricted")
    assert sensitivity_rank("nonsense") == sensitivity_rank("internal")


def test_domain_and_entity_selectors():
    policy = Policy(id="p", resources=["domain:finance", "entity:Salary"])
    assert policy.matches_resource(domain="finance")
    assert policy.matches_resource(entities=["Salary"])
    assert not policy.matches_resource(domain="legal", entities=["Clause"])


def test_subject_selectors():
    policy = Policy(id="p", subjects=["expert:finance-expert", "domain:risk"])
    assert policy.matches_subject(expert_id="finance-expert", domain="finance", principal="")
    assert policy.matches_subject(expert_id="other", domain="risk", principal="")
    assert not policy.matches_subject(expert_id="legal", domain="legal", principal="")


# -- evaluation --------------------------------------------------------------


def test_deny_wins_over_allow():
    engine = _engine(
        Policy(id="allow-all", effect=Effect.ALLOW, priority=999),
        Policy(
            id="deny-finance",
            effect=Effect.DENY,
            priority=1,
            subjects=["contracts-expert"],
            resources=["domain:finance"],
        ),
    )
    decision = engine.may_read_context(
        expert_id="contracts-expert",
        domain="legal",
        chunk=_chunk(domain="finance"),
    )
    assert not decision.allowed
    assert decision.matched_policies == ["deny-finance"]


def test_higher_priority_wins_within_the_same_effect():
    engine = _engine(
        Policy(id="broad", effect=Effect.DENY, priority=10, resources=["*"]),
        Policy(id="narrow", effect=Effect.DENY, priority=500, resources=["domain:legal"]),
    )
    decision = engine.may_read_context(
        expert_id="e", domain="legal", chunk=_chunk(domain="legal")
    )
    assert decision.matched_policies == ["narrow"]


def test_unless_claims_exempts_the_holder():
    engine = _engine(
        Policy(
            id="restricted",
            effect=Effect.DENY,
            resources=["sensitivity:restricted"],
            unless_claims=["risk.incident_review"],
        )
    )
    chunk = _chunk(sensitivity="restricted")
    assert not engine.may_read_context(expert_id="e", domain="risk", chunk=chunk).allowed
    assert engine.may_read_context(
        expert_id="e", domain="risk", chunk=chunk, claims=["risk.incident_review"]
    ).allowed


def test_when_claims_requires_all_of_them():
    engine = _engine(
        Policy(id="p", effect=Effect.DENY, when_claims=["a", "b"], resources=["*"])
    )
    assert engine.evaluate(
        AccessRequest(evaluation_point=EvaluationPoint.CONTEXT_READ, claims=["a"])
    ).allowed
    assert not engine.evaluate(
        AccessRequest(evaluation_point=EvaluationPoint.CONTEXT_READ, claims=["a", "b"])
    ).allowed


def test_evaluation_point_scoping():
    engine = _engine(
        Policy(
            id="emit-only",
            effect=Effect.DENY,
            evaluation_points=[EvaluationPoint.ARTIFACT_EMISSION],
            resources=["*"],
        )
    )
    assert engine.may_read_context(expert_id="e", domain="d", chunk=_chunk()).allowed
    assert not engine.may_emit(domains=["d"], entities=[]).allowed


def test_default_decision_applies_when_nothing_matches():
    permissive = IBACEngine(PolicySet(), default_effect=Effect.ALLOW)
    restrictive = IBACEngine(PolicySet(), default_effect=Effect.DENY)
    request = AccessRequest(evaluation_point=EvaluationPoint.CONTEXT_READ)
    assert permissive.evaluate(request).allowed
    assert not restrictive.evaluate(request).allowed


def test_disabled_engine_allows_everything():
    engine = IBACEngine(
        PolicySet(policies=[Policy(id="deny", effect=Effect.DENY, resources=["*"])]),
        enabled=False,
    )
    assert engine.evaluate(
        AccessRequest(evaluation_point=EvaluationPoint.CONTEXT_READ)
    ).allowed


def test_disabled_tenant_scoping():
    engine = _engine(
        Policy(id="other-tenant", effect=Effect.DENY, tenants=["acme"], resources=["*"])
    )
    decision = engine.evaluate(
        AccessRequest(evaluation_point=EvaluationPoint.CONTEXT_READ, tenant_id="default")
    )
    assert decision.allowed, "a policy scoped to another tenant must not apply"


# -- per-agent filtering -----------------------------------------------------


def test_context_filter_is_per_agent():
    engine = _engine(
        Policy(
            id="legal-no-finance",
            effect=Effect.DENY,
            subjects=["contracts-expert"],
            resources=["domain:finance"],
        )
    )
    finance_chunk = _chunk(domain="finance")

    legal_filter = engine.context_filter(expert_id="contracts-expert", domain="legal")
    finance_filter = engine.context_filter(expert_id="finance-expert", domain="finance")

    assert legal_filter(finance_chunk) is False
    assert finance_filter(finance_chunk) is True


def test_denials_are_written_to_the_audit_log():
    engine = _engine(
        Policy(id="deny-all", effect=Effect.DENY, resources=["*"], evaluation_points=[
            EvaluationPoint.CONTEXT_READ
        ])
    )
    filter_fn = engine.context_filter(
        expert_id="e", domain="legal", session_id="ses-audit"
    )
    filter_fn(_chunk())

    records = AuditLog().query(session_id="ses-audit", decision="deny")
    assert records
    assert records[0].expert_id == "e"


# -- audit log ---------------------------------------------------------------


def test_audit_records_allow_and_deny():
    log = AuditLog()
    log.record(evaluation_point="context_read", decision="allow", session_id="s1", expert_id="e")
    log.record(evaluation_point="context_read", decision="deny", session_id="s1", expert_id="e")

    summary = log.summary(hours=1)
    assert summary["total"] == 2
    assert summary["allowed"] == 1
    assert summary["denied"] == 1
    assert summary["deny_rate"] == 0.5


def test_audit_for_session_is_chronological():
    log = AuditLog()
    for index in range(3):
        log.record(
            evaluation_point="context_read",
            decision="allow",
            session_id="s2",
            resource=f"chunk-{index}",
        )
    records = log.for_session("s2")
    assert [r.resource for r in records] == ["chunk-0", "chunk-1", "chunk-2"]


def test_audit_export_is_jsonl():
    import json

    log = AuditLog()
    log.record(evaluation_point="intent_admission", decision="allow", session_id="s3")
    lines = list(log.export_jsonl(session_id="s3"))
    assert lines
    assert json.loads(lines[0])["decision"] == "allow"


def test_audit_never_raises_on_failure(monkeypatch):
    log = AuditLog()

    def explode(*_args, **_kwargs):
        raise RuntimeError("database gone")

    monkeypatch.setattr("alm.governance.audit.session_scope", explode)
    # Governance must not be able to break a run.
    log.record(evaluation_point="context_read", decision="allow")


def test_policy_set_loads_from_dict():
    policies = PolicySet.from_dict(
        {
            "policies": [
                {
                    "id": "p1",
                    "effect": "deny",
                    "subjects": ["a"],
                    "resources": ["domain:x"],
                }
            ]
        }
    )
    assert len(policies) == 1
    assert policies.by_id("p1").effect is Effect.DENY
