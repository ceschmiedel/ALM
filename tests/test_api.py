"""REST and WebSocket surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from alm.api.app import create_app, set_runtime


@pytest.fixture
def client(runtime):
    set_runtime(runtime)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    set_runtime(None)


def test_health_reports_inventory_and_warnings(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["experts"] == 3
    assert set(payload["domains"]) == {"legal", "finance", "risk"}
    # The demo pack runs on placeholder backends, and health must say so.
    assert any("placeholder" in w for w in payload["warnings"])


def test_ask_returns_answer_with_provenance(client):
    response = client.post(
        "/v1/ask", json={"intent": "What does clause 7.2 cap the liability at?"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["answer"]
    assert payload["trace"]["events"]
    assert "metrics" in payload


def test_ask_can_omit_the_trace(client):
    response = client.post(
        "/v1/ask", json={"intent": "What are the payment terms?", "include_trace": False}
    )
    assert response.json()["trace"] is None


def test_explain_exposes_routing_without_executing(client):
    response = client.post("/v1/explain", json={"intent": "Is clause 7.2 enforceable?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert "would_escalate" in payload


def test_session_is_retrievable_after_a_run(client):
    session_id = client.post("/v1/ask", json={"intent": "What are the payment terms?"}).json()[
        "session_id"
    ]
    response = client.get(f"/v1/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["session_id"] == session_id


def test_unknown_session_is_404(client):
    assert client.get("/v1/sessions/ses_nope").status_code == 404


def test_graph_endpoints(client):
    nodes = client.get("/v1/graph/nodes", params={"kind": "capability"}).json()
    assert nodes
    assert all(node["kind"] == "capability" for node in nodes)

    stats = client.get("/v1/graph/stats").json()
    assert stats["nodes"] > 0

    search = client.post(
        "/v1/graph/search", json={"query": "liability cap", "kinds": ["capability"]}
    ).json()
    assert search


def test_expert_endpoints(client):
    experts = client.get("/v1/experts").json()
    assert len(experts) == 3

    detail = client.get("/v1/experts/contracts-expert").json()
    assert detail["domain"] == "legal"

    assert client.get("/v1/experts/ghost").status_code == 404


def test_promote_check_endpoint(client):
    response = client.post(
        "/v1/experts/contracts-expert/promote-check",
        json={"expert_id": "contracts-expert", "sovereignty_required": True},
    )
    assert response.status_code == 200
    assert response.json()["recommend_dedicated"] is True


def test_model_registry_endpoints(client):
    listing = client.get("/v1/models").json()
    assert listing["models"]
    assert "serving" in listing

    created = client.post(
        "/v1/models",
        json={
            "model_id": "api-added",
            "tier": "slm",
            "backend": "vllm",
            "base_model": "llama-3.2-3b",
            "adapter": "api-v1",
            "api_key": "secret",
        },
    )
    assert created.status_code == 200
    assert "api_key" not in created.json(), "secrets must not be echoed back"

    assert client.delete("/v1/models/api-added").status_code == 200
    assert client.delete("/v1/models/api-added").status_code == 404


def test_cmrag_search_endpoint(client):
    response = client.post(
        "/v1/cmrag/search", json={"query": "liability cap", "domains": ["legal"]}
    )
    assert response.status_code == 200
    assert response.json()["chunks"]


def test_cmrag_search_applies_an_experts_governance(client):
    response = client.post(
        "/v1/cmrag/search",
        json={"query": "budget allocation", "expert_id": "contracts-expert"},
    )
    assert response.status_code == 200
    payload = response.json()
    # The legal expert is denied the finance corpus by the demo pack's policy.
    assert all(chunk["domain"] != "finance" for chunk in payload["chunks"])


def test_audit_endpoints(client):
    client.post("/v1/ask", json={"intent": "What are the payment terms?"})
    assert client.get("/v1/audit").status_code == 200
    assert "total" in client.get("/v1/audit/summary").json()

    export = client.get("/v1/audit/export")
    assert export.status_code == 200


def test_policies_endpoint(client):
    policies = client.get("/v1/policies").json()
    assert any(p["id"] == "restricted-needs-claim" for p in policies)


def test_pack_validate_endpoint(client, pack_path):
    response = client.post("/v1/packs/validate", json={"path": str(pack_path)})
    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_alm_errors_become_structured_400s(client):
    response = client.post("/v1/packs/validate", json={"path": "/nonexistent/pack"})
    assert response.status_code == 400
    assert response.json()["code"] == "pack_error"


def test_websocket_streams_events_then_the_result(client):
    with client.websocket_connect("/v1/stream") as socket:
        socket.send_json({"intent": "What does clause 7.2 cap the liability at?"})

        events = []
        while True:
            message = socket.receive_json()
            if message["type"] == "result":
                assert message["result"]["answer"]
                break
            if message["type"] == "error":  # pragma: no cover
                pytest.fail(message["error"])
            events.append(message)

        assert events, "the run must stream trace events before the result"
        assert any(e["event"]["layer"] == "L2" for e in events)


def test_websocket_rejects_a_missing_intent(client):
    with client.websocket_connect("/v1/stream") as socket:
        socket.send_json({})
        assert socket.receive_json()["type"] == "error"


def test_bearer_token_is_enforced_when_configured(runtime, monkeypatch):
    monkeypatch.setenv("ALM_API_TOKEN", "s3cret")
    from alm.config import reset_settings_cache

    reset_settings_cache()
    set_runtime(runtime)
    try:
        with TestClient(create_app()) as guarded:
            assert guarded.get("/v1/health").status_code == 200, "health stays open"
            assert guarded.get("/v1/experts").status_code == 401
            authorised = guarded.get(
                "/v1/experts", headers={"Authorization": "Bearer s3cret"}
            )
            assert authorised.status_code == 200
    finally:
        set_runtime(None)
        reset_settings_cache()
