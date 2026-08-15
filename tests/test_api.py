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


def test_list_packs_endpoint(client, pack_path, monkeypatch):
    monkeypatch.setenv("ALM_PACKS_DIR", str(pack_path.parent))
    from alm.config import reset_settings_cache

    reset_settings_cache()
    try:
        response = client.get("/v1/packs")
        assert response.status_code == 200
        names = {p["name"] for p in response.json()}
        assert "demo-enterprise" in names
    finally:
        reset_settings_cache()


def test_eval_datasets_endpoint(client, pack_path, monkeypatch):
    monkeypatch.setenv("ALM_PACKS_DIR", str(pack_path.parent))
    from alm.config import reset_settings_cache

    reset_settings_cache()
    try:
        response = client.get("/v1/eval/datasets")
        assert response.status_code == 200
        demo = next(d for d in response.json() if d["pack"] == "demo-enterprise")
        assert demo["eval_files"]
        assert set(demo["domains"]) == {"legal", "finance", "risk"}
    finally:
        reset_settings_cache()


def test_eval_run_and_detail_endpoints(client, tmp_path):
    import json

    dataset_path = tmp_path / "mini.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "What are the payment terms?",
                "expected": "net 30",
                "keywords": ["net 30"],
                "min_keyword_ratio": 1.0,
                "domain": "legal",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    response = client.post(
        "/v1/eval/run",
        json={"dataset": str(dataset_path), "compare": False, "repeats": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["cases"] == 1
    assert payload["federation"]["cases"] == 1
    run_id = payload["run_id"]

    runs = client.get("/v1/eval/runs").json()
    assert any(r["run_id"] == run_id for r in runs)

    detail = client.get(f"/v1/eval/runs/{run_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["case_count"] == 1
    assert body["cases"][0]["case_id"]

    assert client.get("/v1/eval/runs/does-not-exist").status_code == 404


def test_eval_run_requires_dataset_or_pack(client):
    response = client.post("/v1/eval/run", json={"compare": False})
    assert response.status_code == 400


def test_ollama_tags_degrades_gracefully_when_unreachable(client, monkeypatch):
    import httpx

    async def _boom(self, url, *args, **kwargs):  # noqa: ANN001
        raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", _boom)

    response = client.get("/v1/ollama/tags")
    assert response.status_code == 200
    payload = response.json()
    assert payload["reachable"] is False
    assert payload["models"] == []


def test_ollama_tags_lists_local_models(client, monkeypatch):
    import httpx

    async def _fake_get(self, url, *args, **kwargs):  # noqa: ANN001
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "llama3.2:3b",
                        "size": 2_000_000_000,
                        "modified_at": "2026-01-01T00:00:00Z",
                        "details": {
                            "family": "llama",
                            "parameter_size": "3.2B",
                            "quantization_level": "Q4_K_M",
                        },
                    }
                ]
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

    response = client.get("/v1/ollama/tags")
    assert response.status_code == 200
    payload = response.json()
    assert payload["reachable"] is True
    assert payload["models"][0]["name"] == "llama3.2:3b"
    assert payload["models"][0]["parameter_size"] == "3.2B"


def test_ollama_running_endpoint(client, monkeypatch):
    import httpx

    async def _fake_get(self, url, *args, **kwargs):  # noqa: ANN001
        request = httpx.Request("GET", url)
        return httpx.Response(200, json={"models": []}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

    response = client.get("/v1/ollama/running")
    assert response.status_code == 200
    assert response.json() == {"reachable": True, "base_url": response.json()["base_url"], "models": []}


def _agent_payload(**overrides):
    payload = {
        "id": "vendas-expert",
        "domain": "vendas",
        "label": "Especialista de Vendas",
        "description": "Analisa desempenho comercial",
        "tier": "slm",
        "capabilities": [
            {
                "id": "analisar_vendas",
                "description": "Analisa receita e margem por produto",
                "keywords": ["receita", "margem"],
                "examples": ["Qual produto teve maior receita?"],
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_create_agent_registers_it_and_its_domain(client):
    response = client.post("/v1/experts", json=_agent_payload(model="demo-expert"))
    assert response.status_code == 200
    assert response.json()["id"] == "vendas-expert"

    listing = {e["id"] for e in client.get("/v1/experts").json()}
    assert "vendas-expert" in listing

    # The domain did not exist in the demo pack; creating the agent must have
    # registered it, or the router would never be able to reach the agent.
    assert "vendas" in client.get("/v1/graph/stats").json()["domains"]

    capabilities = client.get("/v1/graph/capabilities?domain=vendas").json()
    assert any(c["capability_id"] == "analisar_vendas" for c in capabilities)


def test_create_agent_rejects_duplicates_and_unknown_models(client):
    assert client.post("/v1/experts", json=_agent_payload()).status_code == 200
    assert client.post("/v1/experts", json=_agent_payload()).status_code == 409

    unknown = client.post(
        "/v1/experts", json=_agent_payload(id="other", model="does-not-exist")
    )
    assert unknown.status_code == 400


def test_create_agent_requires_a_capability(client):
    response = client.post("/v1/experts", json=_agent_payload(capabilities=[]))
    assert response.status_code == 400
    assert "capability" in response.json()["detail"]


def test_update_agent_reassigns_the_model(client):
    client.post("/v1/experts", json=_agent_payload(model="demo-expert"))

    response = client.patch(
        "/v1/experts/vendas-expert",
        json={"model": "demo-orchestrator", "label": "Vendas BR"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "demo-orchestrator"
    assert body["label"] == "Vendas BR"

    # The change must survive a registry rebuild — it lives on the graph node.
    inventory = {e["id"]: e for e in client.get("/v1/experts").json()}
    assert inventory["vendas-expert"]["model"] == "demo-orchestrator"

    assert client.patch("/v1/experts/nope", json={"label": "x"}).status_code == 404


def test_delete_agent_removes_its_capabilities_from_routing(client):
    client.post("/v1/experts", json=_agent_payload())
    assert client.delete("/v1/experts/vendas-expert").status_code == 200
    assert client.delete("/v1/experts/vendas-expert").status_code == 404

    # A capability left behind would keep the router matching a missing agent.
    capabilities = client.get("/v1/graph/capabilities").json()
    assert not any(c["capability_id"] == "analisar_vendas" for c in capabilities)


CSV_UPLOAD = (
    b"produto,unidade,receita_2025\n"
    b"Camiseta,Vestuario,1439064\n"
    b"Panela,Utilidades,912346\n"
)


def test_upload_csv_indexes_it_and_makes_it_retrievable(client):
    response = client.post(
        "/v1/cmrag/upload",
        files={"file": ("vendas.csv", CSV_UPLOAD, "text/csv")},
        data={"domain": "vendas", "sensitivity": "internal"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["documents"] == 1
    assert payload["chunks"] > 0
    assert payload["sheets"][0]["rows"] == 2
    assert payload["sheets"][0]["columns"] == ["produto", "unidade", "receita_2025"]

    found = client.post(
        "/v1/cmrag/search", json={"query": "receita da Camiseta", "domains": ["vendas"]}
    ).json()
    assert found["chunks"]
    assert any("Camiseta" in c["text"] for c in found["chunks"])


def test_reuploading_the_same_file_is_a_noop(client):
    files = {"file": ("vendas.csv", CSV_UPLOAD, "text/csv")}
    client.post("/v1/cmrag/upload", files=files, data={"domain": "vendas"})
    second = client.post(
        "/v1/cmrag/upload",
        files={"file": ("vendas.csv", CSV_UPLOAD, "text/csv")},
        data={"domain": "vendas"},
    )
    assert second.json()["sheets"][0]["skipped"] is True
    assert len(client.get("/v1/cmrag/documents?domain=vendas").json()) == 1


def test_upload_rejects_empty_and_unsupported_files(client):
    empty = client.post(
        "/v1/cmrag/upload",
        files={"file": ("x.csv", b"", "text/csv")},
        data={"domain": "vendas"},
    )
    assert empty.status_code == 400

    unsupported = client.post(
        "/v1/cmrag/upload",
        files={"file": ("notes.pdf", b"%PDF-1.4 junk", "application/pdf")},
        data={"domain": "vendas"},
    )
    assert unsupported.status_code == 400
    assert "unsupported" in unsupported.json()["detail"]


def test_documents_can_be_listed_and_deleted(client):
    client.post(
        "/v1/cmrag/upload",
        files={"file": ("vendas.csv", CSV_UPLOAD, "text/csv")},
        data={"domain": "vendas"},
    )
    documents = client.get("/v1/cmrag/documents?domain=vendas").json()
    assert documents
    document_id = documents[0]["document_id"]

    removed = client.delete(f"/v1/cmrag/documents/{document_id}")
    assert removed.status_code == 200
    assert removed.json()["chunks_removed"] > 0
    assert client.delete(f"/v1/cmrag/documents/{document_id}").status_code == 404


def test_model_probe_reports_success_and_failure(client):
    ok = client.post("/v1/models/probe", json={"model_id": "demo-orchestrator"})
    assert ok.status_code == 200
    assert ok.json()["ok"] is True

    # A hosted backend with no key must fail loudly here rather than at the
    # first escalation.
    broken = client.post(
        "/v1/models/probe",
        json={"backend": "openai", "model_name": "gpt-4o", "api_key": ""},
    )
    assert broken.status_code == 200
    assert broken.json()["ok"] is False
    assert broken.json()["error"]

    assert client.post("/v1/models/probe", json={}).status_code == 400
    assert client.post("/v1/models/probe", json={"model_id": "nope"}).status_code == 404


def test_model_backends_endpoint_lists_the_vocabulary(client):
    payload = client.get("/v1/models/backends").json()
    assert "ollama" in payload["backends"]
    assert {t["id"] for t in payload["tiers"]} == {
        "micro_slm",
        "slm",
        "small",
        "orchestrator",
    }
    assert "openai" in payload["hosted_backends"]


def test_dashboard_is_served_at_root_and_ui(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in {307, 308}
    assert response.headers["location"] == "/ui/"

    ui = client.get("/ui/")
    assert ui.status_code == 200
    assert "ALM" in ui.text


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
