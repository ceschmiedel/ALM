"""FastAPI application — REST and WebSocket access to the federation.

The API mirrors what the CLI exposes, with one addition that only makes sense
over a socket: ``/v1/stream`` pushes trace events **as the run happens**, so a
caller watches routing, each expert answering, and arbitration resolving in real
time rather than receiving a finished block of text.

Authentication is a single bearer token (``ALM_API_TOKEN``) — deliberately
minimal. This is a component that belongs behind your own gateway, and pretending
otherwise by shipping half an identity system would be worse than being explicit.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from alm import __version__
from alm.api.schemas import (
    AskRequest,
    AskResponse,
    EvalRequest,
    ExplainRequest,
    GraphSearchRequest,
    HealthResponse,
    ModelCreateRequest,
    PackInstallRequest,
    PromotionCheckRequest,
    SearchRequest,
)
from alm.config import get_settings
from alm.core.errors import ALMError
from alm.evaluation import (
    EvaluationHarness,
    list_runs,
    load_pack_datasets,
    render_comparison,
)
from alm.evaluation.metrics import compare
from alm.experts.pack import discover_packs, load_pack
from alm.federation.runtime import FederationRuntime
from alm.governance.audit import AuditLog
from alm.mlops.drift import DriftDetector, assess_promotion
from alm.mlops.versions import VersionRegistry
from alm.models.registry import ModelRegistry
from alm.models.spec import ModelSpec
from alm.models.tiers import parse_tier
from alm.persistence.database import init_db, session_scope
from alm.persistence.models import SessionRow
from alm.protocol.trace import TraceEvent

logger = logging.getLogger(__name__)

_runtime: FederationRuntime | None = None


def get_runtime() -> FederationRuntime:
    """The process-wide runtime, built on first use."""
    global _runtime
    if _runtime is None:
        _runtime = FederationRuntime.from_database()
    return _runtime


def set_runtime(runtime: FederationRuntime | None) -> None:
    """Inject a runtime — used by tests and by embedders of this app."""
    global _runtime
    _runtime = runtime


_security = HTTPBearer(auto_error=False)


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> None:
    """Enforce the bearer token when one is configured."""
    expected = get_settings().api_token
    if not expected:
        return
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    global _runtime
    if _runtime is not None:
        await _runtime.close()
        _runtime = None


def create_app() -> FastAPI:
    """Build the ASGI application."""
    app = FastAPI(
        title="ALM — Agent Language Model",
        description=(
            "A governed federation of domain models orchestrated by a Context Graph."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    @app.exception_handler(ALMError)
    async def _alm_error_handler(_request, exc: ALMError):
        # ALM errors carry structured details; surfacing them beats a bare 500.
        return JSONResponse(status_code=400, content=exc.to_dict())

    _register_routes(app)

    @app.get("/", include_in_schema=False)
    async def _root() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="dashboard")

    return app


def _register_routes(app: FastAPI) -> None:  # noqa: C901 - a flat router reads better here
    auth = [Depends(require_auth)]

    # -- health and inventory ---------------------------------------------

    @app.get("/v1/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        runtime = get_runtime()
        report = runtime.health()

        warnings: list[str] = []
        if not report["orchestrator_configured"]:
            warnings.append(
                "no orchestrator model registered: planner-driven decomposition, the "
                "LLM judge and the fallback path are all unavailable"
            )
        if not report["semantic_routing"]:
            warnings.append(
                "no embedder configured: routing and retrieval are lexical only"
            )
        if report["uncalibrated_experts"]:
            warnings.append(
                "uncalibrated experts (confidence arbitration is discounted for them): "
                + ", ".join(report["uncalibrated_experts"])
            )
        if report["experts_on_placeholder_backends"]:
            warnings.append(
                "experts bound to placeholder backends, not language models: "
                + ", ".join(report["experts_on_placeholder_backends"])
            )

        return HealthResponse(
            status="ok",
            version=__version__,
            tenant=report["tenant"],
            experts=report["experts"],
            domains=report["graph"]["domains"],
            models=report["models"],
            corpus=report["corpus"],
            semantic_routing=report["semantic_routing"],
            orchestrator_configured=report["orchestrator_configured"],
            fallback_available=report["fallback_available"],
            governance_enabled=report["governance_enabled"],
            warnings=warnings,
        )

    @app.get("/v1/stats", tags=["system"], dependencies=auth)
    async def stats() -> dict[str, Any]:
        return get_runtime().health()

    # -- the federation ----------------------------------------------------

    @app.post("/v1/ask", response_model=AskResponse, tags=["federation"], dependencies=auth)
    async def ask(request: AskRequest) -> AskResponse:
        runtime = get_runtime()
        result = await runtime.run(
            request.intent,
            context=request.context,
            principal=request.principal,
            claims=request.claims,
            requested_outputs=request.requested_outputs,
            session_id=request.session_id,
        )
        return AskResponse(
            session_id=result.session_id,
            status=result.status,
            answer=result.answer,
            confidence=result.confidence,
            experts=result.experts,
            used_fallback=result.metrics.used_fallback,
            citations=[c.model_dump(mode="json") for c in result.citations],
            audit_trail=result.audit_trail(),
            metrics=result.metrics.model_dump(mode="json"),
            plan=result.plan.model_dump(mode="json") if result.plan else None,
            trace=result.trace.to_dict() if request.include_trace else None,
            error=result.error,
        )

    @app.post("/v1/explain", tags=["federation"], dependencies=auth)
    async def explain(request: ExplainRequest) -> dict[str, Any]:
        """Show routing signals without executing — the router is auditable too."""
        return get_runtime().router.explain(request.intent, top_k=request.top_k)

    @app.get("/v1/sessions/{session_id}", tags=["federation"], dependencies=auth)
    async def get_session(session_id: str) -> dict[str, Any]:
        with session_scope() as session:
            row = session.get(SessionRow, session_id)
            if row is None:
                raise HTTPException(status_code=404, detail="session not found")
            return {
                "session_id": row.session_id,
                "status": row.status,
                "intent_text": row.intent_text,
                "answer": row.answer,
                "strategy": row.strategy,
                "router_confidence": row.router_confidence,
                "used_fallback": row.used_fallback,
                "plan": row.plan,
                "answers": row.answers,
                "arbitration": row.arbitration,
                "trace": row.trace,
                "metrics": row.metrics,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            }

    @app.get("/v1/sessions", tags=["federation"], dependencies=auth)
    async def list_sessions(limit: int = Query(default=25, le=200)) -> list[dict[str, Any]]:
        from sqlalchemy import select

        runtime = get_runtime()
        with session_scope() as session:
            rows = session.execute(
                select(SessionRow)
                .where(SessionRow.tenant_id == runtime.tenant_id)
                .order_by(SessionRow.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "session_id": r.session_id,
                    "intent_text": r.intent_text[:200],
                    "status": r.status,
                    "strategy": r.strategy,
                    "used_fallback": r.used_fallback,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                }
                for r in rows
            ]

    # -- live trace --------------------------------------------------------

    @app.websocket("/v1/stream")
    async def stream(socket: WebSocket) -> None:
        """Run an intent and stream trace events as they happen."""
        await socket.accept()
        settings = get_settings()
        try:
            payload = await socket.receive_json()
        except Exception:
            await socket.close(code=1003)
            return

        if settings.api_token and payload.get("token") != settings.api_token:
            await socket.send_json({"type": "error", "error": "unauthorised"})
            await socket.close(code=1008)
            return

        intent = str(payload.get("intent", "")).strip()
        if not intent:
            await socket.send_json({"type": "error", "error": "intent is required"})
            await socket.close(code=1003)
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def on_event(event: TraceEvent) -> None:
            # The trace emits from the running coroutine's thread; hop back onto
            # the loop so the socket write is ordered.
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "event", "event": event.model_dump(mode="json")},
            )

        async def pump() -> None:
            while True:
                message = await queue.get()
                await socket.send_json(message)

        pump_task = asyncio.create_task(pump())
        try:
            result = await get_runtime().run(
                intent,
                context=payload.get("context") or {},
                principal=str(payload.get("principal", "")),
                claims=list(payload.get("claims") or []),
                on_event=on_event,
            )
            await asyncio.sleep(0)  # let queued events drain
            while not queue.empty():
                await socket.send_json(await queue.get())
            await socket.send_json(
                {
                    "type": "result",
                    "result": {
                        "session_id": result.session_id,
                        "status": result.status,
                        "answer": result.answer,
                        "confidence": result.confidence,
                        "experts": result.experts,
                        "audit_trail": result.audit_trail(),
                        "metrics": result.metrics.model_dump(mode="json"),
                    },
                }
            )
        except WebSocketDisconnect:
            return
        except Exception as exc:
            logger.exception("Streaming run failed")
            with_suppress = {"type": "error", "error": str(exc)}
            try:
                await socket.send_json(with_suppress)
            except Exception:
                pass
        finally:
            pump_task.cancel()
            try:
                await socket.close()
            except Exception:
                pass

    # -- context graph -----------------------------------------------------

    @app.get("/v1/graph/nodes", tags=["graph"], dependencies=auth)
    async def graph_nodes(
        kind: str = Query(default=""), domain: str = Query(default="")
    ) -> list[dict[str, Any]]:
        graph = get_runtime().graph
        nodes = graph.find_nodes(kind or None, domain=domain or None)
        return [
            {
                "node_id": n.node_id,
                "kind": str(n.kind),
                "key": n.key,
                "label": n.label,
                "domain": n.domain,
                "description": n.description,
                # Embeddings are large and useless over the wire.
                "attributes": {k: v for k, v in n.attributes.items() if k != "spec"},
            }
            for n in nodes
        ]

    @app.get("/v1/graph/stats", tags=["graph"], dependencies=auth)
    async def graph_stats() -> dict[str, Any]:
        return get_runtime().graph.stats().model_dump()

    @app.post("/v1/graph/search", tags=["graph"], dependencies=auth)
    async def graph_search(request: GraphSearchRequest) -> list[dict[str, Any]]:
        graph = get_runtime().graph
        results = graph.semantic_search(
            request.query,
            kinds=request.kinds or None,
            domain=request.domain or None,
            top_k=request.top_k,
        )
        return [
            {
                "key": node.key,
                "kind": str(node.kind),
                "domain": node.domain,
                "label": node.label,
                "score": round(score, 6),
            }
            for node, score in results
        ]

    @app.get("/v1/graph/capabilities", tags=["graph"], dependencies=auth)
    async def capabilities(domain: str = Query(default="")) -> list[dict[str, Any]]:
        return [
            c.model_dump(mode="json")
            for c in get_runtime().graph.capabilities(domain or None)
        ]

    # -- experts -----------------------------------------------------------

    @app.get("/v1/experts", tags=["experts"], dependencies=auth)
    async def experts() -> list[dict[str, Any]]:
        return get_runtime().experts.inventory()

    @app.get("/v1/experts/{expert_id}", tags=["experts"], dependencies=auth)
    async def expert_detail(expert_id: str) -> dict[str, Any]:
        agent = get_runtime().experts.get(expert_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="expert not found")
        return agent.spec.model_dump(mode="json")

    @app.post("/v1/experts/{expert_id}/promote-check", tags=["experts"], dependencies=auth)
    async def promote_check(
        expert_id: str, request: PromotionCheckRequest
    ) -> dict[str, Any]:
        runtime = get_runtime()
        if runtime.experts.get(expert_id) is None:
            raise HTTPException(status_code=404, detail="expert not found")
        assessment = assess_promotion(
            expert_id=expert_id,
            tenant_id=runtime.tenant_id,
            sovereignty_required=request.sovereignty_required,
            monthly_calls=request.monthly_calls,
            volume_threshold=request.volume_threshold,
            latency_requirement_ms=request.latency_requirement_ms,
            observed_p95_ms=request.observed_p95_ms,
            rag_accuracy=request.rag_accuracy,
            accuracy_target=request.accuracy_target,
        )
        return assessment.model_dump(mode="json")

    @app.get("/v1/experts/{expert_id}/drift", tags=["experts"], dependencies=auth)
    async def expert_drift(expert_id: str) -> dict[str, Any]:
        runtime = get_runtime()
        return DriftDetector(runtime.tenant_id).detect(expert_id).model_dump(mode="json")

    # -- models ------------------------------------------------------------

    @app.get("/v1/models", tags=["models"], dependencies=auth)
    async def models() -> dict[str, Any]:
        registry = get_runtime().models
        return {
            "models": [m.model_dump(mode="json", exclude={"api_key"}) for m in registry.list()],
            "serving": registry.serving_summary(),
        }

    @app.post("/v1/models", tags=["models"], dependencies=auth)
    async def create_model(request: ModelCreateRequest) -> dict[str, Any]:
        registry: ModelRegistry = get_runtime().models
        spec = ModelSpec(
            model_id=request.model_id,
            tier=parse_tier(request.tier),
            backend=request.backend,
            model_name=request.model_name,
            base_model=request.base_model,
            adapter=request.adapter,
            adapter_uri=request.adapter_uri,
            endpoint=request.endpoint,
            api_key=request.api_key,
            params=request.params,
            context_window=request.context_window,
            cost_per_1k_input=request.cost_per_1k_input,
            cost_per_1k_output=request.cost_per_1k_output,
            description=request.description,
        )
        registry.register(spec)
        return spec.model_dump(mode="json", exclude={"api_key"})

    @app.delete("/v1/models/{model_id}", tags=["models"], dependencies=auth)
    async def delete_model(model_id: str) -> dict[str, Any]:
        if not get_runtime().models.remove(model_id):
            raise HTTPException(status_code=404, detail="model not found")
        return {"deleted": model_id}

    @app.get("/v1/models/{model_id}/versions", tags=["models"], dependencies=auth)
    async def model_versions(model_id: str) -> list[dict[str, Any]]:
        return VersionRegistry(get_runtime().tenant_id).history(model_id)

    @app.get("/v1/models/health", tags=["models"], dependencies=auth)
    async def model_health() -> dict[str, bool]:
        return await get_runtime().models.health()

    # -- retrieval ---------------------------------------------------------

    @app.post("/v1/cmrag/search", tags=["cmrag"], dependencies=auth)
    async def cmrag_search(request: SearchRequest) -> dict[str, Any]:
        runtime = get_runtime()
        retriever = runtime.retriever
        if request.expert_id:
            agent = runtime.experts.get(request.expert_id)
            if agent is None:
                raise HTTPException(status_code=404, detail="expert not found")
            retriever = retriever.with_access_filter(
                runtime.ibac.context_filter(
                    expert_id=agent.id, domain=agent.domain, session_id="api-search"
                )
            )
        result = retriever.retrieve(
            request.query,
            domains=request.domains or None,
            entities=request.entities or None,
            top_k=request.top_k,
        )
        return result.to_dict()

    @app.get("/v1/cmrag/stats", tags=["cmrag"], dependencies=auth)
    async def cmrag_stats() -> dict[str, Any]:
        return get_runtime().retriever.stats()

    # -- packs -------------------------------------------------------------

    @app.post("/v1/packs/install", tags=["packs"], dependencies=auth)
    async def install_pack(request: PackInstallRequest) -> dict[str, Any]:
        runtime = get_runtime()
        pack = load_pack(request.path)
        report = pack.install(
            graph=runtime.graph,
            registry=runtime.models,
            store=runtime.chunk_store,
            embedder=runtime.embedder,
            index_corpus=request.index_corpus,
        )
        runtime.apply_pack_policies(pack)
        runtime.experts.refresh()
        runtime.router.invalidate()
        return report

    @app.post("/v1/packs/validate", tags=["packs"], dependencies=auth)
    async def validate_pack(request: PackInstallRequest) -> dict[str, Any]:
        runtime = get_runtime()
        pack = load_pack(request.path)
        problems = pack.validate(registry=runtime.models)
        return {"pack": pack.summary(), "valid": not problems, "problems": problems}

    # -- evaluation --------------------------------------------------------

    @app.post("/v1/eval/run", tags=["evaluation"], dependencies=auth)
    async def run_eval(request: EvalRequest) -> dict[str, Any]:
        runtime = get_runtime()
        if request.dataset:
            paths = [request.dataset]
            pack_name = ""
        elif request.pack:
            pack = load_pack(request.pack)
            paths = [str(p) for p in pack.eval_files]
            pack_name = pack.name
            if not paths:
                raise HTTPException(
                    status_code=400, detail=f"pack {request.pack!r} declares no evaluation files"
                )
        else:
            raise HTTPException(status_code=400, detail="pass 'dataset' or 'pack'")

        dataset = load_pack_datasets(paths, domain=request.domain)
        if not len(dataset):
            raise HTTPException(
                status_code=400, detail="evaluation dataset is empty after filtering"
            )
        harness = EvaluationHarness(runtime, pack_name=pack_name)

        federation = await harness.run_federation(dataset, repeats=request.repeats)
        payload: dict[str, Any] = {
            "federation": federation.metrics.model_dump(mode="json"),
            "run_id": federation.run_id,
            "cases": len(dataset),
        }
        if request.compare:
            baseline = await harness.run_baseline(dataset, repeats=request.repeats)
            comparison = compare(federation.metrics, baseline.metrics)
            payload["baseline"] = baseline.metrics.model_dump(mode="json")
            payload["baseline_run_id"] = baseline.run_id
            payload["comparison"] = comparison.model_dump(mode="json")
            payload["report"] = render_comparison(comparison, domain=request.domain)
        return payload

    @app.get("/v1/eval/runs", tags=["evaluation"], dependencies=auth)
    async def eval_runs(limit: int = Query(default=20, le=100)) -> list[dict[str, Any]]:
        return list_runs(get_runtime().tenant_id, limit=limit)

    @app.get("/v1/eval/runs/{run_id}", tags=["evaluation"], dependencies=auth)
    async def eval_run_detail(run_id: str) -> dict[str, Any]:
        from sqlalchemy import select

        from alm.persistence.models import EvalCaseRow, EvalRunRow

        runtime = get_runtime()
        with session_scope() as session:
            row = session.get(EvalRunRow, run_id)
            if row is None or row.tenant_id != runtime.tenant_id:
                raise HTTPException(status_code=404, detail="evaluation run not found")
            cases = session.execute(
                select(EvalCaseRow)
                .where(EvalCaseRow.run_id == run_id)
                .order_by(EvalCaseRow.id)
            ).scalars().all()
            return {
                "run_id": row.run_id,
                "name": row.name,
                "arm": row.arm,
                "pack": row.pack,
                "dataset": row.dataset,
                "domain": row.domain,
                "metrics": dict(row.metrics or {}),
                "config": dict(row.config or {}),
                "case_count": row.case_count,
                "created_at": row.created_at.isoformat() if row.created_at else "",
                "cases": [
                    {
                        "case_id": c.case_id,
                        "domain": c.domain,
                        "score": c.score,
                        "correct": c.correct,
                        "latency_ms": c.latency_ms,
                        "cost_usd": c.cost_usd,
                        "used_fallback": c.used_fallback,
                        "trace_complete": c.trace_complete,
                        "answer": c.answer,
                        "expected": c.expected,
                    }
                    for c in cases
                ],
            }

    @app.get("/v1/eval/datasets", tags=["evaluation"], dependencies=auth)
    async def eval_datasets() -> list[dict[str, Any]]:
        """Evaluation files declared by every pack under the packs directory.

        Backs the "run a new evaluation" picker — the operator chooses a pack
        by name instead of having to know a JSONL path on the server's disk.
        """
        settings = get_settings()
        out: list[dict[str, Any]] = []
        for path in discover_packs(settings.packs_dir):
            try:
                pack = load_pack(path)
            except ALMError as exc:
                out.append({"pack": path.name, "root": str(path), "error": exc.message})
                continue
            out.append(
                {
                    "pack": pack.name,
                    "root": str(pack.root),
                    "domains": pack.domains(),
                    "eval_files": [str(p) for p in pack.eval_files],
                }
            )
        return out

    # -- ollama --------------------------------------------------------------

    @app.get("/v1/ollama/tags", tags=["ollama"], dependencies=auth)
    async def ollama_tags() -> dict[str, Any]:
        """Models pulled on the local Ollama daemon — the orchestration picker's library."""
        base_url = get_settings().ollama_base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5.0) as http_client:
                response = await http_client.get(f"{base_url}/api/tags")
            response.raise_for_status()
        except Exception as exc:
            return {"reachable": False, "base_url": base_url, "error": str(exc), "models": []}

        data = response.json()
        models = [
            {
                "name": m.get("name", ""),
                "size_bytes": m.get("size", 0),
                "family": (m.get("details") or {}).get("family", ""),
                "parameter_size": (m.get("details") or {}).get("parameter_size", ""),
                "quantization": (m.get("details") or {}).get("quantization_level", ""),
                "modified_at": m.get("modified_at", ""),
            }
            for m in data.get("models", [])
        ]
        return {"reachable": True, "base_url": base_url, "models": models}

    @app.get("/v1/ollama/running", tags=["ollama"], dependencies=auth)
    async def ollama_running() -> dict[str, Any]:
        """Models currently loaded in the daemon's memory, from ``/api/ps``."""
        base_url = get_settings().ollama_base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5.0) as http_client:
                response = await http_client.get(f"{base_url}/api/ps")
            response.raise_for_status()
        except Exception as exc:
            return {"reachable": False, "base_url": base_url, "error": str(exc), "models": []}

        data = response.json()
        models = [
            {
                "name": m.get("name", ""),
                "size_vram_bytes": m.get("size_vram", 0),
                "expires_at": m.get("expires_at", ""),
            }
            for m in data.get("models", [])
        ]
        return {"reachable": True, "base_url": base_url, "models": models}

    # -- packs ---------------------------------------------------------------

    @app.get("/v1/packs", tags=["packs"], dependencies=auth)
    async def list_packs() -> list[dict[str, Any]]:
        settings = get_settings()
        out: list[dict[str, Any]] = []
        for path in discover_packs(settings.packs_dir):
            try:
                out.append(load_pack(path).summary())
            except ALMError as exc:
                out.append({"name": path.name, "root": str(path), "error": exc.message})
        return out

    # -- governance --------------------------------------------------------

    @app.get("/v1/audit", tags=["governance"], dependencies=auth)
    async def audit(
        session_id: str = Query(default=""),
        expert_id: str = Query(default=""),
        decision: str = Query(default=""),
        limit: int = Query(default=100, le=1000),
    ) -> list[dict[str, Any]]:
        log = AuditLog(get_runtime().tenant_id)
        return [
            record.to_dict()
            for record in log.query(
                session_id=session_id,
                expert_id=expert_id,
                decision=decision,
                limit=limit,
            )
        ]

    @app.get("/v1/audit/summary", tags=["governance"], dependencies=auth)
    async def audit_summary(hours: int = Query(default=24, le=24 * 90)) -> dict[str, Any]:
        return AuditLog(get_runtime().tenant_id).summary(hours=hours)

    @app.get("/v1/audit/export", tags=["governance"], dependencies=auth)
    async def audit_export(limit: int = Query(default=10_000, le=100_000)):
        """Stream the decision log as JSON Lines, for an external SIEM."""
        log = AuditLog(get_runtime().tenant_id)

        def generate():
            for line in log.export_jsonl(limit=limit):
                yield line + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @app.get("/v1/policies", tags=["governance"], dependencies=auth)
    async def policies() -> list[dict[str, Any]]:
        return [p.model_dump(mode="json") for p in get_runtime().ibac.policies.policies]


app = create_app()
