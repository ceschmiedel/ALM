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
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from alm import __version__
from alm.api.schemas import (
    AgentCreateRequest,
    AgentUpdateRequest,
    AskRequest,
    AskResponse,
    EvalRequest,
    ExplainRequest,
    GraphSearchRequest,
    HealthResponse,
    ModelCreateRequest,
    ModelProbeRequest,
    PackInstallRequest,
    PromotionCheckRequest,
    SearchRequest,
)
from alm.cmrag.ingest import CorpusIngestor, EntityAnnotator
from alm.cmrag.tabular import TabularError
from alm.config import get_settings
from alm.core.errors import ALMError
from alm.core.telemetry import Stopwatch
from alm.evaluation import (
    EvaluationHarness,
    list_runs,
    load_pack_datasets,
    render_comparison,
)
from alm.evaluation.metrics import compare
from alm.experts.pack import discover_packs, load_pack
from alm.experts.spec import CapabilitySpec, ExpertSpec, PromptSpec, RetrievalSpec
from alm.federation.runtime import FederationRuntime
from alm.governance.audit import AuditLog
from alm.graph.models import Node, NodeKind
from alm.graph.ontology import EntityType, Ontology
from alm.mlops.drift import DriftDetector, assess_promotion
from alm.mlops.versions import VersionRegistry
from alm.models.backends import available_backends, create_backend
from alm.models.registry import ModelRegistry
from alm.models.spec import GenerationRequest, Message, ModelSpec
from alm.models.tiers import TIER_GUIDANCE, parse_tier
from alm.persistence.database import init_db, session_scope
from alm.persistence.models import SessionRow
from alm.protocol.trace import TraceEvent

logger = logging.getLogger(__name__)

#: Upload ceiling. A spreadsheet beyond this is a data-pipeline job, not
#: something to push through a browser and embed synchronously.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

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
        app.mount("/ui", _RevalidatingStatic(directory=static_dir, html=True), name="dashboard")

    return app


class _RevalidatingStatic(StaticFiles):
    """Serve the console with ``Cache-Control: no-cache``.

    Without an explicit directive browsers fall back to *heuristic* caching and
    will happily serve a stale ``app.js`` without revalidating — which, after a
    `git pull`, produces a new ``index.html`` driving old JavaScript: menus that
    render but do nothing. ``no-cache`` means "revalidate before use", not
    "do not store", so the ETag still turns the check into a cheap 304.
    """

    def file_response(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


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

    # -- agent authoring ---------------------------------------------------

    @app.post("/v1/experts", tags=["experts"], dependencies=auth)
    async def create_expert(request: AgentCreateRequest) -> dict[str, Any]:
        """Create an Expert Agent interactively, without a pack on disk."""
        runtime = get_runtime()
        if runtime.experts.get(request.id) is not None:
            raise HTTPException(
                status_code=409, detail=f"expert {request.id!r} already exists"
            )
        if request.model and not runtime.models.exists(request.model):
            raise HTTPException(
                status_code=400,
                detail=f"model {request.model!r} is not registered",
            )
        if not request.capabilities:
            raise HTTPException(
                status_code=400,
                detail=(
                    "an agent needs at least one capability — it is what the "
                    "router matches a request against"
                ),
            )

        spec = _spec_from_request(request)
        _ensure_domain_and_entities(runtime, spec, create=request.create_domain)
        _register_expert(runtime, spec)
        return spec.model_dump(mode="json")

    @app.patch("/v1/experts/{expert_id}", tags=["experts"], dependencies=auth)
    async def update_expert(expert_id: str, request: AgentUpdateRequest) -> dict[str, Any]:
        runtime = get_runtime()
        node = runtime.graph.get_node(NodeKind.EXPERT, expert_id)
        if node is None:
            raise HTTPException(status_code=404, detail="expert not found")

        raw = node.attributes.get("spec")
        if not isinstance(raw, dict):
            raise HTTPException(
                status_code=409,
                detail="this expert node carries no specification; reinstall its pack",
            )
        spec = ExpertSpec.model_validate(raw)

        if request.model is not None:
            if request.model and not runtime.models.exists(request.model):
                raise HTTPException(
                    status_code=400, detail=f"model {request.model!r} is not registered"
                )
            spec.model = request.model
        if request.tier is not None:
            spec.tier = parse_tier(request.tier)
        for field in ("label", "description", "enabled", "authority"):
            value = getattr(request, field)
            if value is not None:
                setattr(spec, field, value)
        if request.retrieval_domains is not None:
            spec.retrieval.domains = request.retrieval_domains
        if request.retrieval_top_k is not None:
            spec.retrieval.top_k = request.retrieval_top_k
        if request.system_prompt is not None:
            spec.prompt.system = request.system_prompt
        if request.answer_language is not None:
            spec.prompt.answer_language = request.answer_language
        if request.capabilities is not None:
            if not request.capabilities:
                raise HTTPException(
                    status_code=400, detail="an agent must keep at least one capability"
                )
            spec.capabilities = [
                CapabilitySpec(**c.model_dump()) for c in request.capabilities
            ]

        _ensure_domain_and_entities(runtime, spec, create=True)
        _register_expert(runtime, spec)
        return spec.model_dump(mode="json")

    @app.delete("/v1/experts/{expert_id}", tags=["experts"], dependencies=auth)
    async def delete_expert(expert_id: str) -> dict[str, Any]:
        runtime = get_runtime()
        node = runtime.graph.get_node(NodeKind.EXPERT, expert_id)
        if node is None:
            raise HTTPException(status_code=404, detail="expert not found")

        # Capability nodes exist only to make this expert routable; leaving them
        # behind would keep the router matching a specialist that is gone.
        raw = node.attributes.get("spec")
        if isinstance(raw, dict):
            for capability in raw.get("capabilities", []) or []:
                capability_id = capability.get("id") if isinstance(capability, dict) else None
                if capability_id:
                    runtime.graph.delete_node(NodeKind.CAPABILITY, capability_id)

        runtime.graph.delete_node(NodeKind.EXPERT, expert_id)
        runtime.experts.refresh()
        runtime.router.invalidate()
        return {"deleted": expert_id}

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

    @app.get("/v1/models/backends", tags=["models"], dependencies=auth)
    async def model_backends() -> dict[str, Any]:
        """The backend and tier vocabulary, so the console never guesses names."""
        return {
            "backends": available_backends(),
            "tiers": [
                {
                    "id": str(tier),
                    "size": guidance["size"],
                    "use_for": guidance["use_for"],
                    "runs_on": guidance["runs_on"],
                }
                for tier, guidance in TIER_GUIDANCE.items()
            ],
            # Backends that reach a third party and therefore need a key.
            "hosted_backends": ["openai", "anthropic", "google", "azure", "openai_compat"],
        }

    @app.post("/v1/models/probe", tags=["models"], dependencies=auth)
    async def model_probe(request: ModelProbeRequest) -> dict[str, Any]:
        """Send one trivial completion to verify a model actually answers.

        A hosted fallback registered with a bad key otherwise looks healthy
        until the first escalation needs it — which is the worst moment to find
        out. This makes the failure immediate and legible.
        """
        runtime = get_runtime()
        if request.model_id:
            spec = runtime.models.get(request.model_id)
            if spec is None:
                raise HTTPException(status_code=404, detail="model not found")
        elif request.backend:
            spec = ModelSpec(
                model_id=f"probe-{request.backend}",
                backend=request.backend,
                model_name=request.model_name,
                endpoint=request.endpoint,
                api_key=request.api_key,
                params=request.params,
            )
        else:
            raise HTTPException(status_code=400, detail="pass 'model_id' or 'backend'")

        probe = GenerationRequest(
            messages=[Message(role="user", content="Responda apenas: ok")],
            max_tokens=16,
            temperature=0.0,
        )
        watch = Stopwatch()
        try:
            result = await create_backend(spec).generate(probe)
        except Exception as exc:
            return {
                "ok": False,
                "backend": spec.backend,
                "model": spec.served_name,
                "error": str(exc)[:500],
                "latency_ms": watch.stop(),
            }
        return {
            "ok": bool(result.text.strip()),
            "backend": spec.backend,
            "model": spec.served_name,
            "reply": result.text.strip()[:200],
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": round(result.latency_ms or watch.stop(), 2),
        }

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

    @app.post("/v1/cmrag/upload", tags=["cmrag"], dependencies=auth)
    async def cmrag_upload(
        file: UploadFile = File(...),
        domain: str = Form(...),
        sensitivity: str = Form("internal"),
    ) -> dict[str, Any]:
        """Ingest a CSV or XLSX upload into a domain's corpus.

        One document per sheet, chunked by row groups that each carry the
        header — see :mod:`alm.cmrag.tabular` for why a table cannot be split
        like prose.
        """
        runtime = get_runtime()
        if not domain.strip():
            raise HTTPException(status_code=400, detail="a domain is required")

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="the uploaded file is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"file is {len(data) / 1e6:.1f} MB; the limit is "
                    f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB"
                ),
            )

        annotator = EntityAnnotator()
        for ontology in _ontologies_from_graph(runtime):
            annotator.load(ontology)

        ingestor = CorpusIngestor(runtime.chunk_store, runtime.embedder, annotator=annotator)
        try:
            reports = ingestor.ingest_table(
                data,
                file.filename or "upload.csv",
                domain=domain.strip(),
                sensitivity=sensitivity or "internal",
            )
        except TabularError as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc

        return {
            "filename": file.filename,
            "domain": domain.strip(),
            "sheets": reports,
            "documents": len(reports),
            "chunks": sum(r.get("chunks", 0) for r in reports),
            "corpus": runtime.chunk_store.stats(),
        }

    @app.get("/v1/cmrag/documents", tags=["cmrag"], dependencies=auth)
    async def cmrag_documents(domain: str = Query(default="")) -> list[dict[str, Any]]:
        return get_runtime().chunk_store.documents(domain or None)

    @app.delete("/v1/cmrag/documents/{document_id}", tags=["cmrag"], dependencies=auth)
    async def cmrag_delete_document(document_id: str) -> dict[str, Any]:
        store = get_runtime().chunk_store
        if store.get_document(document_id) is None:
            raise HTTPException(status_code=404, detail="document not found")
        removed = store.delete_document(document_id)
        return {"deleted": document_id, "chunks_removed": removed}

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


def _ontologies_from_graph(runtime: FederationRuntime) -> list[Ontology]:
    """Rebuild each domain's ontology from the graph, for entity annotation.

    Uploaded data has to be annotated with the same entity types a pack's
    corpus was, or it would be invisible to the ontology-aware half of
    retrieval and would not inherit the sensitivity its domain implies.
    """
    by_domain: dict[str, list[EntityType]] = {}
    for node in runtime.graph.find_nodes(NodeKind.ENTITY_TYPE):
        attributes = node.attributes or {}
        by_domain.setdefault(node.domain, []).append(
            EntityType(
                name=node.key,
                label=node.label,
                description=node.description,
                keywords=list(attributes.get("keywords", []) or []),
                examples=list(attributes.get("examples", []) or []),
                sensitivity=str(attributes.get("sensitivity", "internal") or "internal"),
            )
        )
    return [
        Ontology(domain=domain, entities=entities)
        for domain, entities in by_domain.items()
        if domain
    ]


def _spec_from_request(request: AgentCreateRequest) -> ExpertSpec:
    """Turn the console's flat form into a full :class:`ExpertSpec`."""
    return ExpertSpec(
        id=request.id,
        domain=request.domain,
        label=request.label or request.id,
        description=request.description,
        model=request.model,
        tier=parse_tier(request.tier),
        capabilities=[CapabilitySpec(**c.model_dump()) for c in request.capabilities],
        authority=dict(request.authority),
        prompt=PromptSpec(
            system=request.system_prompt,
            answer_language=request.answer_language,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ),
        retrieval=RetrievalSpec(
            domains=list(request.retrieval_domains),
            top_k=request.retrieval_top_k,
        ),
        enabled=request.enabled,
        metadata={"source": "console"},
    )


def _ensure_domain_and_entities(
    runtime: FederationRuntime, spec: ExpertSpec, *, create: bool
) -> None:
    """Make the graph able to route to this agent.

    Pack installation gets its domain and entity types from an ontology file.
    An agent authored in the console has no such file, so the domain and any
    entity types its capabilities name are registered here — otherwise the
    capability would reference an entity the graph does not know and the
    router would silently never match it well.
    """
    if runtime.graph.get_node(NodeKind.DOMAIN, spec.domain) is None:
        if not create:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"domain {spec.domain!r} does not exist; install a pack that "
                    f"defines it or set create_domain"
                ),
            )
        runtime.graph.register_domain(
            spec.domain,
            label=spec.domain,
            description=f"Domínio criado pelo console para o agente {spec.id}",
        )

    for domain in spec.retrieval.domains:
        if domain and runtime.graph.get_node(NodeKind.DOMAIN, domain) is None:
            runtime.graph.register_domain(domain, label=domain)

    for entity in spec.entity_types():
        if runtime.graph.get_node(NodeKind.ENTITY_TYPE, entity) is None:
            runtime.graph.upsert_node(
                Node(
                    kind=NodeKind.ENTITY_TYPE,
                    key=entity,
                    label=entity,
                    domain=spec.domain,
                    attributes={"keywords": [], "examples": [], "sensitivity": "internal"},
                )
            )


def _register_expert(runtime: FederationRuntime, spec: ExpertSpec) -> None:
    """Write the agent onto the graph and make the live runtime pick it up."""
    runtime.graph.register_expert(
        spec.id,
        domain=spec.domain,
        description=spec.description or spec.label,
        model_id=spec.model,
        capabilities=spec.declarations(),
        authority=spec.authority,
        attributes={
            "spec": spec.model_dump(mode="json"),
            "pack": "",
            "tier": str(spec.tier),
        },
    )
    runtime.experts.refresh()
    runtime.router.invalidate()


app = create_app()
