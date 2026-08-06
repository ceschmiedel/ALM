"""The ``alm`` command-line interface.

Two conventions throughout:

* every command accepts ``--json`` and prints a machine-readable object, so the
  CLI composes into scripts instead of only into terminals;
* nothing is silently skipped. When a command degrades — no orchestrator, no
  embedder, an uncalibrated expert — it says so on stderr rather than producing
  a quietly worse result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from alm import __version__
from alm.core.errors import ALMError
from alm.core.logging import configure_logging

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_COLOUR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text


def bold(text: str) -> str:
    return _c(text, "1")


def dim(text: str) -> str:
    return _c(text, "2")


def green(text: str) -> str:
    return _c(text, "32")


def yellow(text: str) -> str:
    return _c(text, "33")


def red(text: str) -> str:
    return _c(text, "31")


def cyan(text: str) -> str:
    return _c(text, "36")


def emit(payload: Any, *, as_json: bool) -> bool:
    """Print JSON when asked.  Returns True when it handled the output."""
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return True
    return False


def warn(message: str) -> None:
    print(f"{yellow('warning')}  {message}", file=sys.stderr)


def fail(message: str, code: int = 1) -> None:
    print(f"{red('error')}  {message}", file=sys.stderr)
    raise SystemExit(code)


def heading(text: str) -> None:
    print()
    print(bold(text))
    print(dim("─" * max(len(text), 40)))


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def _runtime(pack: str = "", *, install: bool = False):
    from alm.federation.runtime import FederationRuntime
    from alm.persistence.database import init_db

    init_db()
    if pack:
        return FederationRuntime.from_pack(pack, install=install)
    return FederationRuntime.from_database()


def _report_degradation(runtime) -> None:
    health = runtime.health()
    if not health["orchestrator_configured"]:
        warn(
            "no orchestrator model registered — planner decomposition, the LLM judge "
            "and the fallback path are unavailable (`alm model add --tier orchestrator`)"
        )
    if not health["semantic_routing"]:
        warn("no embedder configured — routing and retrieval are lexical only")
    if health["experts_on_placeholder_backends"]:
        warn(
            "these experts are bound to placeholder backends, not language models: "
            + ", ".join(health["experts_on_placeholder_backends"])
        )


# ---------------------------------------------------------------------------
# init / install / serve
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> None:
    from alm.persistence.database import get_engine, init_db

    init_db()
    url = get_engine().url
    if emit({"database": str(url), "status": "ready"}, as_json=args.json):
        return
    print(f"{green('✓')} database ready at {cyan(str(url))}")
    print(dim("  next: alm pack install packs/demo-enterprise"))


def cmd_install(args: argparse.Namespace) -> None:
    """Interactive setup wizard that writes a ``.env``."""
    env_path = Path(args.env or ".env")
    if env_path.exists() and not args.force:
        fail(f"{env_path} already exists; pass --force to overwrite")

    print(bold("ALM setup"))
    print(dim("Press enter to accept the default shown in brackets.\n"))

    def ask(label: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        try:
            value = input(f"  {label}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(130) from None
        return value or default

    database = ask("Database URL", "sqlite:///alm.db")
    provider = ask("Model backend (ollama/openai/anthropic/vllm/heuristic)", "ollama")

    lines = [
        "# Written by `alm install`",
        f"ALM_DATABASE_URL={database}",
        "ALM_HOME=.alm",
        "ALM_PACKS_DIR=packs",
        "",
    ]

    if provider == "ollama":
        base_url = ask("Ollama base URL", "http://localhost:11434")
        lines += [
            f"ALM_OLLAMA_BASE_URL={base_url}",
            "ALM_EMBEDDING_PROVIDER=ollama",
            "ALM_EMBEDDING_MODEL=nomic-embed-text",
            "",
            "# Register models with: alm model add <id> --tier <tier> --backend ollama --model <name>",
        ]
    elif provider == "openai":
        key = ask("OpenAI API key", "")
        lines += [
            f"OPENAI_API_KEY={key}",
            "ALM_EMBEDDING_PROVIDER=openai",
            "ALM_EMBEDDING_MODEL=text-embedding-3-small",
        ]
    elif provider == "anthropic":
        key = ask("Anthropic API key", "")
        lines += [
            f"ANTHROPIC_API_KEY={key}",
            "# Anthropic has no embeddings endpoint; using local hash embeddings.",
            "ALM_EMBEDDING_PROVIDER=hash",
        ]
    elif provider == "vllm":
        base_url = ask("vLLM base URL", "http://localhost:8000/v1")
        lines += [f"ALM_VLLM_BASE_URL={base_url}", "ALM_EMBEDDING_PROVIDER=hash"]
    else:
        lines += ["ALM_EMBEDDING_PROVIDER=hash"]

    lines += [
        "",
        "ALM_ROUTER_CONFIDENCE_THRESHOLD=0.55",
        "ALM_FALLBACK_ENABLED=true",
        "ALM_GOVERNANCE_ENABLED=true",
        "ALM_LOG_LEVEL=INFO",
        "",
    ]
    env_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"{green('✓')} wrote {cyan(str(env_path))}")
    print(dim("  next: alm init && alm pack install packs/demo-enterprise"))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from alm.config import get_settings

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port

    print(f"{green('▸')} ALM API on {cyan(f'http://{host}:{port}')}")
    print(dim(f"  docs: http://{host}:{port}/docs"))
    if not settings.api_token:
        warn("ALM_API_TOKEN is unset — the API is unauthenticated. Do not expose it.")

    uvicorn.run(
        "alm.api.app:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace) -> None:
    async def run() -> None:
        runtime = _runtime(args.pack)
        if not args.json:
            _report_degradation(runtime)

        listener = None
        if args.follow and not args.json:
            def listener(event):  # noqa: ANN001
                print(dim(f"  {event.line()}"), file=sys.stderr)

        try:
            result = await runtime.run(
                args.intent,
                principal=args.principal,
                claims=args.claim or [],
                on_event=listener,
            )
        finally:
            await runtime.close()

        if emit(result.to_dict(include_trace=args.trace), as_json=args.json):
            return

        heading("Answer")
        print(result.answer)

        if result.citations:
            heading("Sources")
            for citation in result.citations[:8]:
                title = citation.title or citation.document_id
                print(f"  {dim('·')} {title} {dim(f'({citation.score:.2f})')}")

        if result.audit_trail():
            heading("Decision record")
            for line in result.audit_trail():
                print(f"  {dim('·')} {line}")

        if args.trace:
            heading("Trace")
            print(result.trace.render(include_detail=args.verbose))

        heading("Run")
        strategy = result.plan.strategy if result.plan else "-"
        colour = yellow if result.metrics.used_fallback else green
        print(f"  strategy    {colour(strategy)}")
        print(f"  experts     {', '.join(result.experts) or '-'}")
        print(f"  confidence  {result.confidence:.2f}")
        print(f"  latency     {result.metrics.wall_ms:.0f} ms")
        print(f"  cost        ${result.metrics.cost_usd:.6f}")
        print(f"  parallelism {result.metrics.dag_width} wide × {result.metrics.dag_depth} deep")
        if result.metrics.conflicts:
            print(
                f"  conflicts   {result.metrics.conflicts_resolved}"
                f"/{result.metrics.conflicts} resolved"
            )

    asyncio.run(run())


def cmd_explain(args: argparse.Namespace) -> None:
    runtime = _runtime(args.pack)
    explanation = runtime.router.explain(args.intent, top_k=args.top_k)
    if emit(explanation, as_json=args.json):
        return

    heading("Routing signals")
    print(f"  entities detected  {', '.join(explanation['entities_detected']) or '-'}")
    print(f"  domains            {', '.join(explanation['domains']) or '-'}")
    print(f"  semantic available {explanation['semantic_available']}")
    print()
    print(f"  {'capability':<24}{'score':>8}{'sem':>8}{'lex':>8}{'ent':>8}  expert")
    print("  " + dim("─" * 72))
    for match in explanation["matches"]:
        print(
            f"  {match['capability_id']:<24}{match['score']:>8.3f}"
            f"{match['semantic_score']:>8.3f}{match['lexical_score']:>8.3f}"
            f"{match['entity_score']:>8.3f}  {match['expert_id']}"
        )
    print()
    verdict = (
        yellow("would escalate to the orchestrator")
        if explanation["would_escalate"]
        else green("would route to the federation")
    )
    print(f"  {verdict} (threshold {explanation['threshold']:.2f})")


# ---------------------------------------------------------------------------
# packs
# ---------------------------------------------------------------------------


def cmd_pack_list(args: argparse.Namespace) -> None:
    from alm.config import get_settings
    from alm.experts.pack import discover_packs, load_pack

    directory = args.dir or str(get_settings().packs_dir)
    packs = discover_packs(directory)
    summaries = []
    for path in packs:
        try:
            summaries.append(load_pack(path).summary())
        except ALMError as exc:
            summaries.append({"name": path.name, "root": str(path), "error": exc.message})

    if emit(summaries, as_json=args.json):
        return
    if not summaries:
        print(dim(f"no packs found under {directory}"))
        return

    heading(f"Packs in {directory}")
    for summary in summaries:
        if "error" in summary:
            print(f"  {red('✗')} {summary['name']}  {dim(summary['error'])}")
            continue
        counts = (
            f"{len(summary['domains'])} domain(s), {len(summary['experts'])} expert(s)"
        )
        print(f"  {green('•')} {bold(summary['name'])} v{summary['version']}  {dim(counts)}")


def cmd_pack_validate(args: argparse.Namespace) -> None:
    from alm.experts.pack import load_pack

    runtime = _runtime()
    pack = load_pack(args.path)
    problems = pack.validate(registry=runtime.models)

    if emit(
        {"pack": pack.summary(), "valid": not problems, "problems": problems},
        as_json=args.json,
    ):
        raise SystemExit(1 if problems else 0)

    heading(f"Validating {pack.name} v{pack.version}")
    print(f"  domains   {', '.join(pack.domains())}")
    print(f"  experts   {', '.join(e.id for e in pack.experts)}")
    print(f"  models    {', '.join(m.model_id for m in pack.models) or '-'}")
    print(f"  policies  {len(pack.policies)}")
    print()
    if not problems:
        print(f"  {green('✓')} pack is valid")
        return
    for problem in problems:
        print(f"  {red('✗')} {problem}")
    raise SystemExit(1)


def cmd_pack_install(args: argparse.Namespace) -> None:
    from alm.experts.pack import load_pack

    runtime = _runtime()
    pack = load_pack(args.path)
    report = pack.install(
        graph=runtime.graph,
        registry=runtime.models,
        store=runtime.chunk_store,
        embedder=runtime.embedder,
        index_corpus=not args.skip_corpus,
    )
    runtime.apply_pack_policies(pack)
    runtime.experts.refresh()

    if emit(report, as_json=args.json):
        return
    heading(f"Installed {pack.name} v{pack.version}")
    print(f"  domains       {', '.join(report['domains'])}")
    print(f"  entity types  {report['entities']}")
    print(f"  experts       {', '.join(report['experts'])}")
    print(f"  capabilities  {report['capabilities']}")
    print(f"  models        {report['models']}")
    print(f"  documents     {report['documents']} ({report['chunks']} chunks)")
    print(f"  policies      {report['policies']}")
    print()
    print(dim("  try: alm ask \"<a question about this domain>\" --trace"))


# ---------------------------------------------------------------------------
# graph / experts / models / cmrag
# ---------------------------------------------------------------------------


def cmd_graph_show(args: argparse.Namespace) -> None:
    runtime = _runtime()
    nodes = runtime.graph.find_nodes(args.kind or None, domain=args.domain or None)
    payload = [
        {
            "kind": str(n.kind),
            "key": n.key,
            "domain": n.domain,
            "label": n.label,
            "description": n.description[:160],
        }
        for n in nodes
    ]
    if emit(payload, as_json=args.json):
        return

    heading("Context Graph")
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for node in payload:
        by_kind.setdefault(node["kind"], []).append(node)
    for kind in sorted(by_kind):
        print(f"\n  {bold(kind)} ({len(by_kind[kind])})")
        for node in sorted(by_kind[kind], key=lambda n: n["key"]):
            domain = dim(f"[{node['domain']}]") if node["domain"] else ""
            print(f"    {node['key']:<28} {domain}")


def cmd_graph_stats(args: argparse.Namespace) -> None:
    runtime = _runtime()
    stats = runtime.graph.stats().model_dump()
    if emit(stats, as_json=args.json):
        return
    heading("Context Graph")
    print(f"  nodes    {stats['nodes']}")
    print(f"  edges    {stats['edges']}")
    print(f"  domains  {', '.join(stats['domains']) or '-'}")
    print(f"  experts  {', '.join(stats['experts']) or '-'}")
    print()
    for kind, count in sorted(stats["by_kind"].items()):
        print(f"    {kind:<16}{count:>5}")


def cmd_expert_list(args: argparse.Namespace) -> None:
    runtime = _runtime()
    inventory = runtime.experts.inventory()
    if emit(inventory, as_json=args.json):
        return
    if not inventory:
        print(dim("no experts installed — try `alm pack install packs/demo-enterprise`"))
        return

    heading("Expert Agents")
    print(f"  {'id':<22}{'domain':<12}{'model':<18}{'tier':<14}{'calibrated':<12}caps")
    print("  " + dim("─" * 86))
    for expert in inventory:
        calibrated = (
            green(f"yes ({expert['calibration_samples']})")
            if expert["calibrated"]
            else yellow("no")
        )
        print(
            f"  {expert['id']:<22}{expert['domain']:<12}{expert['model']:<18}"
            f"{expert['tier']:<14}{calibrated:<12}{len(expert['capabilities'])}"
        )
    print()
    print(dim("  uncalibrated experts are discounted in confidence arbitration"))


def cmd_expert_show(args: argparse.Namespace) -> None:
    runtime = _runtime()
    agent = runtime.experts.get(args.expert_id)
    if agent is None:
        fail(f"expert {args.expert_id!r} is not installed")
    spec = agent.spec
    if emit(spec.model_dump(mode="json"), as_json=args.json):
        return

    heading(f"{spec.label} ({spec.id})")
    print(f"  domain     {spec.domain}")
    print(f"  model      {spec.model or '-'} ({spec.tier})")
    print(f"  retrieval  {', '.join(spec.retrieval.domains)} · top {spec.retrieval.top_k}")
    print(f"  authority  {json.dumps(spec.authority)}")
    print(f"  verifier   {len(spec.verifier_rules)} rule(s)")
    heading("Capabilities")
    for capability in spec.capabilities:
        print(f"  {bold(capability.id)}")
        print(f"    {capability.description}")
        print(f"    {dim('operates on: ' + (', '.join(capability.operates_on) or '-'))}")
        if capability.produces:
            print(f"    {dim('produces: ' + ', '.join(capability.produces))}")


def cmd_expert_promote_check(args: argparse.Namespace) -> None:
    from alm.mlops.drift import assess_promotion

    runtime = _runtime()
    if runtime.experts.get(args.expert_id) is None:
        fail(f"expert {args.expert_id!r} is not installed")

    assessment = assess_promotion(
        expert_id=args.expert_id,
        tenant_id=runtime.tenant_id,
        sovereignty_required=args.sovereignty,
        monthly_calls=args.calls,
        volume_threshold=args.volume_threshold,
        latency_requirement_ms=args.latency_requirement,
        observed_p95_ms=args.observed_p95,
        rag_accuracy=args.rag_accuracy,
        accuracy_target=args.accuracy_target,
    )
    if emit(assessment.model_dump(mode="json"), as_json=args.json):
        return
    print()
    print(assessment.render())


def cmd_model_list(args: argparse.Namespace) -> None:
    runtime = _runtime()
    registry = runtime.models
    models = registry.list()
    serving = registry.serving_summary()

    if emit(
        {
            "models": [m.model_dump(mode="json", exclude={"api_key"}) for m in models],
            "serving": serving,
        },
        as_json=args.json,
    ):
        return

    if not models:
        print(dim("no models registered — try `alm model add`"))
        return

    heading("Models")
    print(f"  {'id':<22}{'tier':<14}{'backend':<12}{'serves':<28}cost/1k in-out")
    print("  " + dim("─" * 92))
    for spec in models:
        serves = (
            f"{spec.adapter} on {spec.base_model}"
            if spec.is_adapter
            else (spec.model_name or spec.model_id)
        )
        cost = f"${spec.cost_per_1k_input:.5f} / ${spec.cost_per_1k_output:.5f}"
        print(
            f"  {spec.model_id:<22}{str(spec.tier):<14}{spec.backend:<12}"
            f"{serves[:27]:<28}{cost}"
        )

    heading("Serving topology")
    print(f"  shared bases    {serving['shared_bases']}")
    print(f"  adapters        {serving['adapters']}")
    print(f"  full loads saved {serving['loads_saved']}")
    if serving["adapter_groups"]:
        print()
        for base, adapters in serving["adapter_groups"].items():
            print(f"    {base}  {dim('←')}  {', '.join(adapters)}")
        print()
        print(
            dim(
                "  many experts, one GPU-resident base: cost is base + adapters, "
                "not N models"
            )
        )


def cmd_model_add(args: argparse.Namespace) -> None:
    from alm.models.spec import ModelSpec
    from alm.models.tiers import parse_tier

    runtime = _runtime()
    params: dict[str, Any] = {}
    for item in args.param or []:
        if "=" not in item:
            fail(f"--param expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value

    spec = ModelSpec(
        model_id=args.model_id,
        tier=parse_tier(args.tier),
        backend=args.backend,
        model_name=args.model or "",
        base_model=args.base_model or "",
        adapter=args.adapter or "",
        adapter_uri=args.adapter_uri or "",
        endpoint=args.endpoint or "",
        api_key=args.api_key or "",
        params=params,
        context_window=args.context_window,
        cost_per_1k_input=args.cost_in,
        cost_per_1k_output=args.cost_out,
        description=args.description or "",
    )
    runtime.models.register(spec)

    if emit(spec.model_dump(mode="json", exclude={"api_key"}), as_json=args.json):
        return
    print(f"{green('✓')} registered {bold(spec.model_id)} ({spec.tier} · {spec.backend})")
    if spec.is_adapter:
        group = runtime.models.adapter_groups().get(spec.base_model, [])
        print(
            dim(
                f"  {len(group)} adapter(s) now share the base {spec.base_model} — "
                f"one set of GPU-resident weights"
            )
        )
    if spec.cost_per_1k_input == 0.0 and spec.cost_per_1k_output == 0.0:
        warn(
            f"{spec.model_id} has no price set, so cost metrics will read $0. "
            "Use --cost-in/--cost-out, or `alm model cost` for a self-hosted estimate."
        )


def cmd_model_remove(args: argparse.Namespace) -> None:
    runtime = _runtime()
    if not runtime.models.remove(args.model_id):
        fail(f"model {args.model_id!r} is not registered")
    print(f"{green('✓')} removed {args.model_id}")


def cmd_model_cost(args: argparse.Namespace) -> None:
    from alm.models.registry import selfhosted_cost_per_1k

    cost = selfhosted_cost_per_1k(
        args.gpu_hourly, args.tokens_per_second, utilisation=args.utilisation
    )
    payload = {
        "gpu_hourly_usd": args.gpu_hourly,
        "tokens_per_second": args.tokens_per_second,
        "utilisation": args.utilisation,
        "cost_per_1k_tokens_usd": cost,
    }
    if emit(payload, as_json=args.json):
        return
    heading("Self-hosted cost")
    print(f"  GPU rate        ${args.gpu_hourly:.2f}/hour")
    print(f"  throughput      {args.tokens_per_second:,.0f} tokens/s")
    print(f"  utilisation     {args.utilisation:.0%}")
    print(f"  {bold('cost per 1k')}     ${cost:.8f}")
    print()
    print(dim("  register it with: alm model add <id> --cost-in <v> --cost-out <v>"))
    print(
        dim(
            "  comparing a federation to a hosted LLM on token price alone is the "
            "mistake this avoids"
        )
    )


def cmd_model_versions(args: argparse.Namespace) -> None:
    from alm.mlops.versions import VersionRegistry

    runtime = _runtime()
    history = VersionRegistry(runtime.tenant_id).history(args.model_id)
    if emit(history, as_json=args.json):
        return
    if not history:
        print(dim(f"no version history for {args.model_id}"))
        return
    heading(f"Versions of {args.model_id}")
    for version in history:
        marker = green("● active") if version["status"] == "active" else dim(version["status"])
        print(f"  {version['version']:<12}{marker:<20}{dim(version['created_at'][:19])}")
        if version["metrics"]:
            print(f"      {dim(json.dumps(version['metrics']))}")


def cmd_model_promote(args: argparse.Namespace) -> None:
    from alm.mlops.versions import VersionRegistry

    runtime = _runtime()
    registry = VersionRegistry(runtime.tenant_id)
    try:
        version = registry.promote(
            args.model_id, args.version, metric=args.metric, force=args.force
        )
    except ALMError as exc:
        fail(exc.message)
    if emit(version.model_dump(mode="json"), as_json=args.json):
        return
    print(f"{green('✓')} {args.model_id} is now serving version {bold(args.version)}")


def cmd_model_rollback(args: argparse.Namespace) -> None:
    from alm.mlops.versions import VersionRegistry

    runtime = _runtime()
    try:
        version = VersionRegistry(runtime.tenant_id).rollback(args.model_id, args.version or "")
    except ALMError as exc:
        fail(exc.message)
    if emit(version.model_dump(mode="json"), as_json=args.json):
        return
    print(f"{green('✓')} rolled {args.model_id} back to version {bold(version.version)}")


def cmd_cmrag_index(args: argparse.Namespace) -> None:
    from alm.cmrag.ingest import CorpusIngestor, EntityAnnotator

    runtime = _runtime()
    annotator = EntityAnnotator()
    from alm.graph.models import NodeKind
    from alm.graph.ontology import EntityType, Ontology

    # Rebuild annotator patterns from the installed ontology, so indexing a new
    # corpus uses the same entity vocabulary the federation routes on.
    for domain in runtime.graph.domains():
        entities = [
            EntityType(
                name=node.key,
                keywords=list(node.attributes.get("keywords", [])),
                sensitivity=str(node.attributes.get("sensitivity", "internal")),
            )
            for node in runtime.graph.find_nodes(NodeKind.ENTITY_TYPE, domain=domain)
        ]
        annotator.load(Ontology(domain=domain, entities=entities))

    ingestor = CorpusIngestor(runtime.chunk_store, runtime.embedder, annotator=annotator)
    reports = ingestor.ingest_directory(args.path, domain=args.domain or "")

    if emit(reports, as_json=args.json):
        return
    indexed = [r for r in reports if not r.get("skipped")]
    heading("Indexed")
    for report in indexed:
        print(f"  {green('•')} {report['title']:<40}{report['chunks']:>4} chunks")
    skipped = len(reports) - len(indexed)
    print()
    print(f"  {len(indexed)} document(s) indexed, {skipped} unchanged")


def cmd_cmrag_search(args: argparse.Namespace) -> None:
    runtime = _runtime()
    retriever = runtime.retriever
    if args.expert:
        agent = runtime.experts.get(args.expert)
        if agent is None:
            fail(f"expert {args.expert!r} is not installed")
        retriever = retriever.with_access_filter(
            runtime.ibac.context_filter(
                expert_id=agent.id, domain=agent.domain, session_id="cli-search"
            )
        )

    result = retriever.retrieve(
        args.query, domains=args.domain or None, top_k=args.top_k
    )
    if emit(result.to_dict(), as_json=args.json):
        return

    heading(f"Retrieved {len(result.chunks)} fragment(s)")
    for chunk in result.chunks:
        print(f"  {bold(f'{chunk.score:.3f}')}  {chunk.title} {dim(f'[{chunk.domain}]')}")
        print(f"        {chunk.text[:160].strip()}…")
        print(
            dim(
                f"        entities: {', '.join(chunk.entities) or '-'} · "
                f"sensitivity: {chunk.sensitivity}"
            )
        )
    if result.denied:
        print()
        print(yellow(f"  {len(result.denied)} fragment(s) withheld by governance"))
    if result.degraded:
        warn("semantic scoring unavailable; results are lexical only")


def cmd_cmrag_stats(args: argparse.Namespace) -> None:
    runtime = _runtime()
    stats = runtime.retriever.stats()
    if emit(stats, as_json=args.json):
        return
    heading("CMRAG corpus")
    print(f"  documents  {stats['documents']}")
    print(f"  chunks     {stats['chunks']}")
    print(f"  embedder   {stats['embedder']}")
    for domain, count in sorted(stats["chunks_by_domain"].items()):
        print(f"    {domain:<16}{count:>5}")
    if stats.get("needs_reindex"):
        warn("stored vectors came from a different embedder — run `alm cmrag reindex`")


def cmd_cmrag_reindex(args: argparse.Namespace) -> None:
    from alm.cmrag.ingest import CorpusIngestor

    runtime = _runtime()
    ingestor = CorpusIngestor(runtime.chunk_store, runtime.embedder)
    count = ingestor.reindex(args.domain or None)
    if emit({"reindexed": count}, as_json=args.json):
        return
    print(f"{green('✓')} re-embedded {count} chunk(s) with {runtime.embedder.signature}")


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def cmd_eval_run(args: argparse.Namespace) -> None:
    async def run() -> None:
        from alm.evaluation import (
            EvaluationHarness,
            load_pack_datasets,
            render_comparison,
        )
        from alm.evaluation.metrics import compare
        from alm.experts.pack import load_pack

        runtime = _runtime()
        if args.dataset:
            paths = [args.dataset]
        elif args.pack:
            paths = [str(p) for p in load_pack(args.pack).eval_files]
            if not paths:
                fail(f"pack {args.pack} declares no evaluation files")
        else:
            fail("pass --dataset <file.jsonl> or --pack <dir>")

        dataset = load_pack_datasets(paths, domain=args.domain or "")
        if not len(dataset):
            fail("evaluation dataset is empty after filtering")

        harness = EvaluationHarness(runtime, pack_name=args.pack or "")

        if not args.json:
            print(dim(f"running {len(dataset)} case(s)…"))

        federation = await harness.run_federation(dataset, repeats=args.repeats)
        payload: dict[str, Any] = {
            "federation": federation.metrics.model_dump(mode="json"),
            "run_id": federation.run_id,
        }

        comparison = None
        if args.compare:
            baseline = await harness.run_baseline(dataset, repeats=args.repeats)
            comparison = compare(federation.metrics, baseline.metrics)
            payload["baseline"] = baseline.metrics.model_dump(mode="json")
            payload["comparison"] = comparison.model_dump(mode="json")

        await runtime.close()

        if emit(payload, as_json=args.json):
            return

        if comparison is not None:
            print()
            print(render_comparison(comparison, domain=args.domain or dataset.domain))
        else:
            metrics = federation.metrics
            heading(f"Federation · {metrics.cases} cases")
            for name, value in metrics.to_row().items():
                print(f"  {name:<24}{value:>12.4f}")

    asyncio.run(run())


def cmd_eval_calibrate(args: argparse.Namespace) -> None:
    async def run() -> None:
        from alm.evaluation import EvaluationHarness, load_pack_datasets
        from alm.experts.pack import load_pack

        runtime = _runtime()
        paths = [args.dataset] if args.dataset else [
            str(p) for p in load_pack(args.pack).eval_files
        ]
        dataset = load_pack_datasets(paths, domain=args.domain or "")
        harness = EvaluationHarness(runtime)
        report = await harness.fit_calibrations(dataset)
        await runtime.close()

        if emit(report, as_json=args.json):
            return
        heading("Confidence calibration")
        for expert_id, calibration in sorted(report.items()):
            status = (
                green("fitted")
                if calibration["fitted"]
                else yellow(f"needs more data ({calibration['samples']} samples)")
            )
            print(f"  {expert_id:<24}{status}")
            if calibration["fitted"]:
                print(
                    dim(
                        f"      Brier {calibration['brier_before']:.4f} → "
                        f"{calibration['brier_after']:.4f}"
                    )
                )
        print()
        print(dim("  only calibrated confidence is allowed to decide an arbitration"))

    asyncio.run(run())


def cmd_eval_list(args: argparse.Namespace) -> None:
    from alm.evaluation import list_runs

    runtime = _runtime()
    runs = list_runs(runtime.tenant_id, limit=args.limit)
    if emit(runs, as_json=args.json):
        return
    if not runs:
        print(dim("no evaluation runs recorded"))
        return
    heading("Evaluation runs")
    print(f"  {'run':<22}{'arm':<12}{'domain':<12}{'cases':>6}{'accuracy':>10}")
    print("  " + dim("─" * 64))
    for run in runs:
        accuracy = run["metrics"].get("domain_accuracy", 0.0)
        print(
            f"  {run['run_id']:<22}{run['arm']:<12}{run['domain'] or '-':<12}"
            f"{run['cases']:>6}{accuracy:>10.3f}"
        )


# ---------------------------------------------------------------------------
# distillation
# ---------------------------------------------------------------------------


def _pipeline(runtime, dataset: str):
    from alm.distillation.pipeline import DistillationPipeline

    return DistillationPipeline(
        serving=runtime.serving,
        chunk_store=runtime.chunk_store,
        tenant_id=runtime.tenant_id,
        dataset=dataset,
    )


def cmd_distill_seed(args: argparse.Namespace) -> None:
    runtime = _runtime()
    seeds = _pipeline(runtime, args.dataset).seed(args.domain, limit=args.limit)
    if emit({"domain": args.domain, "passages": len(seeds)}, as_json=args.json):
        return
    print(f"{green('✓')} {len(seeds)} source passage(s) available for {args.domain}")
    print(dim("  next: alm distill generate --domain " + args.domain))


def cmd_distill_generate(args: argparse.Namespace) -> None:
    async def run() -> None:
        runtime = _runtime()
        if runtime.models.orchestrator() is None:
            fail(
                "distillation needs a teacher: register an orchestrator model with "
                "`alm model add <id> --tier orchestrator ...`"
            )
        ontology = _ontology_for(runtime, args.domain)
        examples = await _pipeline(runtime, args.dataset).generate(
            args.domain,
            ontology=ontology,
            per_passage=args.per_passage,
            limit=args.limit,
        )
        await runtime.close()
        if emit({"domain": args.domain, "generated": len(examples)}, as_json=args.json):
            return
        print(f"{green('✓')} teacher produced {len(examples)} example(s)")
        print(dim("  next: alm distill filter --domain " + args.domain))

    asyncio.run(run())


def cmd_distill_filter(args: argparse.Namespace) -> None:
    runtime = _runtime()
    ontology = _ontology_for(runtime, args.domain)
    report = _pipeline(runtime, args.dataset).filter(args.domain, ontology=ontology)
    if emit(report, as_json=args.json):
        return
    heading(f"Filtered {report['reviewed']} example(s)")
    print(f"  accepted  {green(str(report['accepted']))}")
    print(f"  rejected  {report['rejected']}")
    for reason, count in sorted(report.get("rejection_reasons", {}).items()):
        print(f"    {dim('·')} {reason}: {count}")
    print()
    print(dim("  bad data teaches errors persistently — this stage is not optional"))


def cmd_distill_export(args: argparse.Namespace) -> None:
    runtime = _runtime()
    try:
        report = _pipeline(runtime, args.dataset).export(
            args.domain, args.out, eval_ratio=args.eval_ratio
        )
    except ALMError as exc:
        fail(exc.message)
    if emit(report, as_json=args.json):
        return
    print(f"{green('✓')} {report['train']} train / {report['eval']} eval example(s)")
    print(f"  train  {cyan(report['train_path'])}")
    print(f"  eval   {cyan(report['eval_path'])}")
    print()
    print(dim("  the eval split is never seen during training"))


def cmd_distill_feedback(args: argparse.Namespace) -> None:
    runtime = _runtime()
    report = _pipeline(runtime, args.dataset).ingest_feedback(args.domain or "")
    if emit(report, as_json=args.json):
        return
    heading("Production feedback")
    print(f"  captured  {report['captured']} case(s)")
    for kind, count in sorted(report.get("by_kind", {}).items()):
        print(f"    {dim('·')} {kind}: {count}")
    print()
    print(dim("  every failure and escalation is next cycle's training data"))


def cmd_distill_stats(args: argparse.Namespace) -> None:
    runtime = _runtime()
    stats = _pipeline(runtime, args.dataset).stats(args.domain or "")
    if emit(stats, as_json=args.json):
        return
    heading(f"Dataset {stats['dataset']}")
    print(f"  total  {stats['total']}")
    for status, count in sorted(stats["by_status"].items()):
        print(f"    {status:<16}{count:>5}")


def _ontology_for(runtime, domain: str):
    """Rebuild an Ontology object from what is installed in the graph."""
    from alm.graph.models import NodeKind
    from alm.graph.ontology import EntityType, Ontology

    nodes = runtime.graph.find_nodes(NodeKind.ENTITY_TYPE, domain=domain)
    if not nodes:
        return None
    return Ontology(
        domain=domain,
        entities=[
            EntityType(
                name=n.key,
                description=n.description,
                keywords=list(n.attributes.get("keywords", [])),
                attributes=list(n.attributes.get("attributes", [])),
                sensitivity=str(n.attributes.get("sensitivity", "internal")),
            )
            for n in nodes
        ],
    )


# ---------------------------------------------------------------------------
# audit / doctor / bridge
# ---------------------------------------------------------------------------


def cmd_audit_tail(args: argparse.Namespace) -> None:
    from alm.governance.audit import AuditLog

    runtime = _runtime()
    log = AuditLog(runtime.tenant_id)
    records = (
        log.for_session(args.session)
        if args.session
        else list(reversed(log.query(decision=args.decision or "", limit=args.limit)))
    )
    if emit([r.to_dict() for r in records], as_json=args.json):
        return
    if not records:
        print(dim("no audit records"))
        return
    heading("IBAC decisions")
    for record in records:
        line = record.line()
        print("  " + (green(line) if record.decision == "allow" else red(line)))


def cmd_audit_summary(args: argparse.Namespace) -> None:
    from alm.governance.audit import AuditLog

    runtime = _runtime()
    summary = AuditLog(runtime.tenant_id).summary(hours=args.hours)
    if emit(summary, as_json=args.json):
        return
    heading(f"Audit summary · last {summary['window_hours']}h")
    print(f"  decisions  {summary['total']}")
    print(f"  allowed    {green(str(summary['allowed']))}")
    print(f"  denied     {red(str(summary['denied']))}")
    print(f"  deny rate  {summary['deny_rate']:.1%}")


def cmd_audit_export(args: argparse.Namespace) -> None:
    from alm.governance.audit import AuditLog

    runtime = _runtime()
    for line in AuditLog(runtime.tenant_id).export_jsonl(limit=args.limit):
        print(line)


def cmd_doctor(args: argparse.Namespace) -> None:
    async def run() -> None:
        runtime = _runtime()
        health = runtime.health()
        model_health = await runtime.models.health()
        await runtime.close()

        payload = {**health, "model_reachability": model_health}
        if emit(payload, as_json=args.json):
            return

        heading("ALM health")
        print(f"  version        {__version__}")
        print(f"  tenant         {health['tenant']}")
        print(f"  experts        {health['experts']}")
        print(f"  domains        {', '.join(health['graph']['domains']) or '-'}")
        print(f"  graph          {health['graph']['nodes']} nodes / {health['graph']['edges']} edges")
        print(f"  corpus         {health['corpus']['chunks']} chunks")
        print(f"  embedder       {health['embedder']}")
        print(f"  policies       {health['policies']}")

        heading("Model reachability")
        if not model_health:
            print(dim("  no models registered"))
        for model_id, reachable in sorted(model_health.items()):
            mark = green("✓ reachable") if reachable else red("✗ unreachable")
            print(f"  {model_id:<24}{mark}")

        problems: list[str] = []
        if not health["orchestrator_configured"]:
            problems.append(
                "no orchestrator model — planner decomposition, the LLM judge and "
                "the fallback path are unavailable"
            )
        if not health["semantic_routing"]:
            problems.append("no embedder — routing and retrieval are lexical only")
        if health["uncalibrated_experts"]:
            problems.append(
                "uncalibrated experts (run `alm eval calibrate`): "
                + ", ".join(health["uncalibrated_experts"])
            )
        if health["experts_on_placeholder_backends"]:
            problems.append(
                "experts bound to placeholder backends, not language models: "
                + ", ".join(health["experts_on_placeholder_backends"])
            )

        heading("Findings")
        if not problems:
            print(f"  {green('✓')} nothing to report")
        for problem in problems:
            print(f"  {yellow('!')} {problem}")

    asyncio.run(run())


def cmd_bridge_serve(args: argparse.Namespace) -> None:
    async def run() -> None:
        from alm.bridge.lip_agent import LIPBridge

        runtime = _runtime()
        bridge = LIPBridge(
            runtime,
            coordinator_uri=args.coordinator,
            agent_id=args.agent_id,
        )
        registration = bridge.registration_payload()
        print(f"{green('▸')} publishing {bold(args.agent_id)} on {cyan(args.coordinator)}")
        for capability in registration["capabilities"]:
            print(f"    {dim('·')} {capability['capability_id']}")
        try:
            await bridge.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            await runtime.close()

    asyncio.run(run())


def cmd_config_show(args: argparse.Namespace) -> None:
    from alm.config import get_settings

    settings = get_settings().redacted()
    if emit(settings, as_json=args.json):
        return
    heading("Resolved configuration")
    for key, value in sorted(settings.items()):
        print(f"  {key:<32}{value}")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:  # noqa: C901 - a flat parser is clearer
    parser = argparse.ArgumentParser(
        prog="alm",
        description="ALM — a governed federation of domain models.",
    )
    parser.add_argument("--version", action="version", version=f"alm {__version__}")
    parser.add_argument("--log-level", default="", help="DEBUG, INFO, WARNING, ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_json(target: argparse.ArgumentParser) -> None:
        target.add_argument("--json", action="store_true", help="machine-readable output")

    # init / install / serve / doctor / config
    p = sub.add_parser("init", help="Create or migrate the database")
    add_json(p)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("install", help="Interactive setup wizard")
    p.add_argument("--env", default=".env")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("serve", help="Start the REST + WebSocket API")
    p.add_argument("--host", default="")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("doctor", help="Health, inventory and degradation report")
    add_json(p)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("config", help="Show resolved configuration")
    add_json(p)
    p.set_defaults(func=cmd_config_show)

    # ask / explain
    p = sub.add_parser("ask", help="Run one intent through the federation")
    p.add_argument("intent")
    p.add_argument("--pack", default="", help="Install and use this pack first")
    p.add_argument("--principal", default="", help="Identity IBAC evaluates against")
    p.add_argument("--claim", action="append", help="Grant a claim (repeatable)")
    p.add_argument("--trace", action="store_true", help="Print the layer-by-layer trail")
    p.add_argument("--follow", action="store_true", help="Stream events as they happen")
    p.add_argument("--verbose", action="store_true", help="Include trace detail")
    add_json(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("explain", help="Show routing signals without executing")
    p.add_argument("intent")
    p.add_argument("--pack", default="")
    p.add_argument("--top-k", type=int, default=5)
    add_json(p)
    p.set_defaults(func=cmd_explain)

    # pack
    pack = sub.add_parser("pack", help="Manage domain packs").add_subparsers(
        dest="pack_command", required=True
    )
    p = pack.add_parser("list", help="List discoverable packs")
    p.add_argument("--dir", default="")
    add_json(p)
    p.set_defaults(func=cmd_pack_list)

    p = pack.add_parser("validate", help="Check a pack before installing it")
    p.add_argument("path")
    add_json(p)
    p.set_defaults(func=cmd_pack_validate)

    p = pack.add_parser("install", help="Install a pack into the federation")
    p.add_argument("path")
    p.add_argument("--skip-corpus", action="store_true")
    add_json(p)
    p.set_defaults(func=cmd_pack_install)

    # graph
    graph = sub.add_parser("graph", help="Inspect the Context Graph").add_subparsers(
        dest="graph_command", required=True
    )
    p = graph.add_parser("show", help="List nodes")
    p.add_argument("--kind", default="")
    p.add_argument("--domain", default="")
    add_json(p)
    p.set_defaults(func=cmd_graph_show)

    p = graph.add_parser("stats", help="Graph statistics")
    add_json(p)
    p.set_defaults(func=cmd_graph_stats)

    # expert
    expert = sub.add_parser("expert", help="Inspect Expert Agents").add_subparsers(
        dest="expert_command", required=True
    )
    p = expert.add_parser("list", help="List installed experts")
    add_json(p)
    p.set_defaults(func=cmd_expert_list)

    p = expert.add_parser("show", help="Show one expert in detail")
    p.add_argument("expert_id")
    add_json(p)
    p.set_defaults(func=cmd_expert_show)

    p = expert.add_parser(
        "promote-check",
        help="Evaluate the business triggers for giving this expert its own model",
    )
    p.add_argument("expert_id")
    p.add_argument("--sovereignty", action="store_true", help="Data must not leave the perimeter")
    p.add_argument("--calls", type=int, default=None, help="Monthly call volume")
    p.add_argument("--volume-threshold", type=int, default=50_000)
    p.add_argument("--latency-requirement", type=float, default=None, help="Required p95 in ms")
    p.add_argument("--observed-p95", type=float, default=None)
    p.add_argument("--rag-accuracy", type=float, default=None)
    p.add_argument("--accuracy-target", type=float, default=None)
    add_json(p)
    p.set_defaults(func=cmd_expert_promote_check)

    # model
    model = sub.add_parser("model", help="Manage the model registry").add_subparsers(
        dest="model_command", required=True
    )
    p = model.add_parser("list", help="List models and the serving topology")
    add_json(p)
    p.set_defaults(func=cmd_model_list)

    p = model.add_parser("add", help="Register a model")
    p.add_argument("model_id")
    p.add_argument("--tier", default="slm", help="micro_slm | slm | small | orchestrator")
    p.add_argument("--backend", default="heuristic")
    p.add_argument("--model", default="", help="Name the backend expects")
    p.add_argument("--base-model", default="", help="Shared base for a LoRA adapter")
    p.add_argument("--adapter", default="", help="Adapter name served over the base")
    p.add_argument("--adapter-uri", default="")
    p.add_argument("--endpoint", default="")
    p.add_argument("--api-key", default="")
    p.add_argument("--param", action="append", help="key=value (repeatable)")
    p.add_argument("--context-window", type=int, default=8192)
    p.add_argument("--cost-in", type=float, default=0.0, help="USD per 1k input tokens")
    p.add_argument("--cost-out", type=float, default=0.0, help="USD per 1k output tokens")
    p.add_argument("--description", default="")
    add_json(p)
    p.set_defaults(func=cmd_model_add)

    p = model.add_parser("remove", help="Remove a model")
    p.add_argument("model_id")
    p.set_defaults(func=cmd_model_remove)

    p = model.add_parser("cost", help="Convert a GPU-hour rate into cost per 1k tokens")
    p.add_argument("--gpu-hourly", type=float, required=True)
    p.add_argument("--tokens-per-second", type=float, required=True)
    p.add_argument("--utilisation", type=float, default=0.7)
    add_json(p)
    p.set_defaults(func=cmd_model_cost)

    p = model.add_parser("versions", help="Version history for a model")
    p.add_argument("model_id")
    add_json(p)
    p.set_defaults(func=cmd_model_versions)

    p = model.add_parser("promote", help="Promote a version (gated on evaluation)")
    p.add_argument("model_id")
    p.add_argument("version")
    p.add_argument("--metric", default="domain_accuracy")
    p.add_argument("--force", action="store_true", help="Bypass the evaluation gate")
    add_json(p)
    p.set_defaults(func=cmd_model_promote)

    p = model.add_parser("rollback", help="Reactivate a previous version")
    p.add_argument("model_id")
    p.add_argument("version", nargs="?", default="")
    add_json(p)
    p.set_defaults(func=cmd_model_rollback)

    # cmrag
    cmrag = sub.add_parser("cmrag", help="Corpus and retrieval").add_subparsers(
        dest="cmrag_command", required=True
    )
    p = cmrag.add_parser("index", help="Index a corpus directory")
    p.add_argument("path")
    p.add_argument("--domain", default="", help="Infer from the path when omitted")
    add_json(p)
    p.set_defaults(func=cmd_cmrag_index)

    p = cmrag.add_parser("search", help="Query the corpus")
    p.add_argument("query")
    p.add_argument("--domain", action="append")
    p.add_argument("--expert", default="", help="Apply this expert's IBAC filter")
    p.add_argument("--top-k", type=int, default=6)
    add_json(p)
    p.set_defaults(func=cmd_cmrag_search)

    p = cmrag.add_parser("stats", help="Corpus statistics")
    add_json(p)
    p.set_defaults(func=cmd_cmrag_stats)

    p = cmrag.add_parser("reindex", help="Recompute embeddings")
    p.add_argument("--domain", default="")
    add_json(p)
    p.set_defaults(func=cmd_cmrag_reindex)

    # eval
    evaluation = sub.add_parser("eval", help="The controlled experiment").add_subparsers(
        dest="eval_command", required=True
    )
    p = evaluation.add_parser("run", help="Run the federation, optionally against the baseline")
    p.add_argument("--pack", default="")
    p.add_argument("--dataset", default="")
    p.add_argument("--domain", default="")
    p.add_argument("--compare", action="store_true", help="Also run the monolithic baseline")
    p.add_argument("--repeats", type=int, default=1)
    add_json(p)
    p.set_defaults(func=cmd_eval_run)

    p = evaluation.add_parser("calibrate", help="Fit per-expert confidence calibration")
    p.add_argument("--pack", default="")
    p.add_argument("--dataset", default="")
    p.add_argument("--domain", default="")
    add_json(p)
    p.set_defaults(func=cmd_eval_calibrate)

    p = evaluation.add_parser("list", help="Recent evaluation runs")
    p.add_argument("--limit", type=int, default=20)
    add_json(p)
    p.set_defaults(func=cmd_eval_list)

    # distill
    distill = sub.add_parser("distill", help="Teacher→student pipeline").add_subparsers(
        dest="distill_command", required=True
    )
    for name, func, helptext in (
        ("seed", cmd_distill_seed, "Collect source passages from the domain corpus"),
        ("generate", cmd_distill_generate, "Have the teacher write training examples"),
        ("filter", cmd_distill_filter, "Validate, deduplicate and score examples"),
        ("export", cmd_distill_export, "Write the train/eval JSONL split"),
        ("feedback", cmd_distill_feedback, "Ingest production failures and escalations"),
        ("stats", cmd_distill_stats, "Dataset statistics"),
    ):
        p = distill.add_parser(name, help=helptext)
        p.add_argument("--domain", default="", required=name not in {"feedback", "stats"})
        p.add_argument("--dataset", default="default", help="Named training dataset")
        if name == "seed":
            p.add_argument("--limit", type=int, default=200)
        if name == "generate":
            p.add_argument("--limit", type=int, default=50)
            p.add_argument("--per-passage", type=int, default=3)
        if name == "export":
            p.add_argument("--out", required=True)
            p.add_argument("--eval-ratio", type=float, default=0.2)
        add_json(p)
        p.set_defaults(func=func)

    # audit
    audit = sub.add_parser("audit", help="IBAC decision log").add_subparsers(
        dest="audit_command", required=True
    )
    p = audit.add_parser("tail", help="Recent decisions")
    p.add_argument("--session", default="")
    p.add_argument("--decision", default="", help="allow | deny")
    p.add_argument("--limit", type=int, default=50)
    add_json(p)
    p.set_defaults(func=cmd_audit_tail)

    p = audit.add_parser("summary", help="Aggregate decisions")
    p.add_argument("--hours", type=int, default=24)
    add_json(p)
    p.set_defaults(func=cmd_audit_summary)

    p = audit.add_parser("export", help="Stream the log as JSON Lines")
    p.add_argument("--limit", type=int, default=10_000)
    p.set_defaults(func=cmd_audit_export)

    # bridge
    bridge = sub.add_parser("bridge", help="Liquid Interface Protocol bridge").add_subparsers(
        dest="bridge_command", required=True
    )
    p = bridge.add_parser("serve", help="Publish the federation on an Agentic Bus")
    p.add_argument("--coordinator", default=os.getenv("ALM_LIP_COORDINATOR_URI", "ws://localhost:8765"))
    p.add_argument("--agent-id", default=os.getenv("ALM_LIP_AGENT_ID", "alm-federation"))
    p.set_defaults(func=cmd_bridge_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level or None)

    try:
        args.func(args)
    except ALMError as exc:
        fail(exc.message)
    except KeyboardInterrupt:
        print()
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
