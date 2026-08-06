# Deployment

## Local (SQLite, Ollama)

```bash
pip install -e .
alm install                # writes .env
alm init
ollama pull qwen2.5:0.5b && ollama pull qwen2.5:3b && ollama pull nomic-embed-text

alm model add router     --tier micro_slm    --backend ollama --model qwen2.5:0.5b
alm model add expert     --tier slm          --backend ollama --model qwen2.5:3b
alm model add teacher    --tier orchestrator --backend openai --model gpt-4o

alm pack install packs/demo-enterprise
alm serve
```

Nothing leaves the machine except calls to the orchestrator tier — and that tier can be self-hosted
too, which is the point when data residency is a veto rather than a preference.

## Docker

```bash
docker compose up -d              # API + PostgreSQL
docker compose --profile ollama up -d   # …and a local Ollama
```

## PostgreSQL

```bash
pip install "alm-federation[postgres]"
export ALM_DATABASE_URL=postgresql+psycopg://alm:alm@localhost:5432/alm
alm init
```

CMRAG stores vectors as JSON and scores with NumPy, which needs no extension and is fast enough well
past the size of a typical domain corpus (a full scan of 100k × 384-dim vectors is a few
milliseconds), because the scan is always pre-filtered by domain and tenant.

For larger corpora, add a pgvector column and index:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE chunks ADD COLUMN embedding_vec vector(768);
CREATE INDEX ON chunks USING hnsw (embedding_vec vector_cosine_ops);
```

## vLLM with multi-adapter serving

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --enable-lora \
  --lora-modules legal=/adapters/legal-v3 finance=/adapters/finance-v1 \
  --max-loras 8 --port 8000
```

```bash
alm model add legal-expert   --tier slm --backend vllm \
  --base-model meta-llama/Llama-3.2-3B-Instruct --adapter legal
alm model add finance-expert --tier slm --backend vllm \
  --base-model meta-llama/Llama-3.2-3B-Instruct --adapter finance

alm model list      # shared bases: 1 · adapters: 2 · full loads saved: 1
```

## Operations

```bash
alm doctor                        # inventory, reachability, degradation
alm audit summary --hours 24
alm eval run --pack ... --compare # re-run before every promotion
```

Health and `alm doctor` report **degradation explicitly**: a missing orchestrator (no planner, no
judge, no fallback), a missing embedder (lexical-only routing), uncalibrated experts, and any expert
bound to a placeholder backend.

## Configuration

Every knob is an `ALM_*` environment variable; see [`.env.example`](../.env.example). The ones that
change behaviour most:

| Variable | Effect |
|---|---|
| `ALM_ROUTER_CONFIDENCE_THRESHOLD` | Below this, escalate instead of guessing |
| `ALM_FALLBACK_ENABLED` | Turning it off removes the never-worse-than-baseline guarantee |
| `ALM_MAX_PARALLELISM` | Caps concurrent expert calls so a wide plan cannot saturate a shared GPU |
| `ALM_ARBITRATION_STRATEGIES` | Strategy chain order |
| `ALM_GOVERNANCE_ENABLED` | Disabling it allows every context read |
| `ALM_API_TOKEN` | Required bearer token; unset means the API is unauthenticated |
