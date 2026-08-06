"""Teacher→student distillation.

The real bottleneck of ALM is not training — it is having quality domain data.
The fastest path uses the frontier model as a teacher and distils its knowledge
into the small model, in five stages:

1. **Seed with the real thing** — start from the domain's own documents,
   anchored to the ontology. Synthetic data unmoored from the corpus teaches a
   domain that does not exist.
2. **Generate with the teacher** — question/answer pairs and reasoning chains
   over that material, covering the cases the agent will actually see.
3. **Filter and validate** — drop incorrect examples by automatic checks and
   sampled human review. Bad data teaches errors *persistently*, which is why
   this stage is not optional.
4. **Train the student** — fine-tune against a held-out evaluation set the
   model has never seen.
5. **Close the loop** — capture production failures and escalations and feed
   them back. This is where the federation improves with use.

Stage 4 lives in :mod:`alm.distillation.training`; the rest is here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select

from alm.cmrag.store import ChunkStore
from alm.core.errors import DistillationError
from alm.core.jsonutil import extract_json, truncate
from alm.core.telemetry import UsageMeter
from alm.graph.ontology import Ontology
from alm.models.serving import ServingRouter
from alm.models.spec import GenerationRequest, Message
from alm.models.tiers import ModelTier
from alm.persistence.database import session_scope
from alm.persistence.models import FeedbackRow, TrainingExampleRow

logger = logging.getLogger(__name__)

_TEACHER_SYSTEM = """\
You are generating training data for a small domain model in the {domain} domain.

From the supplied source passage, write {count} question/answer pairs that a
practitioner would genuinely ask, together with the reasoning that connects the
passage to each answer.

Rules:
- Every answer must be fully supported by the passage. Invent nothing.
- Use the domain's own vocabulary: {vocabulary}
- Reference the ontology entities involved: {entities}
- Vary difficulty: some direct lookups, some requiring you to combine two facts.
- If the passage cannot support {count} good pairs, return fewer. Quality over count.

Return ONLY a JSON array:
[{{"question": "...", "answer": "...", "reasoning": "...", "entities": ["..."]}}]
"""


class TrainingExample(BaseModel):
    """One distilled example."""

    prompt: str
    completion: str
    reasoning: str = ""
    domain: str = ""
    expert_id: str = ""
    entities: list[str] = Field(default_factory=list)
    source: str = "teacher"
    source_document_id: str = ""
    quality_score: float = 0.0
    status: str = "pending"
    reject_reason: str = ""
    split: str = "train"

    def content_hash(self) -> str:
        payload = f"{self.prompt.strip().lower()}|{self.completion.strip().lower()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class DistillationPipeline:
    """Seeds, generates, filters and exports domain training data."""

    def __init__(
        self,
        *,
        serving: ServingRouter,
        chunk_store: ChunkStore,
        tenant_id: str = "default",
        dataset: str = "default",
    ) -> None:
        self.serving = serving
        self.chunk_store = chunk_store
        self.tenant_id = tenant_id
        self.dataset = dataset

    # -- stage 1 · seed ----------------------------------------------------

    def seed(self, domain: str, *, limit: int = 200) -> list[dict[str, Any]]:
        """Collect source passages from the domain corpus.

        Seeding from the real corpus rather than from a prompt is what keeps the
        student anchored in the client's actual documents — which is where the
        defensibility of a per-client model comes from.
        """
        chunks = self.chunk_store.chunks_for_domain(domain)
        if not chunks:
            raise DistillationError(
                f"no indexed corpus for domain {domain!r}; run `alm cmrag index` first",
                domain=domain,
            )
        seeds = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "entities": list(c.entities or []),
            }
            for c in chunks
            if len(c.text.split()) >= 30
        ]
        logger.info("Seeded %d passage(s) from domain %s", len(seeds), domain)
        return seeds[:limit]

    # -- stage 2 · generate ------------------------------------------------

    async def generate(
        self,
        domain: str,
        *,
        ontology: Ontology | None = None,
        expert_id: str = "",
        per_passage: int = 3,
        limit: int = 50,
        meter: UsageMeter | None = None,
    ) -> list[TrainingExample]:
        """Have the teacher write examples over the seeded passages."""
        seeds = self.seed(domain, limit=limit)
        vocabulary = ", ".join(ontology.vocabulary()[:40]) if ontology else "the domain's own terms"
        entity_names = ", ".join(ontology.entity_names()) if ontology else "—"

        examples: list[TrainingExample] = []
        for seed in seeds:
            request = GenerationRequest(
                messages=[
                    Message(
                        role="system",
                        content=_TEACHER_SYSTEM.format(
                            domain=domain,
                            count=per_passage,
                            vocabulary=vocabulary,
                            entities=entity_names,
                        ),
                    ),
                    Message(role="user", content=truncate(seed["text"], 4000)),
                ],
                json_mode=True,
                temperature=0.3,
                max_tokens=1500,
                metadata={"question": "generate training pairs", "max_sentences": 4},
            )
            try:
                result = await self.serving.generate(
                    request,
                    tier=ModelTier.ORCHESTRATOR,
                    meter=meter,
                    component="distill.teacher",
                    allow_escalation=False,
                )
            except Exception as exc:
                logger.warning("Teacher generation failed for a passage: %s", exc)
                continue

            parsed = extract_json(result.text)
            if not isinstance(parsed, list):
                continue
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                question = str(item.get("question", "")).strip()
                answer = str(item.get("answer", "")).strip()
                if not question or not answer:
                    continue
                examples.append(
                    TrainingExample(
                        prompt=question,
                        completion=answer,
                        reasoning=str(item.get("reasoning", "")),
                        domain=domain,
                        expert_id=expert_id,
                        entities=[str(e) for e in item.get("entities", [])]
                        or list(seed["entities"]),
                        source_document_id=seed["document_id"],
                    )
                )

        logger.info("Teacher produced %d raw example(s) for %s", len(examples), domain)
        self._persist(examples)
        return examples

    # -- stage 3 · filter --------------------------------------------------

    def filter(
        self,
        domain: str,
        *,
        ontology: Ontology | None = None,
        min_answer_words: int = 4,
        max_answer_words: int = 400,
    ) -> dict[str, Any]:
        """Validate, deduplicate and score the pending examples.

        Every rejection is recorded with its reason rather than deleted, so the
        yield of a distillation run is inspectable — a teacher that produces 80%
        rejects is telling you something about the corpus.
        """
        vocabulary = set(ontology.vocabulary()) if ontology else set()
        seen: set[str] = set()
        accepted = 0
        reviewed = 0
        rejections: dict[str, int] = {}

        # Query and mutate inside one session: rows loaded through a closed
        # session are detached, and writing to them would be silently discarded.
        with session_scope() as session:
            stmt = select(TrainingExampleRow).where(
                TrainingExampleRow.dataset == self.dataset,
                TrainingExampleRow.tenant_id == self.tenant_id,
                TrainingExampleRow.status == "pending",
            )
            if domain:
                stmt = stmt.where(TrainingExampleRow.domain == domain)
            pending = list(session.execute(stmt).scalars().all())
            reviewed = len(pending)

            for row in pending:
                reason = ""
                words = len(row.completion.split())

                if row.content_hash in seen:
                    reason = "duplicate"
                elif words < min_answer_words:
                    reason = "answer too short to be informative"
                elif words > max_answer_words:
                    reason = "answer too long; likely rambling or off-task"
                elif row.prompt.strip().lower() == row.completion.strip().lower():
                    reason = "answer restates the question"
                elif _looks_like_refusal(row.completion):
                    reason = "teacher refused or hedged instead of answering"

                seen.add(row.content_hash)

                if reason:
                    row.status = "rejected"
                    row.reject_reason = reason
                    rejections[reason] = rejections.get(reason, 0) + 1
                    continue

                row.quality_score = _quality_score(row, vocabulary)
                row.status = "accepted"
                accepted += 1

        if not reviewed:
            return {"domain": domain, "reviewed": 0, "accepted": 0, "rejected": 0}

        return {
            "domain": domain,
            "reviewed": reviewed,
            "accepted": accepted,
            "rejected": reviewed - accepted,
            "rejection_reasons": rejections,
            "acceptance_rate": round(accepted / reviewed, 4),
        }

    # -- stage 4 · export --------------------------------------------------

    def export(
        self,
        domain: str,
        out_path: str | Path,
        *,
        eval_ratio: float = 0.2,
        seed: int = 17,
    ) -> dict[str, Any]:
        """Write accepted examples as JSONL, split into train and eval.

        The split is deterministic and content-hash based, so re-exporting after
        adding examples keeps previously-eval cases in eval. A case that drifts
        into the training set between runs silently invalidates every comparison
        made against it.
        """
        rows = self._load(domain, status="accepted")
        if not rows:
            raise DistillationError(
                f"no accepted examples for domain {domain!r}; run `alm distill filter` first",
                domain=domain,
            )

        train_path = Path(out_path)
        eval_path = train_path.with_name(f"{train_path.stem}.eval{train_path.suffix}")
        train_path.parent.mkdir(parents=True, exist_ok=True)

        train_count = 0
        eval_count = 0
        with train_path.open("w", encoding="utf-8") as train_file, eval_path.open(
            "w", encoding="utf-8"
        ) as eval_file:
            for row in rows:
                bucket = int(row.content_hash[:8], 16) % 1000
                is_eval = bucket < int(eval_ratio * 1000)
                record = {
                    "messages": [
                        {"role": "user", "content": row.prompt},
                        {"role": "assistant", "content": row.completion},
                    ],
                    "domain": row.domain,
                    "entities": list(row.entities or []),
                    "reasoning": row.reasoning,
                }
                line = json.dumps(record, ensure_ascii=False) + "\n"
                if is_eval:
                    eval_file.write(line)
                    eval_count += 1
                else:
                    train_file.write(line)
                    train_count += 1

        with session_scope() as session:
            for row in rows:
                live = session.get(TrainingExampleRow, row.id)
                if live is not None:
                    bucket = int(live.content_hash[:8], 16) % 1000
                    live.split = "eval" if bucket < int(eval_ratio * 1000) else "train"

        logger.info(
            "Exported %d train / %d eval example(s) for %s", train_count, eval_count, domain
        )
        return {
            "domain": domain,
            "train_path": str(train_path),
            "eval_path": str(eval_path),
            "train": train_count,
            "eval": eval_count,
        }

    # -- stage 5 · feedback loop -------------------------------------------

    def ingest_feedback(self, domain: str = "", *, limit: int = 500) -> dict[str, Any]:
        """Turn captured production failures and escalations into seed material.

        Every case an expert got wrong or handed to the LLM is exactly what the
        teacher should be generating training data about next. This is the loop
        that makes the federation improve with use rather than decay.
        """
        with session_scope() as session:
            stmt = select(FeedbackRow).where(
                FeedbackRow.tenant_id == self.tenant_id,
                FeedbackRow.consumed.is_(False),
            )
            if domain:
                stmt = stmt.where(FeedbackRow.domain == domain)
            rows = session.execute(stmt.limit(limit)).scalars().all()

            examples: list[TrainingExample] = []
            for row in rows:
                if not row.intent_text:
                    continue
                examples.append(
                    TrainingExample(
                        prompt=row.intent_text,
                        completion=row.expected or "",
                        reasoning=f"captured from production ({row.kind})",
                        domain=row.domain or domain,
                        expert_id=row.expert_id,
                        source="production",
                        # No verified answer yet: these are *questions to cover*,
                        # not answers to imitate. They stay pending until the
                        # teacher answers them or a human supplies the label.
                        status="needs_answer" if not row.expected else "pending",
                    )
                )
                row.consumed = True

        self._persist(examples)
        by_kind: dict[str, int] = {}
        for row in rows:
            by_kind[row.kind] = by_kind.get(row.kind, 0) + 1
        return {
            "domain": domain or "all",
            "captured": len(examples),
            "by_kind": by_kind,
        }

    # -- storage -----------------------------------------------------------

    def _persist(self, examples: Sequence[TrainingExample]) -> int:
        if not examples:
            return 0
        with session_scope() as session:
            existing = {
                h
                for (h,) in session.execute(
                    select(TrainingExampleRow.content_hash).where(
                        TrainingExampleRow.dataset == self.dataset
                    )
                ).all()
            }
            written = 0
            for example in examples:
                digest = example.content_hash()
                if digest in existing:
                    continue
                existing.add(digest)
                session.add(
                    TrainingExampleRow(
                        dataset=self.dataset,
                        tenant_id=self.tenant_id,
                        domain=example.domain,
                        expert_id=example.expert_id,
                        prompt=example.prompt,
                        completion=example.completion,
                        reasoning=example.reasoning,
                        entities=example.entities,
                        source=example.source,
                        source_document_id=example.source_document_id,
                        split=example.split,
                        status=example.status,
                        quality_score=example.quality_score,
                        content_hash=digest,
                    )
                )
                written += 1
        return written

    def _load(self, domain: str, *, status: str = "") -> list[TrainingExampleRow]:
        with session_scope() as session:
            stmt = select(TrainingExampleRow).where(
                TrainingExampleRow.dataset == self.dataset,
                TrainingExampleRow.tenant_id == self.tenant_id,
            )
            if domain:
                stmt = stmt.where(TrainingExampleRow.domain == domain)
            if status:
                stmt = stmt.where(TrainingExampleRow.status == status)
            return list(session.execute(stmt).scalars().all())

    def stats(self, domain: str = "") -> dict[str, Any]:
        rows = self._load(domain)
        by_status: dict[str, int] = {}
        by_domain: dict[str, int] = {}
        for row in rows:
            by_status[row.status] = by_status.get(row.status, 0) + 1
            by_domain[row.domain] = by_domain.get(row.domain, 0) + 1
        return {
            "dataset": self.dataset,
            "total": len(rows),
            "by_status": by_status,
            "by_domain": by_domain,
        }


_REFUSAL_MARKERS = (
    "i cannot", "i can't", "as an ai", "i am unable", "não posso",
    "i do not have enough", "insufficient information",
)


def _looks_like_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _quality_score(row: TrainingExampleRow, vocabulary: set[str]) -> float:
    """Heuristic quality signal used to rank and sample for human review."""
    score = 0.5
    if row.reasoning and len(row.reasoning.split()) > 10:
        score += 0.2
    if row.entities:
        score += 0.15
    if vocabulary:
        words = {w.lower().strip(".,;:") for w in row.completion.split()}
        if words & vocabulary:
            score += 0.15
    return round(min(score, 1.0), 4)
