"""Training data for a decision model — labelled typed questions.

The bottleneck of a decision model is not the training loop, it is labelled
decisions.  This module assembles them from two kinds of source and writes one
file a trainer can read:

* **Public datasets** whose human labels already *are* decisions — entailment,
  sentiment, intent — recast as choice / score / yes-no questions.  Each source
  is pinned to a revision and carries its licence.
* **Rule-based generators** (:mod:`alm.decision.synth_pt`) whose labels are
  computed from the facts in the state.

No model labels anything here, and no output of a hosted decision model is
ever used as a label.

A record is the ``/v1/systemone`` request shape with a ``label`` on every
question — the option name for a choice, the level index for a score,
``true``/``false`` for a yes/no::

    {"state": "...", "questions": {"fila": {"type": "choice", "instructions": "...",
                                            "criteria": {"financeiro": "...", "outro": null},
                                            "label": "financeiro"}},
     "_meta": {"source": "massive_intent_pt", "lang": "pt", "licence": "cc-by-4.0", ...}}

This is also the labelled-request shape the open decision-model trainers read
(Kev's ``load_records``), so the file feeds them unchanged.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import random
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from alm.core.errors import ALMError
from alm.decision import synth_pt

logger = logging.getLogger(__name__)

Record = dict[str, Any]
Converter = Callable[[list[dict[str, Any]], random.Random], list[Record]]

#: Licence classes, from least to most restrictive for commercial training.
PERMISSIVE, SHARE_ALIKE, UNSTATED = "permissive", "share-alike", "unstated"


class DatasetError(ALMError):
    """A source could not be fetched or converted."""


# -- wording ------------------------------------------------------------------


def _vary(options: list[str], rng: random.Random) -> str:
    return rng.choice(options)


def _wrap_state(text: str, rng: random.Random) -> Any:
    """Mostly plain text; sometimes structured, as real callers send it."""
    roll = rng.random()
    if roll < 0.12:
        return {"documento": text}
    if roll < 0.20:
        return {"mensagem": {"canal": rng.choice(["e-mail", "chat", "formulário"]), "texto": text}}
    return text


# -- converters ---------------------------------------------------------------


def _boolq(rows: list[dict[str, Any]], rng: random.Random) -> list[Record]:
    out = []
    for row in rows:
        question: dict[str, Any] = {
            "type": "noul",
            "instructions": row["question"].strip().rstrip("?") + "?",
            "label": bool(row["label"]),
        }
        if rng.random() < 0.4:
            question["criteria"] = {
                "true": "O texto sustenta a resposta sim",
                "false": "O texto sustenta a resposta não, ou não diz",
            }
        out.append({"state": _wrap_state(row["passage"], rng), "questions": {"resposta": question}})
    return out


_ENTAILMENT = [
    "A afirmação a seguir decorre do texto? Afirmação: “{h}”",
    "Com base apenas no texto, é correto afirmar que: “{h}”?",
    "O texto permite concluir que “{h}”?",
]


def _entailment(premise: str, hypothesis: str, entailed: bool, rng: random.Random) -> Record:
    return {
        "state": _wrap_state(premise, rng),
        "questions": {
            "decorre": {
                "type": "noul",
                "instructions": _vary(_ENTAILMENT, rng).format(h=hypothesis.strip()),
                "label": entailed,
            }
        },
    }


def _rte(rows: list[dict[str, Any]], rng: random.Random) -> list[Record]:
    # GLUE RTE: 0 = entailment, 1 = not entailment.
    return [_entailment(r["premise"], r["hypothesis"], r["label"] == 0, rng) for r in rows]


def _wnli(rows: list[dict[str, Any]], rng: random.Random) -> list[Record]:
    # GLUE WNLI: 1 = entailment.
    return [_entailment(r["sentence1"], r["sentence2"], r["label"] == 1, rng) for r in rows]


_ANSWERS = [
    "O trecho contém a resposta para a pergunta: “{q}”",
    "Este trecho responde à pergunta “{q}”?",
]


def _qnli(rows: list[dict[str, Any]], rng: random.Random) -> list[Record]:
    # GLUE QNLI: 0 = the sentence answers the question.
    return [
        {
            "state": _wrap_state(row["sentence"], rng),
            "questions": {
                "responde": {
                    "type": "noul",
                    "instructions": _vary(_ANSWERS, rng).format(q=row["question"].strip()),
                    "label": row["label"] == 0,
                }
            },
        }
        for row in rows
    ]


def _sst2(rows: list[dict[str, Any]], rng: random.Random) -> list[Record]:
    out = []
    for row in rows:
        positive = row["label"] == 1
        if rng.random() < 0.5:
            question = {
                "type": "choice",
                "instructions": "Qual é o sentimento deste trecho de crítica?",
                "criteria": {"negativo": "Avaliação desfavorável", "positivo": "Avaliação favorável"},
                "label": "positivo" if positive else "negativo",
            }
        else:
            negative_form = rng.random() < 0.5
            question = {
                "type": "noul",
                "instructions": "O trecho expressa uma opinião "
                + ("negativa?" if negative_form else "positiva?"),
                "label": positive != negative_form,
            }
        out.append({"state": _wrap_state(row["sentence"], rng), "questions": {"sentimento": question}})
    return out


def _copa(rows: list[dict[str, Any]], rng: random.Random) -> list[Record]:
    out = []
    for row in rows:
        ask = "causa" if row["question"] == "cause" else "consequência"
        out.append(
            {
                "state": row["premise"],
                "questions": {
                    "alternativa": {
                        "type": "choice",
                        "instructions": f"Qual alternativa é a {ask} mais plausível do que o texto descreve?",
                        "criteria": {"primeira": row["choice1"], "segunda": row["choice2"]},
                        "label": "primeira" if row["label"] == 0 else "segunda",
                    }
                },
            }
        )
    return out


_INTENT = [
    "Qual intenção melhor descreve o pedido do usuário?",
    "O que o usuário quer que o assistente faça?",
    "Classifique o pedido em uma das intenções.",
]


def _massive(rows: list[dict[str, Any]], rng: random.Random) -> list[Record]:
    """Intent classification, with a random subset of the intents as options.

    Varying which and how many options appear teaches the model to read the
    option list instead of memorising a fixed label set.
    """
    intents = sorted({row["label_text"] for row in rows})
    out = []
    for row in rows:
        truth = row["label_text"]
        others = rng.sample([i for i in intents if i != truth], min(len(intents) - 1, rng.randint(2, 11)))
        options = [truth, *others]
        rng.shuffle(options)
        show = rng.random() < 0.5
        out.append(
            {
                "state": _wrap_state(row["text"], rng),
                "questions": {
                    "intencao": {
                        "type": "choice",
                        "instructions": _vary(_INTENT, rng),
                        "criteria": {o: (o.replace("_", " ") if show else None) for o in options},
                        "label": truth,
                    }
                },
            }
        )
    return out


# -- sources ------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One public dataset, pinned and licensed."""

    name: str
    repo: str
    revision: str
    path: str
    convert: Converter
    lang: str
    #: Licence of the hosting repository, as its dataset card states it.
    licence: str
    #: Licence class of the *underlying* data, which is what constrains training.
    licence_class: str
    note: str = ""

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{self.path}"


_EXTRAGLUE = ("PORTULAN/extraglue", "d3d7ee103cc4961d735ad0e4b403b5718498fd43")


def _extraglue(name: str, task: str, convert: Converter, licence_class: str, note: str) -> Source:
    return Source(
        name=name, repo=_EXTRAGLUE[0], revision=_EXTRAGLUE[1], path=f"data/{task}_pt-BR/train.jsonl",
        convert=convert, lang="pt-BR", licence="mit", licence_class=licence_class,
        note=f"Machine translation of {note}",
    )


SOURCES: dict[str, Source] = {
    s.name: s
    for s in (
        Source(
            name="massive_intent_pt", repo="mteb/amazon_massive_intent",
            revision="940fd47a81eaa7f2cc7b129674d945d618ac38c2", path="train/pt.json.gz",
            convert=_massive, lang="pt", licence="apache-2.0", licence_class=PERMISSIVE,
            note="Amazon MASSIVE (CC BY 4.0), European Portuguese",
        ),
        _extraglue("copa_pt", "copa", _copa, PERMISSIVE, "COPA (BSD 2-Clause)"),
        _extraglue("wnli_pt", "wnli", _wnli, PERMISSIVE, "WNLI (CC BY 4.0)"),
        _extraglue("boolq_pt", "boolq", _boolq, SHARE_ALIKE, "BoolQ (CC BY-SA 3.0)"),
        _extraglue("qnli_pt", "qnli", _qnli, SHARE_ALIKE, "QNLI, derived from SQuAD (CC BY-SA 4.0)"),
        _extraglue("rte_pt", "rte", _rte, UNSTATED, "RTE, whose original licence is not stated"),
        _extraglue("sst2_pt", "sst2", _sst2, UNSTATED, "SST-2, whose original licence is not stated"),
    )
}


def allowed_sources(*, only_permissive: bool = False, allow_unstated: bool = False) -> list[Source]:
    """Sources admitted by a licence policy.

    The default admits permissive and share-alike data and leaves out data whose
    original licence is not stated.  Whether share-alike terms reach a model
    trained on the data is a legal question this code cannot settle;
    ``only_permissive`` is the conservative setting.
    """
    admitted = {PERMISSIVE}
    if not only_permissive:
        admitted.add(SHARE_ALIKE)
        if allow_unstated:
            admitted.add(UNSTATED)
    return [s for s in SOURCES.values() if s.licence_class in admitted]


def fetch_rows(source: Source, cache_dir: Path, *, client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Download a source file once, at its pinned revision, and parse it."""
    target = cache_dir / source.repo.replace("/", "__") / source.revision[:12] / source.path
    if not target.exists():
        owned = client is None
        client = client or httpx.Client(timeout=300.0, follow_redirects=True)
        try:
            response = client.get(source.url)
        except httpx.HTTPError as exc:
            raise DatasetError(f"could not download {source.name} from {source.url}: {exc}") from exc
        finally:
            if owned:
                client.close()
        if response.status_code != 200:
            raise DatasetError(f"{source.name}: {source.url} returned HTTP {response.status_code}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
    raw = target.read_bytes()
    text = gzip.decompress(raw).decode("utf-8") if source.path.endswith(".gz") else raw.decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# -- augmentation -------------------------------------------------------------

#: "None of the above" must appear both as the right answer and as a wrong
#: alternative, in varied wording, or a model learns to pick the wording.
NONE_OPTIONS: list[tuple[str, str | None]] = [
    ("outro", "Nenhuma das anteriores"), ("nenhuma", "Nenhuma destas opções"), ("outro", "Outra coisa"),
    ("nao_listado", "Não está entre as opções"), ("nenhuma_das_anteriores", None), ("outro", None),
    ("sem_correspondencia", "Nenhuma opção corresponde"), ("outro", "Uma categoria que não aparece na lista"),
]
DISTRACTORS = {
    "clima": "Previsão do tempo para o fim de semana",
    "receita_de_bolo": "Uma receita de bolo de cenoura",
    "futebol": "Resultado do jogo de ontem",
}


def augment(record: Record, rng: random.Random, *, p_none: float = 0.10,
            p_none_wrong: float = 0.12, p_distractor: float = 0.12) -> Record:
    """Shuffle every choice's options; sometimes add a "none" option or a distractor.

    With ``p_none`` the true option is *removed* and a "none of the above"
    option becomes the answer; with ``p_none_wrong`` the same kind of option is
    added as a wrong alternative.  Seeing both keeps "none" from being a
    shortcut.  Position-named options (COPA's ``primeira``/``segunda``) keep
    their order.
    """
    out: Record = {**record, "questions": {}}
    for qid, question in record["questions"].items():
        if question["type"] != "choice":
            out["questions"][qid] = question
            continue
        criteria, label = dict(question["criteria"]), question["label"]
        positional = set(criteria) <= {"primeira", "segunda", "terceira"}
        if not positional:
            roll = rng.random()
            none_options = [o for o in NONE_OPTIONS if o[0] not in criteria]
            distractors = [d for d in DISTRACTORS if d not in criteria]
            if roll < p_none and len(criteria) > 2 and none_options:
                key, description = rng.choice(none_options)
                criteria.pop(label)
                criteria[key], label = description, key
            elif roll < p_none + p_none_wrong and none_options:
                key, description = rng.choice(none_options)
                criteria[key] = description
            elif roll < p_none + p_none_wrong + p_distractor and distractors:
                key = rng.choice(distractors)
                criteria[key] = DISTRACTORS[key]
            keys = list(criteria)
            rng.shuffle(keys)
            criteria = {k: criteria[k] for k in keys}
        out["questions"][qid] = {**question, "criteria": criteria, "label": label}
    return out


# -- assembly -----------------------------------------------------------------


def state_hash(state: Any) -> str:
    """A whitespace- and case-insensitive fingerprint of a state."""
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def bench_hashes(paths: Iterable[str | Path]) -> set[str]:
    """Fingerprints of every state in benchmark files, to keep them out of training."""
    hashes: set[str] = set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                hashes.add(state_hash(json.loads(line)["state"]))
    return hashes


@dataclass
class BuildResult:
    train: list[Record]
    dev: list[Record]
    manifest: dict[str, Any] = field(default_factory=dict)


def _describe(records: list[Record]) -> dict[str, Any]:
    types: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for record in records:
        sources[record["_meta"]["source"]] += 1
        for question in record["questions"].values():
            types[question["type"]] += 1
    return {"records": len(records), "by_type": dict(sorted(types.items())), "by_source": dict(sorted(sources.items()))}


def build(
    *,
    sources: list[Source],
    n_per_source: int,
    n_synthetic: int,
    seed: int,
    cache_dir: Path,
    dev_ratio: float = 0.05,
    exclude_hashes: set[str] | None = None,
    client: httpx.Client | None = None,
) -> BuildResult:
    """Assemble, augment, deduplicate and split the training set.

    Everything is drawn from seeded generators, so the same arguments rebuild
    the same file.  Only the *train* split of each public source is read; their
    test splits stay untouched for evaluation.
    """
    exclude_hashes = exclude_hashes or set()
    records: list[Record] = []
    dropped = Counter[str]()

    for source in sources:
        rng = random.Random(f"{seed}:{source.name}")
        rows = fetch_rows(source, cache_dir, client=client)
        picked = sorted(rng.sample(range(len(rows)), min(n_per_source, len(rows))))
        converted = source.convert([rows[i] for i in picked], rng)
        for index, record in zip(picked, converted, strict=True):
            record["_meta"] = {
                "source": source.name, "id": f"{source.name}/{index}", "lang": source.lang,
                "repo": source.repo, "revision": source.revision, "licence": source.licence,
                "licence_class": source.licence_class,
            }
            for question in record["questions"].values():
                question.setdefault("src", source.name)
            records.append(augment(record, rng))

    rng = random.Random(f"{seed}:synth_pt")
    for index, record in enumerate(synth_pt.generate(n_synthetic, rng)):
        family = next(iter(record["questions"].values()))["src"]
        record["_meta"] = {
            "source": family, "id": f"{family}/{index}", "lang": "pt-BR",
            "licence": "mit", "licence_class": PERMISSIVE,
        }
        records.append(augment(record, rng))

    seen: set[str] = set()
    kept: list[Record] = []
    for record in records:
        fingerprint = state_hash(record["state"])
        key = fingerprint + json.dumps(
            {q: v["instructions"] for q, v in record["questions"].items()}, sort_keys=True, ensure_ascii=False
        )
        if fingerprint in exclude_hashes:
            dropped["in_benchmark"] += 1
        elif key in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(key)
            kept.append(record)

    random.Random(seed).shuffle(kept)
    n_dev = int(len(kept) * dev_ratio)
    dev, train = kept[:n_dev], kept[n_dev:]
    manifest = {
        "seed": seed,
        "n_per_source": n_per_source,
        "n_synthetic": n_synthetic,
        "train": _describe(train),
        "dev": _describe(dev),
        "dropped": dict(dropped),
        "sources": [
            {"name": s.name, "repo": s.repo, "revision": s.revision, "path": s.path, "lang": s.lang,
             "licence": s.licence, "licence_class": s.licence_class, "note": s.note}
            for s in sources
        ],
        "synthetic": {"generator": "alm.decision.synth_pt", "families": sorted(synth_pt.GENERATORS)},
        "labels": "human labels of the public sources and labels computed by rule; no model-labelled data",
    }
    return BuildResult(train=train, dev=dev, manifest=manifest)


def write(result: BuildResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": out_dir / "train.jsonl", "dev": out_dir / "dev.jsonl", "manifest": out_dir / "manifest.json"}
    for split in ("train", "dev"):
        records = getattr(result, split)
        paths[split].write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
    paths["manifest"].write_text(json.dumps(result.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
