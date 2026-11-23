"""Training data for the decision model: converters, generators, augmentation, assembly."""

from __future__ import annotations

import gzip
import json
import random
import re

import httpx
import pytest

from alm.decision import synth_pt
from alm.decision.bench import load_items, question_from_wire
from alm.decision.dataset import (
    PERMISSIVE,
    SHARE_ALIKE,
    SOURCES,
    UNSTATED,
    allowed_sources,
    augment,
    bench_hashes,
    build,
    state_hash,
    write,
)

_ROWS = {
    "boolq": [{"passage": f"Texto {i}.", "question": f"é verdade {i}", "label": i % 2} for i in range(6)],
    "rte": [{"premise": f"P{i}", "hypothesis": f"H{i}", "label": i % 2} for i in range(6)],
    "qnli": [{"question": f"Q{i}?", "sentence": f"S{i}", "label": i % 2} for i in range(6)],
    "sst2": [{"sentence": f"frase {i}", "label": i % 2} for i in range(6)],
    "wnli": [{"sentence1": f"A{i}", "sentence2": f"B{i}", "label": i % 2} for i in range(6)],
    "copa": [
        {"premise": f"P{i}", "choice1": "um", "choice2": "dois", "question": "cause", "label": i % 2}
        for i in range(6)
    ],
    "massive": [
        {"text": f"pedido {i}", "label_text": f"intent_{i % 12}", "label": f"intent_{i % 12}"}
        for i in range(36)
    ],
}


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("pt.json.gz"):
        body = "\n".join(json.dumps(r) for r in _ROWS["massive"]).encode()
        return httpx.Response(200, content=gzip.compress(body))
    task = re.search(r"data/(\w+?)_pt-BR/train\.jsonl", path).group(1)
    return httpx.Response(200, text="\n".join(json.dumps(r) for r in _ROWS[task]))


def _client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(_handler))


def _questions(record):
    return list(record["questions"].values())


# -- licence policy ----------------------------------------------------------


def test_licence_policy_admits_sources_by_class():
    default = {s.name for s in allowed_sources()}
    assert {s.licence_class for s in allowed_sources()} == {PERMISSIVE, SHARE_ALIKE}
    assert {s.licence_class for s in allowed_sources(only_permissive=True)} == {PERMISSIVE}
    with_unstated = allowed_sources(allow_unstated=True)
    assert {s.licence_class for s in with_unstated} == {PERMISSIVE, SHARE_ALIKE, UNSTATED}
    assert "rte_pt" not in default and "sst2_pt" not in default
    # Every source is pinned to a full commit hash.
    assert all(re.fullmatch(r"[0-9a-f]{40}", s.revision) for s in SOURCES.values())


# -- converters --------------------------------------------------------------


def test_glue_label_conventions_are_respected():
    rng = random.Random(0)
    # RTE and QNLI use 0 for "entailed"/"answers"; WNLI and BoolQ use 1 for true.
    assert [_questions(r)[0]["label"] for r in SOURCES["rte_pt"].convert(_ROWS["rte"][:2], rng)] == [True, False]
    assert [_questions(r)[0]["label"] for r in SOURCES["qnli_pt"].convert(_ROWS["qnli"][:2], rng)] == [True, False]
    assert [_questions(r)[0]["label"] for r in SOURCES["wnli_pt"].convert(_ROWS["wnli"][:2], rng)] == [False, True]
    assert [_questions(r)[0]["label"] for r in SOURCES["boolq_pt"].convert(_ROWS["boolq"][:2], rng)] == [False, True]


def test_sst2_yes_no_form_flips_with_the_question_polarity():
    for seed in range(20):
        record = SOURCES["sst2_pt"].convert([{"sentence": "ótimo filme", "label": 1}], random.Random(seed))[0]
        question = _questions(record)[0]
        if question["type"] == "choice":
            assert question["label"] == "positivo"
        else:
            assert question["label"] == ("positiva" in question["instructions"])


def test_massive_offers_the_true_intent_among_a_varying_subset():
    records = SOURCES["massive_intent_pt"].convert(_ROWS["massive"], random.Random(1))
    sizes = set()
    for record, row in zip(records, _ROWS["massive"], strict=True):
        question = _questions(record)[0]
        assert question["label"] == row["label_text"]
        assert question["label"] in question["criteria"]
        sizes.add(len(question["criteria"]))
    assert len(sizes) > 1


def test_copa_keeps_its_positional_options_in_order():
    record = SOURCES["copa_pt"].convert(_ROWS["copa"][:1], random.Random(0))[0]
    for seed in range(10):
        question = _questions(augment(record, random.Random(seed)))[0]
        assert list(question["criteria"]) == ["primeira", "segunda"]


# -- rule-based generators ---------------------------------------------------


@pytest.mark.parametrize("family", sorted(synth_pt.GENERATORS))
def test_generated_records_are_servable_and_labelled_within_their_options(family):
    rng = random.Random(7)
    labels = set()
    for _ in range(60):
        record = synth_pt.GENERATORS[family](rng)
        question = _questions(record)[0]
        question_from_wire(question)
        label = question["label"]
        if question["type"] == "choice":
            assert label in question["criteria"]
        elif question["type"] == "score":
            assert 0 <= label < len(question["criteria"])
        else:
            assert isinstance(label, bool)
        labels.add(str(label))
    # No family may collapse onto a single answer.
    assert len(labels) > 1


def test_reorder_label_follows_the_stated_rule():
    rng = random.Random(3)
    for _ in range(50):
        record = synth_pt.reorder(rng)
        state = record["state"]
        minimum = int(re.search(r"menos de (\d+) dias", state["regra"]).group(1))
        cover = state["estoque_atual_unidades"] / state["consumo_medio_diario_unidades"]
        assert _questions(record)[0]["label"] == (cover < minimum)


def test_count_label_matches_the_listed_values():
    rng = random.Random(5)
    for _ in range(50):
        record = synth_pt.count_above(rng)
        question = _questions(record)[0]
        limit = int(re.search(r"mais de (\d+) pedidos", question["instructions"]).group(1))
        values = [int(v) for v in re.findall(r": (\d+)", record["state"])]
        assert question["label"] == sum(v > limit for v in values)


def test_eligibility_is_balanced():
    rng = random.Random(11)
    passed = sum(_questions(synth_pt.eligibility(rng))[0]["label"] for _ in range(400))
    assert 150 < passed < 250


# -- augmentation ------------------------------------------------------------


def _choice_record():
    return {
        "state": "s",
        "questions": {
            "q": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B", "c": "C"}, "label": "a"}
        },
    }


def test_augmentation_keeps_the_label_valid_and_uses_none_both_ways():
    removed = added_wrong = 0
    for seed in range(300):
        question = _questions(augment(_choice_record(), random.Random(seed)))[0]
        assert question["label"] in question["criteria"]
        if question["label"] != "a":
            assert "a" not in question["criteria"]
            removed += 1
        elif len(question["criteria"]) > 3:
            added_wrong += 1
    # "None of the above" must be right sometimes and wrong sometimes.
    assert removed > 5 and added_wrong > 5


def test_augmentation_leaves_other_question_types_alone():
    record = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "?", "label": True}}}
    assert augment(record, random.Random(0))["questions"] == record["questions"]


# -- assembly ----------------------------------------------------------------


def _build(tmp_path, **overrides):
    arguments = dict(
        sources=allowed_sources(allow_unstated=True),
        n_per_source=6,
        n_synthetic=40,
        seed=0,
        cache_dir=tmp_path / "cache",
        dev_ratio=0.1,
        client=_client(),
    )
    arguments.update(overrides)
    return build(**arguments)


def test_build_is_reproducible_and_records_provenance(tmp_path):
    first, second = _build(tmp_path), _build(tmp_path)
    assert first.train == second.train
    assert first.dev and first.train

    for record in first.train:
        meta = record["_meta"]
        assert meta["source"] and meta["licence_class"] and meta["lang"].startswith("pt")
        assert all("label" in q for q in record["questions"].values())

    manifest = first.manifest
    assert manifest["train"]["records"] == len(first.train)
    assert {s["name"] for s in manifest["sources"]} == set(SOURCES)
    assert "no model-labelled data" in manifest["labels"]


def test_build_keeps_benchmark_states_out_of_training(tmp_path):
    bench = tmp_path / "bench.jsonl"
    leaked = _ROWS["rte"][0]["premise"]
    bench.write_text(json.dumps({"id": "x", "state": f"  {leaked.upper()} ", "expected": "no", "question": {}}))
    hashes = bench_hashes([bench])
    assert state_hash(leaked) in hashes

    result = _build(tmp_path, exclude_hashes=hashes)
    assert result.manifest["dropped"]["in_benchmark"] >= 1
    assert all(state_hash(r["state"]) not in hashes for r in result.train + result.dev)


def test_written_files_load_as_benchmark_items(tmp_path):
    paths = write(_build(tmp_path), tmp_path / "out")
    assert json.loads(paths["manifest"].read_text())["seed"] == 0
    items = load_items(paths["train"])
    assert items
    assert all(item.expected in item.labels for item in items)
    assert {item.type for item in items} == {"choice", "noul", "score"}
