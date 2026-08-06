"""The `alm` command-line interface."""

from __future__ import annotations

import json

import pytest

from alm.cli import build_parser, main


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    try:
        code = main(list(argv))
    except SystemExit as exc:  # commands may exit non-zero deliberately
        code = int(exc.code or 0)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_parser_exposes_the_documented_commands():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])

    for argv in (
        ["init"],
        ["ask", "hello"],
        ["pack", "install", "p"],
        ["model", "list"],
        ["eval", "run"],
        ["audit", "tail"],
        ["bridge", "serve"],
    ):
        assert parser.parse_args(argv).func is not None


def test_init_creates_the_database(capsys):
    code, out, _ = _run(capsys, "init", "--json")
    assert code == 0
    assert json.loads(out)["status"] == "ready"


def test_config_show_masks_secrets(capsys, monkeypatch):
    monkeypatch.setenv("ALM_API_TOKEN", "s3cret")
    from alm.config import reset_settings_cache

    reset_settings_cache()
    code, out, _ = _run(capsys, "config", "--json")
    assert code == 0
    assert json.loads(out)["api_token"] == "***"
    reset_settings_cache()


def test_pack_validate_accepts_the_demo_pack(capsys, pack_path):
    code, out, _ = _run(capsys, "pack", "validate", str(pack_path), "--json")
    payload = json.loads(out)
    assert payload["valid"] is True
    assert code == 0


def test_pack_validate_rejects_a_broken_pack(capsys, tmp_path):
    pack = tmp_path / "broken"
    pack.mkdir()
    (pack / "pack.yaml").write_text(
        "name: broken\nontologies: []\nexperts: []\n", encoding="utf-8"
    )
    code, out, _ = _run(capsys, "pack", "validate", str(pack), "--json")
    payload = json.loads(out)
    assert payload["valid"] is False
    assert payload["problems"]
    assert code == 1


def test_pack_install_then_inspect(capsys, pack_path):
    code, out, _ = _run(capsys, "pack", "install", str(pack_path), "--json")
    assert code == 0
    report = json.loads(out)
    assert report["experts"]
    assert report["chunks"] > 0

    code, out, _ = _run(capsys, "expert", "list", "--json")
    assert code == 0
    assert len(json.loads(out)) == 3

    code, out, _ = _run(capsys, "graph", "stats", "--json")
    assert json.loads(out)["nodes"] > 0


def test_ask_produces_a_result(capsys, pack_path):
    _run(capsys, "pack", "install", str(pack_path), "--json")
    code, out, _ = _run(
        capsys, "ask", "What does clause 7.2 cap the liability at?", "--json"
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["status"] == "success"
    assert payload["answer"]
    assert payload["audit_trail"] is not None


def test_explain_reports_routing_signals(capsys, pack_path):
    _run(capsys, "pack", "install", str(pack_path), "--json")
    code, out, _ = _run(capsys, "explain", "Is clause 7.2 enforceable?", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["matches"]
    assert payload["would_escalate"] is False


def test_model_add_and_list(capsys):
    _run(capsys, "init", "--json")
    code, _, _ = _run(
        capsys,
        "model",
        "add",
        "legal-lora",
        "--tier",
        "slm",
        "--backend",
        "vllm",
        "--base-model",
        "llama-3.2-3b",
        "--adapter",
        "legal-v1",
        "--json",
    )
    assert code == 0

    code, out, _ = _run(capsys, "model", "list", "--json")
    payload = json.loads(out)
    assert payload["serving"]["adapters"] == 1
    assert "llama-3.2-3b" in payload["serving"]["adapter_groups"]


def test_model_cost_converts_gpu_hours(capsys):
    _run(capsys, "init", "--json")
    code, out, _ = _run(
        capsys,
        "model",
        "cost",
        "--gpu-hourly",
        "2.0",
        "--tokens-per-second",
        "1000",
        "--json",
    )
    assert code == 0
    assert json.loads(out)["cost_per_1k_tokens_usd"] > 0


def test_model_promote_is_gated(capsys):
    _run(capsys, "init", "--json")
    _run(capsys, "model", "add", "m1", "--json")

    from alm.mlops.versions import VersionRegistry

    VersionRegistry().record("m1", "1")
    code, _, err = _run(capsys, "model", "promote", "m1", "1", "--json")
    assert code == 1
    assert "no recorded" in err


def test_cmrag_search_from_the_cli(capsys, pack_path):
    _run(capsys, "pack", "install", str(pack_path), "--json")
    code, out, _ = _run(
        capsys, "cmrag", "search", "liability cap", "--domain", "legal", "--json"
    )
    assert code == 0
    assert json.loads(out)["chunks"]


def test_audit_summary(capsys, pack_path):
    _run(capsys, "pack", "install", str(pack_path), "--json")
    _run(capsys, "ask", "What are the payment terms?", "--json")
    code, out, _ = _run(capsys, "audit", "summary", "--json")
    assert code == 0
    assert "total" in json.loads(out)


def test_doctor_reports_findings(capsys, pack_path):
    _run(capsys, "pack", "install", str(pack_path), "--json")
    code, out, _ = _run(capsys, "doctor", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["experts"] == 3
    assert "model_reachability" in payload


def test_expert_promote_check(capsys, pack_path):
    _run(capsys, "pack", "install", str(pack_path), "--json")
    code, out, _ = _run(
        capsys, "expert", "promote-check", "contracts-expert", "--sovereignty", "--json"
    )
    assert code == 0
    assert json.loads(out)["recommend_dedicated"] is True


def test_eval_run_requires_a_dataset(capsys, pack_path):
    _run(capsys, "pack", "install", str(pack_path), "--json")
    code, _, err = _run(capsys, "eval", "run", "--json")
    assert code == 1
    assert "--dataset" in err


def test_eval_run_over_the_pack(capsys, pack_path):
    _run(capsys, "pack", "install", str(pack_path), "--json")
    code, out, _ = _run(
        capsys,
        "eval",
        "run",
        "--dataset",
        str(pack_path / "eval" / "legal.jsonl"),
        "--json",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["federation"]["cases"] > 0
    assert payload["federation"]["auditability"] == 1.0


def test_unknown_expert_fails_cleanly(capsys, pack_path):
    _run(capsys, "pack", "install", str(pack_path), "--json")
    code, _, err = _run(capsys, "expert", "show", "ghost", "--json")
    assert code == 1
    assert "not installed" in err
