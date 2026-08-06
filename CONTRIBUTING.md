# Contributing

Thanks for considering a contribution.

## Setup

Requires **Python ≥ 3.11** (the codebase uses `StrEnum`) and **pip ≥ 21.3** (editable installs of a
`setup.py`-less project need PEP 660). On macOS the system `python3` is 3.9 — name a newer
interpreter explicitly when creating the venv.

```bash
git clone https://github.com/ceschmiedel/ALM.git
cd ALM

python3.12 -m venv venv && source venv/bin/activate
python -m pip install --upgrade pip

pip install -e ".[dev]"
pytest
ruff check .
```

## Before opening a pull request

```bash
ruff check alm/ tests/
pytest -q
alm pack validate packs/demo-enterprise
```

CI runs the same three, plus an end-to-end smoke test on Python 3.11 and 3.12.

## What we especially welcome

**Domain packs.** A pack for a new vertical — its ontology, experts, policies and evaluation set —
is the highest-value contribution, because it exercises the parts of the architecture that generic
tests cannot: whether the capability declarations actually route, and whether the federation beats
the baseline in that domain.

**Backends.** Register a new inference backend with `alm.models.backends.register_backend`.

**Arbitration strategies.** Subclass `ArbitrationStrategy` and return `None` to defer to the next
strategy in the chain.

## House rules

- **Tests describe behaviour, not implementation.** A test named after the property it protects
  survives a refactor; one named after a method does not.
- **Degradation is reported, never silent.** If a component cannot do its job — no embedder, no
  orchestrator, an uncalibrated expert — say so in the trace, in `alm doctor`, and in `/v1/health`.
  A quietly worse answer is the failure mode this architecture exists to avoid.
- **New knobs get a default that works.** Someone should be able to install a pack and ask a
  question without configuring anything.
- **Do not tune the demo pack to make the federation win.** The evaluation harness must be willing
  to return a verdict against the federation; that willingness is the whole point.

## Reporting a bug

Include the output of `alm doctor --json` and, where relevant,
`alm explain "<the question>" --json` — routing signals usually explain a surprising answer.
