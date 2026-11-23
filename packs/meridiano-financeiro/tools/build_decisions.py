"""Generate this pack's decision benchmark from the Grupo Meridiano workbook.

    python packs/meridiano-financeiro/tools/build_decisions.py [caminho-do-xlsx] [destino.jsonl]

The output is a JSON Lines file in the JevBench record shape, in Brazilian
Portuguese, for `alm decide bench`. Edit this file, never the .jsonl.

Two properties matter and are worth keeping when adding a family.

**Every expected answer is computed, not judged.** Each item states the facts
in `state` and the rule in the question; the label comes from arithmetic over
the workbook (or, for routing, from which area the question was written for).
No model labelled anything, so a model's score here is not agreement with
another model.

**Families are balanced.** Candidates are drawn round-robin across expected
labels, so always answering the commonest label scores near chance and a
position or "yes" bias shows up as lost accuracy instead of hiding.

The state never contains the derived number the question asks about — the
growth rate, the attainment, the net price — only the inputs to it.
"""

from __future__ import annotations

import json
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import openpyxl

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
SRC = sys.argv[1] if len(sys.argv) > 1 else str(_REPO / "dataset" / "Grupo_Demo_Dataset_Financeiro.xlsx")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else _HERE.parent / "decisions" / "fpa.jsonl"

SEED = 20261123
YEARS = [2025, 2026, 2027]
MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

UNITS = {
    "Materiais de Construcao": ("materiais_construcao", "Materiais de Construção"),
    "Utilidades Domesticas": ("utilidades_domesticas", "Utilidades Domésticas"),
    "Papelaria e Impressao": ("papelaria_impressao", "Papelaria e Impressão"),
    "Servicos Automotivos": ("servicos_automotivos", "Serviços Automotivos"),
    "Vestuario": ("vestuario", "Vestuário"),
}
UNIT_CRITERIA = {slug: f"Unidade {name}" for slug, name in UNITS.values()}

wb = openpyxl.load_workbook(SRC, read_only=True, data_only=True)
SHEETS = {name: [list(r) for r in wb[name].iter_rows(values_only=True)] for name in wb.sheetnames}
wb.close()


# -- formatting ---------------------------------------------------------------


def brl(value: float, decimals: int = 0) -> str:
    text = f"{abs(float(value)):,.{decimals}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return f"-R$ {text}" if value < 0 else f"R$ {text}"


def cap(text: str) -> str:
    """Upper-case the first letter only, so acronyms such as EBITDA survive."""
    return text[:1].upper() + text[1:]


def pct(value: float, decimals: int = 1) -> str:
    return f"{float(value) * 100:.{decimals}f}".replace(".", ",") + "%"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# -- data ---------------------------------------------------------------------


def records(sheet: str) -> list[dict]:
    header, *rows = SHEETS[sheet]
    return [dict(zip(header, row, strict=False)) for row in rows if row and row[0] is not None]


DRE_UN = records("DRE_por_UN")
HEADCOUNT = records("Headcount_Folha")
RECEIVABLES = records("Contas_Receber")
INVENTORY = records("Estoque")
KPIS = records("KPIs_Mensais")

# Annual figures per unit.
ANNUAL: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
MONTHLY_REVENUE: dict[tuple[str, int, int], float] = {}
for row in DRE_UN:
    key = (row["Unidade"], row["Ano"])
    for column in ("Receita Bruta", "Receita Liquida", "Lucro Bruto", "EBITDA da Unidade"):
        ANNUAL[key][column] += row[column]
    MONTHLY_REVENUE[(row["Unidade"], row["Ano"], row["Mes"])] = row["Receita Bruta"]
for row in HEADCOUNT:
    ANNUAL[(row["Unidade"], row["Ano"])]["Folha Total"] += row["Folha Total"]
    if row["Mes"] == 12:
        ANNUAL[(row["Unidade"], row["Ano"])]["Headcount"] = row["Headcount"]
for row in RECEIVABLES:
    if row["Mes"] == 12:
        ANNUAL[(row["Unidade"], row["Ano"])]["DSO"] = row["DSO aplic."]
for row in INVENTORY:
    if row["Mes"] == 12:
        ANNUAL[(row["Unidade"], row["Ano"])]["DIO"] = row["DIO (dias)"]
for figures in ANNUAL.values():
    figures["Margem Bruta"] = figures["Lucro Bruto"] / figures["Receita Liquida"]

#: column → (name shown, formatter)
METRICS = {
    "Receita Bruta": ("receita bruta", brl),
    "Lucro Bruto": ("lucro bruto", brl),
    "EBITDA da Unidade": ("EBITDA da unidade", brl),
    "Margem Bruta": ("margem bruta", pct),
    "Folha Total": ("folha de pagamento total", brl),
    "Headcount": ("número de funcionários em dezembro", lambda v: str(int(v))),
    "DSO": ("prazo médio de recebimento (dias)", lambda v: str(int(v))),
    "DIO": ("prazo médio de estocagem (dias)", lambda v: str(int(v))),
}


def _vendas() -> tuple[list[dict], list[dict]]:
    rows = SHEETS["Vendas_Representantes"]
    start = next(i for i, r in enumerate(rows) if r and r[0] == "Representante") + 1
    reps = []
    for row in rows[start:]:
        if not row or row[0] in (None, "TOTAL / MEDIA"):
            break
        for index, year in enumerate(YEARS):
            reps.append(
                {"rep": row[0], "unit": row[1], "year": year,
                 "realizado": row[2 + 3 * index], "meta": row[3 + 3 * index]}
            )
    products = [
        {"unit": r[0], "product": r[1], "price": r[2], "cost": r[3]}
        for r in SHEETS["CMV_Custos_Produto"][4:]
        if r and r[0] in UNITS
    ]
    return reps, products


REPS, PRODUCTS = _vendas()
SCENARIOS = {r[0]: r[1:4] for r in SHEETS["Cenarios"][4:] if r and r[0]}


# -- assembly -----------------------------------------------------------------

ITEMS: list[dict] = []


def balanced(candidates: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Up to ``n`` candidates, drawn round-robin across expected labels."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_label[str(candidate["expected"])].append(candidate)
    for group in by_label.values():
        rng.shuffle(group)
    picked: list[dict] = []
    labels = sorted(by_label)
    while len(picked) < n and any(by_label.values()):
        for label in labels:
            if by_label[label] and len(picked) < n:
                picked.append(by_label[label].pop())
    return picked


def emit(family: str, basis: str, candidates: list[dict], n: int) -> None:
    rng = random.Random(f"{SEED}-{family}")
    for index, item in enumerate(balanced(candidates, n, rng), start=1):
        item_id = f"{family}-{index:03d}"
        ITEMS.append(
            {
                "id": item_id,
                "family": family,
                "lang": "pt-BR",
                # A stable third of the items is set aside for fitting temperatures.
                "split": "dev" if zlib.crc32(item_id.encode()) % 10 < 3 else "test",
                "state": item["state"],
                "labels": item["labels"],
                "expected": item["expected"],
                "question": item["question"],
                "provenance": {
                    "source": "Grupo Meridiano Varejo (dataset sintético), dataset/Grupo_Demo_Dataset_Financeiro.xlsx",
                    "label_basis": basis,
                    "license": "MIT",
                },
            }
        )


def noul(instructions: str, when_true: str = "", when_false: str = "") -> dict:
    question = {"type": "noul", "instructions": instructions}
    if when_true:
        question["criteria"] = {"true": when_true, "false": when_false}
    return question


YES_NO = ["no", "yes"]


# 1 · year-over-year comparison of one unit -----------------------------------

candidates = []
for unit, (_, name) in UNITS.items():
    for column in ("Receita Bruta", "Lucro Bruto", "EBITDA da Unidade", "Margem Bruta"):
        label, fmt = METRICS[column]
        for y1, y2 in ((2025, 2026), (2026, 2027), (2025, 2027)):
            a, b = ANNUAL[(unit, y1)][column], ANNUAL[(unit, y2)][column]
            if fmt(a) == fmt(b):
                continue
            for direction in ("maior", "menor"):
                truth = b > a if direction == "maior" else b < a
                candidates.append(
                    {
                        "state": f"Unidade {name} — {label} por ano:\n"
                        + table(["Ano", cap(label)], [[str(y), fmt(ANNUAL[(unit, y)][column])] for y in YEARS]),
                        "labels": YES_NO,
                        "expected": "yes" if truth else "no",
                        "question": noul(
                            f"A {label} da unidade {name} em {y2} foi {direction} do que em {y1}?"
                            if column != "Lucro Bruto" and column != "EBITDA da Unidade"
                            else f"O {label} da unidade {name} em {y2} foi {direction} do que em {y1}?"
                        ),
                    }
                )
emit("comparacao_anual", "comparação aritmética dos valores anuais da planilha", candidates, 30)


# 2 · which unit leads on a metric --------------------------------------------

candidates = []
rng = random.Random(f"{SEED}-order")
for column, (label, fmt) in METRICS.items():
    for year in YEARS:
        values = {unit: ANNUAL[(unit, year)][column] for unit in UNITS}
        for direction, pick in (("maior", max), ("menor", min)):
            best = pick(values, key=values.get)
            # A tie has no single right answer.
            if sum(1 for v in values.values() if fmt(v) == fmt(values[best])) > 1:
                continue
            order = list(UNITS)
            rng.shuffle(order)
            candidates.append(
                {
                    "state": f"{cap(label)} por unidade de negócio em {year}:\n"
                    + table(["Unidade", cap(label)], [[UNITS[u][1], fmt(values[u])] for u in order]),
                    "labels": [slug for slug, _ in UNITS.values()],
                    "expected": UNITS[best][0],
                    "question": {
                        "type": "choice",
                        "instructions": f"Qual unidade de negócio teve o {direction} valor de {label} em {year}?",
                        "criteria": UNIT_CRITERIA,
                    },
                }
            )
emit("unidade_extrema", "máximo ou mínimo calculado entre as cinco unidades", candidates, 30)


# 3 · growth band --------------------------------------------------------------

GROWTH_LEVELS = [
    "Queda: a receita diminuiu",
    "Estável: cresceu até 5%",
    "Crescimento moderado: cresceu mais de 5% e até 15%",
    "Crescimento forte: cresceu mais de 15%",
]


def growth_level(before: float, after: float) -> int:
    growth = after / before - 1.0
    return 0 if growth < 0 else 1 if growth <= 0.05 else 2 if growth <= 0.15 else 3


candidates = []
for unit, (_, name) in UNITS.items():
    for year in (2026, 2027):
        for month in range(1, 13):
            before = MONTHLY_REVENUE[(unit, year - 1, month)]
            after = MONTHLY_REVENUE[(unit, year, month)]
            # Skip cases that sit on a band edge once rounded for display.
            if any(abs((after / before - 1.0) - edge) < 0.004 for edge in (0.0, 0.05, 0.15)):
                continue
            candidates.append(
                {
                    "state": f"Receita bruta da unidade {name}:\n"
                    f"- {MONTHS[month - 1]}/{year - 1}: {brl(before)}\n"
                    f"- {MONTHS[month - 1]}/{year}: {brl(after)}",
                    "labels": [str(i) for i in range(len(GROWTH_LEVELS))],
                    "expected": str(growth_level(before, after)),
                    "question": {
                        "type": "score",
                        "instructions": "Classifique a variação da receita bruta do mês em relação ao mesmo mês do ano anterior.",
                        "criteria": GROWTH_LEVELS,
                    },
                }
            )
emit("faixa_crescimento", "variação percentual calculada e enquadrada nas faixas do critério", candidates, 28)


# 4 · sales target attainment ---------------------------------------------------

TARGET_CRITERIA = {
    "abaixo": "Realizado inferior a 95% da meta",
    "na_meta": "Realizado entre 95% e 105% da meta, inclusive",
    "acima": "Realizado superior a 105% da meta",
}
candidates = []
for rep in REPS:
    attainment = rep["realizado"] / rep["meta"]
    if any(abs(attainment - edge) < 0.003 for edge in (0.95, 1.05)):
        continue
    candidates.append(
        {
            "state": {
                "representante": rep["rep"],
                "segmento": UNITS[rep["unit"]][1],
                "ano": rep["year"],
                "meta_anual": brl(rep["meta"]),
                "vendas_realizadas": brl(rep["realizado"]),
            },
            "labels": list(TARGET_CRITERIA),
            "expected": "abaixo" if attainment < 0.95 else "acima" if attainment > 1.05 else "na_meta",
            "question": {
                "type": "choice",
                "instructions": "Como o representante ficou em relação à meta anual de vendas?",
                "criteria": TARGET_CRITERIA,
            },
        }
    )
emit("meta_representante", "atingimento calculado (realizado ÷ meta) e enquadrado nas faixas", candidates, 27)


# 5 · discount policy: allowed or not -------------------------------------------

POLICY = (
    "Política comercial: é proibido vender abaixo do custo unitário, em qualquer hipótese. "
    "Descontos acima de 15% sobre o preço de tabela só podem ser fechados com aprovação "
    "registrada do gerente."
)
DISCOUNTS = [0.05, 0.10, 0.12, 0.18, 0.22, 0.28, 0.32, 0.40, 0.46, 0.50, 0.55]

sale_candidates, approver_candidates = [], []
APPROVER_CRITERIA = {
    "vendedor": "Desconto de até 10%, com preço final igual ou acima do custo",
    "gerente": "Desconto acima de 10% e até 20%, com preço final igual ou acima do custo",
    "diretoria": "Desconto acima de 20%, com preço final igual ou acima do custo",
    "vetada": "Preço final abaixo do custo unitário: a venda não pode ser aprovada por ninguém",
}
rng = random.Random(f"{SEED}-sales")
for product in PRODUCTS:
    for discount in DISCOUNTS:
        net = product["price"] * (1.0 - discount)
        if abs(net - product["cost"]) < 0.02 * product["cost"]:
            continue
        above_cost = net >= product["cost"]
        facts = (
            f"Produto: {product['product'].title()} ({UNITS[product['unit']][1]})\n"
            f"Preço de tabela: {brl(product['price'], 2)}\n"
            f"Custo unitário: {brl(product['cost'], 2)}\n"
            f"Desconto proposto: {pct(discount, 0)}"
        )
        approved = rng.random() < 0.5
        allowed = above_cost and (discount <= 0.15 or approved)
        sale_candidates.append(
            {
                "state": f"{POLICY}\n\n{facts}\n"
                f"Aprovação do gerente registrada: {'sim' if approved else 'não'}",
                "labels": YES_NO,
                "expected": "yes" if allowed else "no",
                "question": noul(
                    "Pela política, a venda pode ser fechada exatamente como proposta?",
                    "O preço final não fica abaixo do custo e, se o desconto passa de 15%, há aprovação do gerente.",
                    "O preço final fica abaixo do custo, ou o desconto passa de 15% sem aprovação do gerente.",
                ),
            }
        )
        approver = (
            "vetada" if not above_cost
            else "vendedor" if discount <= 0.10
            else "gerente" if discount <= 0.20
            else "diretoria"
        )
        approver_candidates.append(
            {
                "state": facts,
                "labels": list(APPROVER_CRITERIA),
                "expected": approver,
                "question": {
                    "type": "choice",
                    "instructions": "Pela tabela de alçadas, quem precisa aprovar esta venda?",
                    "criteria": APPROVER_CRITERIA,
                },
            }
        )
emit("venda_permitida", "preço final calculado e comparado ao custo e à regra de aprovação", sale_candidates, 28)
emit("alcada_desconto", "preço final e faixa de desconto calculados conforme a tabela de alçadas", approver_candidates, 28)


# 6 · cash conversion cycle -----------------------------------------------------

cycles = sorted(row["DSO (dias)"] + row["DIO (dias)"] - row["DPO (dias)"] for row in KPIS)
threshold = round(cycles[len(cycles) // 2])
candidates = []
for row in KPIS:
    cycle = row["DSO (dias)"] + row["DIO (dias)"] - row["DPO (dias)"]
    if abs(cycle - threshold) < 0.3:
        continue
    candidates.append(
        {
            "state": f"Indicadores de capital de giro do grupo em {MONTHS[row['Mes'] - 1]}/{row['Ano']}:\n"
            f"- Prazo médio de recebimento (DSO): {row['DSO (dias)']:.1f} dias\n".replace(".", ",")
            + f"- Prazo médio de estocagem (DIO): {row['DIO (dias)']:.1f} dias\n".replace(".", ",")
            + f"- Prazo médio de pagamento (DPO): {row['DPO (dias)']:.1f} dias".replace(".", ","),
            "labels": YES_NO,
            "expected": "yes" if cycle > threshold else "no",
            "question": noul(
                f"O ciclo de caixa do mês (DSO + DIO − DPO) ficou acima de {threshold} dias?"
            ),
        }
    )
emit("ciclo_de_caixa", "ciclo de caixa calculado (DSO + DIO − DPO) e comparado ao limite", candidates, 24)


# 7 · quarter trend -------------------------------------------------------------

candidates = []
for unit, (_, name) in UNITS.items():
    for year in YEARS:
        for start in (1, 4, 7, 10):
            values = [MONTHLY_REVENUE[(unit, year, m)] for m in range(start, start + 3)]
            rising = values[0] < values[1] < values[2]
            candidates.append(
                {
                    "state": f"Receita bruta mensal da unidade {name} em {year}:\n"
                    + "\n".join(f"- {MONTHS[start - 1 + i]}: {brl(v)}" for i, v in enumerate(values)),
                    "labels": YES_NO,
                    "expected": "yes" if rising else "no",
                    "question": noul(
                        "A receita cresceu em todos os meses do trimestre, isto é, cada mês ficou acima do mês anterior?"
                    ),
                }
            )
emit("tendencia_trimestre", "comparação mês a mês dos três valores", candidates, 24)


# 8 · scenarios -----------------------------------------------------------------

SCENARIO_NAMES = ["conservador", "base", "otimista"]
SCENARIO_LINES = {
    "Receita Bruta": "receita bruta",
    "EBITDA": "EBITDA",
    "Lucro Liquido": "lucro líquido",
    "Caixa aprox. fim 2027": "caixa ao fim de 2027",
}
scenario_table = table(
    ["Linha (2027)", "Conservador", "Base", "Otimista"],
    [[cap(label), *[brl(v) for v in SCENARIOS[line]]] for line, label in SCENARIO_LINES.items()],
)
candidates = []
for line, label in SCENARIO_LINES.items():
    values = sorted(SCENARIOS[line])
    # Thresholds between consecutive scenarios, below all and above all.
    cuts = [values[0] * 0.9 - 1000, (values[0] + values[1]) / 2, (values[1] + values[2]) / 2, values[2] * 1.1 + 1000]
    for cut in cuts:
        cut = round(cut, -3)
        count = sum(1 for v in SCENARIOS[line] if v > cut)
        candidates.append(
            {
                "state": f"Análise de cenários para 2027, grupo consolidado:\n{scenario_table}",
                "labels": ["0", "1", "2", "3"],
                "expected": str(count),
                "question": {
                    "type": "score",
                    "instructions": f"Em quantos dos três cenários o valor de {label} fica acima de {brl(cut)}?",
                    "criteria": ["Nenhum cenário", "Um cenário", "Dois cenários", "Os três cenários"],
                },
            }
        )
emit("cenarios", "contagem dos cenários que superam o limite", candidates, 16)


# 9 · routing -------------------------------------------------------------------

AREAS = {
    "resultado_margens": "Receita, custos, margens, EBITDA e lucro (demonstração de resultado)",
    "capital_giro": "Contas a receber e a pagar, estoques, prazos médios, ciclo de caixa e liquidez",
    "pessoal_folha": "Quadro de funcionários, salários, encargos e folha de pagamento",
    "vendas_metas": "Desempenho de representantes comerciais contra metas de vendas",
    "cenarios_sensibilidade": "Projeções, cenários conservador/base/otimista e análises de sensibilidade",
}
ROUTING = {
    "resultado_margens": [
        "Por que a margem EBITDA do grupo caiu de 2025 para 2027?",
        "Quanto o CMV representou da receita líquida em Vestuário no ano passado?",
        "Qual unidade mais contribuiu para o lucro bruto do grupo?",
        "O lucro líquido de 2026 foi maior que o de 2025?",
        "Quanto pagamos de impostos sobre vendas em Materiais de Construção?",
        "A despesa de marketing cresceu mais rápido que a receita?",
    ],
    "capital_giro": [
        "Quantos dias, em média, levamos para receber dos clientes de Serviços Automotivos?",
        "Quanto do contas a receber está vencido há mais de 90 dias?",
        "O estoque de Utilidades Domésticas está girando mais devagar?",
        "Estamos pagando fornecedores antes de receber dos clientes?",
        "Qual foi a liquidez corrente do grupo em dezembro?",
        "Quanto capital está parado em estoque na Papelaria?",
    ],
    "pessoal_folha": [
        "Quantos funcionários a unidade de Vestuário tinha no fim de 2026?",
        "Quanto os encargos pesam sobre os salários em Materiais de Construção?",
        "A folha de pagamento cresceu depois das contratações de julho?",
        "Qual é o custo médio por funcionário em Serviços Automotivos?",
        "Quanto gastamos com salários no grupo inteiro em 2025?",
        "Houve aumento de quadro na Papelaria no último ano?",
    ],
    "vendas_metas": [
        "A Renata Villaça bateu a meta de vendas em 2026?",
        "Qual representante ficou mais longe da meta no ano passado?",
        "Quem lidera o ranking de atingimento dos três anos?",
        "As metas dos representantes do Sul estão altas demais?",
        "Quanto o Diego Fontes vendeu acima da meta em 2025?",
        "Qual foi o atingimento médio da equipe comercial em 2027?",
    ],
    "cenarios_sensibilidade": [
        "O que acontece com o EBITDA de 2027 se a margem bruta cair 3 pontos?",
        "No cenário conservador, o grupo fecha 2027 com prejuízo?",
        "Quanto de caixa teremos ao fim de 2027 no cenário otimista?",
        "Qual é o impacto no lucro se os custos operacionais subirem 4%?",
        "Qual premissa de crescimento foi usada para projetar 2027?",
        "Se a receita crescer 6 pontos a menos, quanto perdemos de EBITDA?",
    ],
}
candidates = [
    {
        "state": question,
        "labels": list(AREAS),
        "expected": area,
        "question": {
            "type": "choice",
            "instructions": "Qual área de análise financeira deve responder a esta pergunta?",
            "criteria": AREAS,
        },
    }
    for area, questions in ROUTING.items()
    for question in questions
]
emit("roteamento", "por construção: cada pergunta foi escrita para uma área", candidates, 30)


# -- write ----------------------------------------------------------------------

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in ITEMS) + "\n", encoding="utf-8")

by_family: dict[str, list[dict]] = defaultdict(list)
for item in ITEMS:
    by_family[item["family"]].append(item)
print(f"{len(ITEMS)} items → {OUT}")
for family, items in by_family.items():
    counts = defaultdict(int)
    for item in items:
        counts[str(item["expected"])] += 1
    spread = ", ".join(f"{label}={n}" for label, n in sorted(counts.items()))
    print(f"  {family:22s} {len(items):>3}  {items[0]['question']['type']:6s}  {spread}")
