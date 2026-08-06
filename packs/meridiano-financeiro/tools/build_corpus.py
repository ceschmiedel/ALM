"""Convert the Grupo Meridiano financial workbook into this pack's corpus.

    python tools/build_corpus.py <caminho-do-xlsx> [destino]

Requires openpyxl, which ALM itself does not depend on — install it in a
throwaway environment rather than in the ALM venv.

Two decisions are worth knowing before editing this file.

The corpus is **narrative and pre-aggregated** on purpose. Retrieval hands the
expert six chunks, and a 3B model quotes what it is given far more reliably than
it computes over raw monthly series. So every figure a question might ask for is
stated outright, already aggregated.

Superlatives are **computed and written as sentences** rather than left implicit
in a table. An earlier draft only tabulated per-unit margins and gave Utilidades
Domesticas a section of its own; the model then answered "qual unidade tem a
maior margem bruta" with that unit — the salient one — instead of Vestuario, the
correct one. Symmetric per-unit profiles plus explicit superlative sentences
fixed it. Keep that shape.
"""

import sys
from pathlib import Path

import openpyxl

_HERE = Path(__file__).resolve().parent
SRC = sys.argv[1] if len(sys.argv) > 1 else str(_HERE.parent / "source.xlsx")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else _HERE.parent / "corpus" / "finance"

wb = openpyxl.load_workbook(SRC, read_only=True, data_only=True)
rows = {name: [list(r) for r in wb[name].iter_rows(values_only=True)] for name in wb.sheetnames}
wb.close()

YEARS = [2025, 2026, 2027]
UNITS = [
    "Materiais de Construcao",
    "Utilidades Domesticas",
    "Papelaria e Impressao",
    "Servicos Automotivos",
    "Vestuario",
]


def brl(v, decimals=0):
    """Format a number the way the rest of the corpus reads it: pt-BR, R$."""
    if v is None:
        return "n/d"
    s = f"{float(v):,.{decimals}f}"
    s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return f"R$ {s}"


def pct(v, decimals=1):
    if v is None:
        return "n/d"
    return f"{float(v) * 100:.{decimals}f}".replace(".", ",") + "%"


def num(v, decimals=1):
    if v is None:
        return "n/d"
    return f"{float(v):.{decimals}f}".replace(".", ",")


def wide_row(sheet, label):
    """A row from a wide sheet: {'meses': [...36], 2025: t, 2026: t, 2027: t}."""
    for r in rows[sheet]:
        if r and r[0] == label:
            return {"meses": r[2:38], 2025: r[39], 2026: r[40], 2027: r[41]}
    return None


def wide_table(sheet, labels, fmt=brl):
    out = ["| Linha | 2025 | 2026 | 2027 |", "|---|---|---|---|"]
    for label in labels:
        r = wide_row(sheet, label)
        if not r:
            continue
        f = pct if "%" in label else fmt
        out.append(f"| {label} | {f(r[2025])} | {f(r[2026])} | {f(r[2027])} |")
    return "\n".join(out)


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")
    print(f"  {name:44s} {len(text):>7,} chars")


# --------------------------------------------------------------------------
# 1. Visão geral — entity, provenance, and the known dynamics (the answer key)
# --------------------------------------------------------------------------
write(
    "visao-geral-grupo-meridiano.md",
    """
# Grupo Meridiano Varejo — visão geral do grupo e do conjunto de dados

O Grupo Meridiano Varejo Ltda é um grupo varejista consolidado, entidade
fictícia e sintética, usado como base de conhecimento financeira. O modelo
cobre 36 meses, de janeiro de 2025 a dezembro de 2027, com demonstração de
resultado, balanço patrimonial e fluxo de caixa totalmente integrados.

## Unidades de negócio

O grupo opera cinco unidades de negócio, cada uma com perfil próprio de margem,
sazonalidade, prazos e crescimento:

- Materiais de Construção
- Utilidades Domésticas
- Papelaria e Impressão
- Serviços Automotivos
- Vestuário

## Procedência dos dados

A receita de 2025 e 2026 é real, derivada de notas fiscais agregadas por mês e
por unidade — é a linha de topo verdadeira. A projeção de 2027 é modelada: cada
mês de 2027 parte do mesmo mês de 2026 e aplica o crescimento por unidade
definido nas premissas. Custos, balanço e fluxo de caixa são sintéticos: CMV,
folha, ocupação, capital de giro, imobilizado, dívida, impostos e dividendos são
modelados a partir dos drivers. Nenhum custo real foi usado.

## Dinâmicas conhecidas do negócio

O conjunto de dados contém padrões intencionais que um analista de FP&A deve
identificar. São seis:

**Rentabilidade declinante.** A receita cresce todo ano, mas o EBITDA e a margem
líquida caem. A margem EBITDA vai de cerca de 8,6% em 2025 para cerca de 5,6% em
2027. Crescimento não virou lucro.

**Compressão de margem.** Utilidades Domésticas tem margem bruta caindo de 38% em
2025 para 32% em 2027. É o principal motor da queda de rentabilidade do grupo,
por ser a única unidade cujo lucro bruto encolhe em valor absoluto no período.
Por receita, é a segunda maior unidade do grupo, atrás de Vestuário.

**Deterioração do capital de giro.** O DSO consolidado sobe de aproximadamente
21 dias para aproximadamente 26 dias, e o ciclo de conversão de caixa se alonga,
pressionando o caixa operacional.

**Sazonalidade.** Vestuário concentra vendas em novembro e dezembro. Os meses de
janeiro são fracos e chegam a gerar prejuízo pontual.

**Ciclo de capex.** Há uma reforma de lojas concentrada entre junho e agosto de
2025, de R$ 30.000 por mês, visível no fluxo de investimento.

**Posição de caixa líquido.** O grupo mantém mais caixa do que dívida, ou seja,
dívida líquida negativa. Por construção, múltiplos como Dívida Líquida/EBITDA
aparecem negativos.

## Integridade e limitações

O modelo de três demonstrações é integrado e não tem referências circulares. Os
juros são calculados sobre os saldos de abertura de cada mês, o que quebra o
laço juros-caixa-lucro. O balanço fecha exatamente, com diferença zero, nos 36
meses.

Os dados são sintéticos, para demonstração e teste. Não representam uma empresa
real. As premissas de custo, capital de giro e estrutura de capital foram
calibradas para plausibilidade, não para precisão setorial. A projeção de 2027 é
ilustrativa e sensível às premissas. Os números tributários são simplificações.
""",
)

# --------------------------------------------------------------------------
# 2. Premissas e drivers
# --------------------------------------------------------------------------
prem = rows["Premissas"]
globais = "\n".join(
    f"- {r[0]}: {pct(r[1]) if isinstance(r[1], float) and r[1] < 1 else brl(r[1]) if isinstance(r[1], (int, float)) and r[1] >= 100 else num(r[1], 0)}"
    for r in prem[4:15]
    if r and r[0] and r[1] is not None
)
aberturas = "\n".join(f"- {r[0]}: {brl(r[1])}" for r in prem[17:35] if r and r[0] and r[1] is not None)

drv_hdr = prem[50]
drv_lines = ["| " + " | ".join(str(h) for h in drv_hdr if h) + " |",
             "|" + "---|" * len([h for h in drv_hdr if h])]
for r in prem[51:56]:
    if not r or not r[0]:
        continue
    cells = [str(r[0])]
    for v in r[1:16]:
        if v is None:
            cells.append("n/d")
        elif isinstance(v, float) and v < 1:
            cells.append(pct(v))
        else:
            cells.append(num(v, 0))
    drv_lines.append("| " + " | ".join(cells) + " |")

cen_mult = "\n".join(
    f"- {r[0]}: multiplicador de crescimento {num(r[1], 2)}x, delta de margem "
    f"{num(r[2] * 100, 1)} p.p., multiplicador de opex variável {num(r[3], 2)}x"
    for r in prem[45:48]
    if r and r[0]
)

write(
    "premissas-e-drivers.md",
    f"""
# Premissas e drivers do modelo — Grupo Meridiano Varejo

Todos os resultados do grupo derivam dos parâmetros abaixo. Alterar qualquer um
deles recalcula demonstração de resultado, balanço, fluxo de caixa e KPIs.

## Parâmetros globais

{globais}

O capex de expansão é uma reforma de lojas de R$ 90.000 no total, distribuída em
R$ 30.000 por mês entre junho e agosto de 2025.

## Saldos de abertura em dezembro de 2024

{aberturas}

## Drivers por unidade de negócio

{chr(10).join(drv_lines)}

## Distribuição de aging de contas a receber e a pagar

A carteira se distribui em 70% a vencer, 15% em 1-30 dias, 8% em 31-60 dias, 4%
em 61-90 dias e 3% acima de 90 dias.

## Multiplicadores de cenário aplicados a 2027

{cen_mult}
""",
)

# --------------------------------------------------------------------------
# 3. DRE consolidado
# --------------------------------------------------------------------------
dre_labels = [
    "Receita Bruta", "(-) Impostos sobre Vendas", "Receita Liquida", "(-) CMV",
    "Lucro Bruto", "Margem Bruta %", "(-) Folha Direta", "(-) Aluguel/Ocupacao",
    "(-) Marketing", "(-) Frete/Logistica", "(-) Outras Desp. Diretas",
    "EBITDA das Unidades", "(-) Overhead Corporativo (G&A)", "EBITDA Consolidado",
    "Margem EBITDA %", "(-) Depreciacao e Amortizacao", "EBIT",
    "(+/-) Resultado Financeiro", "LAIR (Lucro antes IR)", "(-) IR/CSLL",
    "Lucro Liquido", "Margem Liquida %",
]
rb = wide_row("DRE_Consolidado", "Receita Bruta")
eb = wide_row("DRE_Consolidado", "EBITDA Consolidado")
ll = wide_row("DRE_Consolidado", "Lucro Liquido")
me = wide_row("DRE_Consolidado", "Margem EBITDA %")

write(
    "dre-consolidado.md",
    f"""
# Demonstração de resultado consolidada — Grupo Meridiano Varejo

Valores anuais em reais, consolidados para as cinco unidades de negócio.

## Resultado anual de 2025 a 2027

{wide_table("DRE_Consolidado", dre_labels)}

## Números principais de cada ano

- A receita bruta consolidada de 2025 foi de {brl(rb[2025])}.
- A receita bruta consolidada de 2026 foi de {brl(rb[2026])}.
- A receita bruta consolidada de 2027 foi de {brl(rb[2027])}.
- O EBITDA consolidado de 2025 foi de {brl(eb[2025])}, o de 2026 {brl(eb[2026])} e
  o de 2027 {brl(eb[2027])}.
- O lucro líquido de 2025 foi de {brl(ll[2025])}, o de 2026 {brl(ll[2026])} e o de
  2027 {brl(ll[2027])}.
- A margem EBITDA foi de {pct(me[2025])} em 2025, {pct(me[2026])} em 2026 e
  {pct(me[2027])} em 2027.

## Leitura do resultado

A receita bruta cresce de forma consistente nos três anos: {brl(rb[2025])} em
2025, {brl(rb[2026])} em 2026 e {brl(rb[2027])} em 2027. O crescimento acumulado
no período é de {pct(rb[2027] / rb[2025] - 1)}.

O EBITDA consolidado, porém, anda na direção oposta: cai de {brl(eb[2025])} em
2025 para {brl(eb[2026])} em 2026 e {brl(eb[2027])} em 2027. A margem EBITDA
comprime de {pct(me[2025])} para {pct(me[2027])}.

O lucro líquido segue o mesmo caminho, de {brl(ll[2025])} em 2025 para
{brl(ll[2027])} em 2027, uma queda de {pct(1 - ll[2027] / ll[2025])} no período.

Este é o achado central do conjunto de dados: o grupo cresce em faturamento e
encolhe em resultado. O crescimento de receita não se converteu em lucro. A
causa principal está na compressão de margem bruta da unidade Utilidades
Domésticas, detalhada no documento de desempenho por unidade de negócio.
""",
)

# --------------------------------------------------------------------------
# 4. Balanço patrimonial
# --------------------------------------------------------------------------
bal_labels = [
    "Caixa e Equivalentes", "Contas a Receber", "Estoques",
    "Outros Ativos Circulantes", "Ativo Circulante", "Imobilizado Liquido",
    "Intangivel Liquido", "Ativo Nao Circulante", "ATIVO TOTAL",
    "Fornecedores", "Emprestimos Curto Prazo", "Obrigacoes Fiscais",
    "Obrigacoes Trabalhistas", "Outros Passivos Circulantes",
    "Passivo Circulante", "Emprestimos Longo Prazo", "Passivo Nao Circulante",
    "Capital Social", "Lucros Acumulados", "Patrimonio Liquido",
    "PASSIVO + PL TOTAL",
]
caixa = wide_row("Balanco_Consolidado", "Caixa e Equivalentes")
div_cp = wide_row("Balanco_Consolidado", "Emprestimos Curto Prazo")
div_lp = wide_row("Balanco_Consolidado", "Emprestimos Longo Prazo")
chk = wide_row("Balanco_Consolidado", "CHECAGEM (Ativo - Passivo-PL)")
max_chk = max(abs(v) for v in chk["meses"] if v is not None)

write(
    "balanco-patrimonial.md",
    f"""
# Balanço patrimonial consolidado — Grupo Meridiano Varejo

Saldos de fechamento de dezembro de cada ano, em reais.

## Posição patrimonial em 2025, 2026 e 2027

{wide_table("Balanco_Consolidado", bal_labels)}

## Posição de caixa e endividamento

O caixa e equivalentes fecha 2025 em {brl(caixa[2025])}, 2026 em
{brl(caixa[2026])} e 2027 em {brl(caixa[2027])}.

A dívida total ao fim de 2027 soma {brl((div_cp[2027] or 0) + (div_lp[2027] or 0))},
sendo {brl(div_cp[2027])} de curto prazo e {brl(div_lp[2027])} de longo prazo.

O grupo mantém posição de caixa líquido: o caixa supera a dívida total em todos
os anos. A dívida líquida é portanto negativa, e o múltiplo Dívida
Líquida/EBITDA aparece negativo por construção — não é sinal de alavancagem, e
sim de excesso de caixa sobre dívida.

## Integridade do balanço

O balanço fecha exatamente nos 36 meses. A linha de checagem, que calcula Ativo
menos Passivo mais Patrimônio Líquido, tem desvio máximo de {max_chk:.6f} em todo
o período — zero para efeitos práticos.
""",
)

# --------------------------------------------------------------------------
# 5. Fluxo de caixa
# --------------------------------------------------------------------------
dfc_labels = [
    "Lucro Liquido", "(+) Depreciacao e Amortizacao", "(-) Var. Contas a Receber",
    "(-) Var. Estoques", "(-) Var. Outros Ativos Circ.", "(+) Var. Fornecedores",
    "(+) Var. Obrigacoes Fiscais", "(+) Var. Obrigacoes Trabalhistas",
    "(+) Var. Outros Passivos Circ.", "(=) Fluxo de Caixa Operacional",
    "(-) Capex", "(=) Fluxo de Caixa de Investimento",
    "(+/-) Captacao/Amortizacao de Divida", "(-) Dividendos",
    "(=) Fluxo de Caixa de Financiamento", "Variacao de Caixa",
]
capex = wide_row("DFC_Consolidado", "(-) Capex")
capex_meses = capex["meses"]
meses_lbl = [f"{y}-{m:02d}" for y in YEARS for m in range(1, 13)]
pico = sorted(
    ((abs(v or 0), meses_lbl[i]) for i, v in enumerate(capex_meses)), reverse=True
)[:3]

write(
    "fluxo-de-caixa.md",
    f"""
# Demonstração de fluxo de caixa consolidada — Grupo Meridiano Varejo

Método indireto, conciliado linha a linha com o balanço. Totais anuais em reais.

## Fluxo de caixa anual de 2025 a 2027

{wide_table("DFC_Consolidado", dfc_labels)}

## Ciclo de investimento

O capex anual soma {brl(capex[2025])} em 2025, {brl(capex[2026])} em 2026 e
{brl(capex[2027])} em 2027.

Os três meses de maior investimento do período são
{", ".join(f"{m} ({brl(v)})" for v, m in pico)}. Essa concentração corresponde à
reforma de lojas de R$ 90.000 executada entre junho e agosto de 2025, um evento
não recorrente. O capex dos demais meses é capex de manutenção, definido como
1,5% da receita líquida do mês.

## Geração de caixa operacional

O fluxo de caixa operacional é {brl(wide_row("DFC_Consolidado", "(=) Fluxo de Caixa Operacional")[2025])}
em 2025, {brl(wide_row("DFC_Consolidado", "(=) Fluxo de Caixa Operacional")[2026])} em 2026 e
{brl(wide_row("DFC_Consolidado", "(=) Fluxo de Caixa Operacional")[2027])} em 2027.
A pressão sobre a geração operacional vem da deterioração do capital de giro: o
alongamento do prazo médio de recebimento consome caixa à medida que a receita
cresce.
""",
)

# --------------------------------------------------------------------------
# 6. KPIs mensais
# --------------------------------------------------------------------------
kpi_hdr = rows["KPIs_Mensais"][0]
kpi_rows = [r for r in rows["KPIs_Mensais"][1:] if r and r[0]]
keep = ["ChaveMes", "Receita Bruta", "Margem Bruta %", "Margem EBITDA %",
        "Margem Liquida %", "EBITDA", "Lucro Liquido", "DSO (dias)",
        "DIO (dias)", "DPO (dias)", "Ciclo Caixa (dias)", "Liquidez Corrente"]
idx = [kpi_hdr.index(k) for k in keep]
lines = ["| " + " | ".join(keep) + " |", "|" + "---|" * len(keep)]
for r in kpi_rows:
    cells = []
    for k, i in zip(keep, idx):
        v = r[i]
        if v is None:
            cells.append("n/d")
        elif "%" in k:
            cells.append(pct(v))
        elif k in ("Receita Bruta", "EBITDA", "Lucro Liquido"):
            cells.append(brl(v))
        elif k == "ChaveMes":
            cells.append(str(v))
        else:
            cells.append(num(v))
    lines.append("| " + " | ".join(cells) + " |")


def kpi_val(chave, col):
    for r in kpi_rows:
        if r[kpi_hdr.index("ChaveMes")] == chave:
            return r[kpi_hdr.index(col)]
    return None


write(
    "kpis-mensais.md",
    f"""
# Indicadores mensais — Grupo Meridiano Varejo

Vinte indicadores por mês, de janeiro de 2025 a dezembro de 2027. A tabela
abaixo traz os principais.

## Série mensal de indicadores

{chr(10).join(lines)}

## Capital de giro e ciclo de caixa

O prazo médio de recebimento, o DSO, sobe de {num(kpi_val("2025-01", "DSO (dias)"))} dias
em janeiro de 2025 para {num(kpi_val("2027-12", "DSO (dias)"))} dias em dezembro de
2027. O ciclo de conversão de caixa acompanha, indo de
{num(kpi_val("2025-01", "Ciclo Caixa (dias)"))} dias para
{num(kpi_val("2027-12", "Ciclo Caixa (dias)"))} dias.

Esse alongamento é a deterioração do capital de giro: o grupo demora mais para
receber dos clientes enquanto o volume cresce, e a diferença sai do caixa
operacional. É a segunda causa da queda de geração de caixa, ao lado da
compressão de margem.

## Liquidez e cobertura

A liquidez corrente parte de {num(kpi_val("2025-01", "Liquidez Corrente"), 2)} em
janeiro de 2025 e fecha 2027 em {num(kpi_val("2027-12", "Liquidez Corrente"), 2)}.
A dívida líquida é negativa em todo o período, refletindo a posição de caixa
líquido do grupo.
""",
)

# --------------------------------------------------------------------------
# 7. Desempenho por unidade de negócio
# --------------------------------------------------------------------------
un_hdr = rows["DRE_por_UN"][0]
un_rows = [r for r in rows["DRE_por_UN"][1:] if r and r[0]]
ci = {c: un_hdr.index(c) for c in un_hdr if c}

agg = {}
for r in un_rows:
    key = (r[ci["Unidade"]], r[ci["Ano"]])
    a = agg.setdefault(key, dict.fromkeys(
        ["Receita Bruta", "Receita Liquida", "CMV", "Lucro Bruto",
         "EBITDA da Unidade", "Marketing", "Folha Direta"], 0.0))
    for k in a:
        a[k] += r[ci[k]] or 0.0

un_lines = ["| Unidade | Ano | Receita Bruta | Receita Líquida | Lucro Bruto | Margem Bruta | EBITDA | Margem EBITDA |",
            "|---|---|---|---|---|---|---|---|"]
for u in UNITS:
    for y in YEARS:
        a = agg.get((u, y))
        if not a:
            continue
        mb = a["Lucro Bruto"] / a["Receita Liquida"] if a["Receita Liquida"] else 0
        mE = a["EBITDA da Unidade"] / a["Receita Liquida"] if a["Receita Liquida"] else 0
        un_lines.append(
            f"| {u} | {y} | {brl(a['Receita Bruta'])} | {brl(a['Receita Liquida'])} | "
            f"{brl(a['Lucro Bruto'])} | {pct(mb)} | {brl(a['EBITDA da Unidade'])} | {pct(mE)} |"
        )

ud = {y: agg[("Utilidades Domesticas", y)] for y in YEARS}
ud_mb = {y: ud[y]["Lucro Bruto"] / ud[y]["Receita Liquida"] for y in YEARS}
rank25 = sorted(UNITS, key=lambda u: -agg[(u, 2025)]["Receita Bruta"])

# Per-unit drivers, read from Premissas rather than restated by hand.
drv = {}
for r in prem[51:56]:
    if r and r[0]:
        drv[r[0]] = {
            "mb": [r[1], r[2], r[3]],
            "dso": [r[11], r[12], r[13]],
            "dio": r[14],
            "dpo": r[15],
        }


def mb(u, y):
    return agg[(u, y)]["Lucro Bruto"] / agg[(u, y)]["Receita Liquida"]


# The superlatives the evaluation asks for, computed rather than asserted. A 3B
# model quotes a sentence far more reliably than it scans a table, so each of
# these gets stated outright instead of being left implicit in the rows above.
maior_receita = max(UNITS, key=lambda u: agg[(u, 2025)]["Receita Bruta"])
menor_receita = min(UNITS, key=lambda u: agg[(u, 2025)]["Receita Bruta"])
maior_margem = max(UNITS, key=lambda u: mb(u, 2025))
menor_margem = min(UNITS, key=lambda u: mb(u, 2025))
maior_ebitda = max(UNITS, key=lambda u: agg[(u, 2025)]["EBITDA da Unidade"])
dso_mais_longo = max(UNITS, key=lambda u: drv[u]["dso"][2])
dso_mais_curto = min(UNITS, key=lambda u: drv[u]["dso"][2])

perfis = []
for u in UNITS:
    d = drv[u]
    perfis.append(
        f"**{u}.** Receita bruta de {brl(agg[(u, 2025)]['Receita Bruta'])} em 2025 e "
        f"{brl(agg[(u, 2027)]['Receita Bruta'])} em 2027. Margem bruta de {pct(mb(u, 2025))} "
        f"em 2025 e {pct(mb(u, 2027))} em 2027. EBITDA de {brl(agg[(u, 2025)]['EBITDA da Unidade'])} "
        f"em 2025 e {brl(agg[(u, 2027)]['EBITDA da Unidade'])} em 2027. Prazo de recebimento de "
        f"{num(d['dso'][0], 0)} dias em 2025 e {num(d['dso'][2], 0)} dias em 2027, prazo de "
        f"estocagem de {num(d['dio'], 0)} dias e prazo de pagamento de {num(d['dpo'], 0)} dias."
    )

write(
    "desempenho-por-unidade.md",
    f"""
# Desempenho por unidade de negócio — Grupo Meridiano Varejo

Resultado até EBITDA por unidade e por ano, agregado a partir da série mensal.

## Resultado anual por unidade

{chr(10).join(un_lines)}

## Comparação entre as unidades

Em 2025, por receita bruta, a ordem das unidades é:
{", ".join(f"{i + 1}. {u} ({brl(agg[(u, 2025)]['Receita Bruta'])})" for i, u in enumerate(rank25))}.

A unidade de maior receita do grupo é {maior_receita}, com
{brl(agg[(maior_receita, 2025)]["Receita Bruta"])} em 2025. A de menor receita é
{menor_receita}, com {brl(agg[(menor_receita, 2025)]["Receita Bruta"])}.

A unidade com a maior margem bruta do grupo é {maior_margem}, com
{pct(mb(maior_margem, 2025))} em 2025 e {pct(mb(maior_margem, 2027))} em 2027.
A unidade com a menor margem bruta é {menor_margem}, com {pct(mb(menor_margem, 2025))}
em 2025 e {pct(mb(menor_margem, 2027))} em 2027.

A unidade com o maior EBITDA é {maior_ebitda}, com
{brl(agg[(maior_ebitda, 2025)]["EBITDA da Unidade"])} em 2025.

A unidade com o prazo de recebimento mais alongado é {dso_mais_longo}, com DSO de
{num(drv[dso_mais_longo]["dso"][0], 0)} dias em 2025 subindo para
{num(drv[dso_mais_longo]["dso"][2], 0)} dias em 2027. A de prazo mais curto é
{dso_mais_curto}, com {num(drv[dso_mais_curto]["dso"][2], 0)} dias em 2027.

## Perfil de cada unidade de negócio

{(chr(10) + chr(10)).join(perfis)}

## Compressão de margem em Utilidades Domésticas

Utilidades Domésticas é a segunda maior unidade por receita, atrás de Vestuário,
e é onde está o problema de rentabilidade. A margem bruta da unidade cai de
{pct(ud_mb[2025])} em 2025 para {pct(ud_mb[2026])} em 2026 e {pct(ud_mb[2027])}
em 2027 — uma perda de {num((ud_mb[2025] - ud_mb[2027]) * 100)} pontos
percentuais em três anos.

Utilidades Domésticas é a única unidade cujo lucro bruto cai em valor absoluto no
período, de {brl(ud[2025]["Lucro Bruto"])} em 2025 para
{brl(ud[2027]["Lucro Bruto"])} em 2027, mesmo com a receita crescendo. Todas as
outras quatro unidades aumentam o lucro bruto. Por isso a compressão de margem
desta unidade é o principal motor isolado da queda de rentabilidade consolidada:
ela drena o resultado enquanto as demais o sustentam.

As demais unidades mantêm margem bruta estável no período: Materiais de
Construção em torno de 30%, Papelaria e Impressão em torno de 42%, Serviços
Automotivos entre 46% e 47%, e Vestuário entre 52% e 53%.

## Sazonalidade de Vestuário

Vestuário concentra vendas em novembro e dezembro. Os meses de janeiro são
fracos em todas as unidades e podem gerar prejuízo pontual no consolidado.
""",
)

# --------------------------------------------------------------------------
# 8. Capital de giro
# --------------------------------------------------------------------------
def agg_tidy(sheet, value_cols):
    hdr = rows[sheet][0]
    ci2 = {c: hdr.index(c) for c in hdr if c}
    out = {}
    for r in rows[sheet][1:]:
        if not r or not r[0]:
            continue
        key = (r[ci2["Unidade"]], r[ci2["Ano"]])
        a = out.setdefault(key, dict.fromkeys(value_cols, 0.0))
        for k in value_cols:
            a[k] += r[ci2[k]] or 0.0
    return out


ar = agg_tidy("Contas_Receber", ["AR Total", "> 90 dias"])
ap = agg_tidy("Contas_Pagar", ["AP Total"])
est = agg_tidy("Estoque", ["Estoque Final", "CMV do Mes"])

cg_lines = ["| Unidade | Ano | Contas a Receber (média mensal) | Contas a Pagar (média mensal) | Estoque (média mensal) |",
            "|---|---|---|---|---|"]
for u in UNITS:
    for y in YEARS:
        cg_lines.append(
            f"| {u} | {y} | {brl(ar[(u, y)]['AR Total'] / 12)} | "
            f"{brl(ap[(u, y)]['AP Total'] / 12)} | {brl(est[(u, y)]['Estoque Final'] / 12)} |"
        )

write(
    "capital-de-giro.md",
    f"""
# Capital de giro — contas a receber, contas a pagar e estoque

Saldos médios mensais por unidade de negócio e por ano, em reais.

## Saldos médios por unidade

{chr(10).join(cg_lines)}

## Prazos por unidade

A unidade com o prazo de recebimento mais alongado do grupo é {dso_mais_longo},
com DSO de {num(drv[dso_mais_longo]["dso"][0], 0)} dias em 2025 e
{num(drv[dso_mais_longo]["dso"][2], 0)} dias em 2027. A unidade com o prazo mais
curto é {dso_mais_curto}, com {num(drv[dso_mais_curto]["dso"][2], 0)} dias em 2027.

Prazos de recebimento (DSO), estocagem (DIO) e pagamento (DPO) por unidade, em
dias:

{chr(10).join([
    "| Unidade | DSO 2025 | DSO 2026 | DSO 2027 | DIO | DPO |",
    "|---|---|---|---|---|---|",
] + [
    f"| {u} | {num(drv[u]['dso'][0], 0)} | {num(drv[u]['dso'][1], 0)} | "
    f"{num(drv[u]['dso'][2], 0)} | {num(drv[u]['dio'], 0)} | {num(drv[u]['dpo'], 0)} |"
    for u in UNITS
])}

Utilidades Domésticas é a unidade que mais se deteriora em termos relativos:
o DSO mais que dobra, de {num(drv["Utilidades Domesticas"]["dso"][0], 0)} para
{num(drv["Utilidades Domesticas"]["dso"][2], 0)} dias. Ainda assim permanece
abaixo do prazo de {dso_mais_longo}, que é o mais longo do grupo em termos
absolutos.

## Aging da carteira

Tanto contas a receber quanto contas a pagar seguem a mesma distribuição de
aging: 70% a vencer, 15% em 1 a 30 dias, 8% em 31 a 60 dias, 4% em 61 a 90 dias
e 3% acima de 90 dias.
""",
)

# --------------------------------------------------------------------------
# 9. Pessoal e folha
# --------------------------------------------------------------------------
hc_hdr = rows["Headcount_Folha"][0]
hci = {c: hc_hdr.index(c) for c in hc_hdr if c}
hc = {}
for r in rows["Headcount_Folha"][1:]:
    if not r or not r[0]:
        continue
    key = (r[hci["Unidade"]], r[hci["Ano"]])
    a = hc.setdefault(key, {"Headcount": [], "Folha Total": 0.0, "Custo Medio (R$)": []})
    a["Headcount"].append(r[hci["Headcount"]] or 0)
    a["Folha Total"] += r[hci["Folha Total"]] or 0.0
    a["Custo Medio (R$)"].append(r[hci["Custo Medio (R$)"]] or 0)

hc_lines = ["| Unidade | Ano | Headcount médio | Custo médio mensal | Folha anual (com encargos) |",
            "|---|---|---|---|---|"]
for u in UNITS:
    for y in YEARS:
        a = hc[(u, y)]
        hc_lines.append(
            f"| {u} | {y} | {num(sum(a['Headcount']) / len(a['Headcount']))} | "
            f"{brl(sum(a['Custo Medio (R$)']) / len(a['Custo Medio (R$)']))} | {brl(a['Folha Total'])} |"
        )

write(
    "pessoal-e-folha.md",
    f"""
# Quadro de pessoal e folha — Grupo Meridiano Varejo

Headcount e folha com encargos por unidade de negócio e ano. Os encargos são 68%
sobre os salários.

## Headcount e folha por unidade

{chr(10).join(hc_lines)}

## Folha consolidada

A folha direta consolidada soma {brl(abs(wide_row("DRE_Consolidado", "(-) Folha Direta")[2025]))}
em 2025, {brl(abs(wide_row("DRE_Consolidado", "(-) Folha Direta")[2026]))} em 2026 e
{brl(abs(wide_row("DRE_Consolidado", "(-) Folha Direta")[2027]))} em 2027.

A folha cresce mais rápido que a receita líquida no período, o que contribui para
a compressão do EBITDA. As provisões trabalhistas correspondem a 40% da folha do
mês.
""",
)

# --------------------------------------------------------------------------
# 10. Cenários e sensibilidade
# --------------------------------------------------------------------------
cen = rows["Cenarios"]
cen_lines = ["| Linha (2027) | Conservador | Base | Otimista |", "|---|---|---|---|"]
for r in cen[4:20]:
    if not r or not r[0]:
        continue
    f = pct if "%" in str(r[0]) else (num if "EBITDA" in str(r[0]) and "/" in str(r[0]) else brl)
    cen_lines.append(f"| {r[0]} | {f(r[1])} | {f(r[2])} | {f(r[3])} |")

sens = rows["Sensibilidade"]
gridA = ["| " + " | ".join(str(c) for c in sens[4] if c is not None) + " |",
         "|" + "---|" * len([c for c in sens[4] if c is not None])]
for r in sens[5:10]:
    gridA.append("| " + str(r[0]) + " | " + " | ".join(brl(v) for v in r[1:6]) + " |")
gridB = ["| " + " | ".join(str(c) for c in sens[13] if c is not None) + " |",
         "|" + "---|" * len([c for c in sens[13] if c is not None])]
for r in sens[14:19]:
    gridB.append("| " + str(r[0]) + " | " + " | ".join(brl(v) for v in r[1:6]) + " |")

write(
    "cenarios-e-sensibilidade.md",
    f"""
# Cenários e sensibilidade para 2027 — Grupo Meridiano Varejo

O caso Base reproduz o modelo mensal. Os cenários Conservador e Otimista aplicam
multiplicadores sobre 2026 e são sobreposições analíticas: não alteram o modelo
mensal principal.

## Resultado de 2027 nos três cenários

{chr(10).join(cen_lines)}

No cenário Conservador o grupo dá prejuízo em 2027, com lucro líquido de
{brl(cen[14][1])}. No Base o lucro é {brl(cen[14][2])} e no Otimista
{brl(cen[14][3])}. A distância entre o pior e o melhor caso é de
{brl(cen[14][3] - cen[14][1])}, o que mostra o quanto o resultado de 2027 é
sensível às premissas.

## Sensibilidade do EBITDA de 2027

Grid variando crescimento de receita contra delta de margem bruta. Valores em
reais.

{chr(10).join(gridA)}

A margem bruta domina o resultado. Mover a margem em 3 pontos percentuais altera
o EBITDA muito mais do que mover o crescimento de receita em 6 pontos
percentuais. Em outras palavras: para este grupo, defender margem vale mais do
que vender mais.

## Sensibilidade do lucro líquido de 2027

Grid variando delta de margem bruta contra multiplicador de custos operacionais.
Valores em reais.

{chr(10).join(gridB)}

A combinação de margem 3 pontos percentuais abaixo do base com custos
operacionais 10% acima leva o lucro líquido a {brl(sens[14][5])}, o pior canto do
grid.
""",
)

# --------------------------------------------------------------------------
# 11. Custos e margens por produto
# --------------------------------------------------------------------------
prod = rows["CMV_Custos_Produto"]
prod_lines = ["| " + " | ".join(str(c) for c in prod[3] if c) + " |",
              "|" + "---|" * len([c for c in prod[3] if c])]
for r in prod[4:]:
    if not r or not r[0]:
        continue
    prod_lines.append(
        f"| {r[0]} | {r[1]} | {brl(r[2], 2)} | {brl(r[3], 2)} | {brl(r[4], 2)} | {pct(r[5])} |"
    )

write(
    "custos-e-margens-por-produto.md",
    f"""
# Custos e margens por produto — Grupo Meridiano Varejo

Preços médios reais extraídos das notas fiscais. O custo unitário e a margem são
derivados da margem bruta de 2025 do respectivo segmento.

## Referência de preço, custo e margem

{chr(10).join(prod_lines)}
""",
)

# --------------------------------------------------------------------------
# 12. Vendas por representante
# --------------------------------------------------------------------------
rep = rows["Vendas_Representantes"]
rep_hdr = rep[7]
# The summary block ends at the TOTAL / MEDIA row, which carries no ranking.
# A separate cadastro block follows and must not be mixed in.
rep_rows = [r for r in rep[8:20] if r and r[0] and r[12] is not None]
total_rep = rep[20]
cad_hdr, cad_rows = rep[23], [r for r in rep[24:36] if r and r[0]]
rep_lines = ["| " + " | ".join(str(c) for c in rep_hdr if c) + " |",
             "|" + "---|" * len([c for c in rep_hdr if c])]
for r in rep_rows:
    rep_lines.append(
        f"| {r[0]} | {r[1]} | {brl(r[2])} | {brl(r[3])} | {pct(r[4])} | {brl(r[5])} | "
        f"{brl(r[6])} | {pct(r[7])} | {brl(r[8])} | {brl(r[9])} | {pct(r[10])} | "
        f"{pct(r[11])} | {int(r[12])} |"
    )

ranked = sorted(rep_rows, key=lambda r: r[12])
acima = [r for r in rep_rows if r[11] >= 1.0]
abaixo = [r for r in rep_rows if r[11] < 1.0]

write(
    "vendas-por-representante.md",
    f"""
# Performance de vendas por representante — Grupo Meridiano Varejo

Metas contra realizado por representante e segmento, de janeiro de 2025 a
dezembro de 2027. As metas crescem 6% ao ano.

## Resumo anual por representante

{chr(10).join(rep_lines)}

No consolidado do grupo, o realizado soma {brl(total_rep[2])} em 2025 contra meta
de {brl(total_rep[3])}, atingimento de {pct(total_rep[4])}. Em 2027 o realizado é
{brl(total_rep[8])} contra meta de {brl(total_rep[9])}, atingimento de
{pct(total_rep[10])}. No acumulado de três anos o grupo atinge {pct(total_rep[11])}
da meta.

## Cadastro de representantes

{chr(10).join([
    "| " + " | ".join(str(c) for c in cad_hdr if c) + " |",
    "|" + "---|" * len([c for c in cad_hdr if c]),
] + [
    f"| {r[0]} | {r[1]} | {r[2]} | {pct(r[3])} | {brl(r[4])} | {brl(r[5])} | "
    f"{brl(r[6])} | {num(r[7], 2)} |"
    for r in cad_rows
])}

## Leitura da performance comercial

{len(rep_rows)} representantes cobrem os cinco segmentos. Destes,
{len(acima)} bateram a meta no acumulado de três anos e {len(abaixo)} ficaram
abaixo.

Os três melhores no acumulado de três anos são
{", ".join(f"{r[0]} ({r[1]}, {pct(r[11])})" for r in ranked[:3])}.

Os três piores são
{", ".join(f"{r[0]} ({r[1]}, {pct(r[11])})" for r in ranked[-3:])}.

O segmento de Papelaria e Impressão concentra os melhores desempenhos relativos,
enquanto Utilidades Domésticas — justamente a unidade de maior volume e de maior
compressão de margem — tem representantes entre os que mais ficam abaixo da meta.
""",
)

print("\nOK")
