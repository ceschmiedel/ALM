"""Rule-based decision generators in Brazilian Portuguese.

Public Portuguese data teaches a decision model to read: entailment, sentiment,
intent.  It does not teach what business decisions mostly are — applying a
stated rule to stated facts.  These generators fill that gap with records whose
label is *computed*: each draws random facts, states a rule, and derives the
answer by arithmetic, so no label is ever a model's opinion.

They cover the skills, not the domain.  A decision model should learn to compare
two figures, place a change in a band, apply a threshold, count and follow a
trend wherever those appear — logistics, HR, support, credit — so a benchmark
over a different domain (the Meridiano financial model) measures transfer
rather than recall.  Nothing here uses that workbook.

Every generator takes a ``random.Random`` and returns one labelled request.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

Record = dict[str, Any]

_MONTHS = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
           "setembro", "outubro", "novembro", "dezembro"]
_PEOPLE = ["Ana Lima", "Bruno Rocha", "Camila Dias", "Daniel Farias", "Elisa Mota", "Fábio Reis",
           "Gabriela Nunes", "Hugo Barros", "Isabela Cunha", "João Paiva", "Karina Alves", "Lucas Teles"]
_REGIONS = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
_DEPOTS = ["Recife", "Campinas", "Curitiba", "Goiânia", "Manaus", "Salvador", "Joinville", "Belém"]
_ITEMS = ["parafusos", "cabos de rede", "toner", "filtros de ar", "embalagens", "luvas térmicas",
          "pneus", "baterias", "rolamentos", "etiquetas"]


def money(value: float) -> str:
    text = f"{abs(value):,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return f"-R$ {text}" if value < 0 else f"R$ {text}"


def integer(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def percent(value: float, decimals: int = 0) -> str:
    return f"{value * 100:.{decimals}f}".replace(".", ",") + "%"


def _record(state: Any, qid: str, question: dict[str, Any], family: str) -> Record:
    question["src"] = f"synth_pt/{family}"
    return {"state": state, "questions": {qid: question}}


# -- comparison ---------------------------------------------------------------


def comparison(rng: random.Random) -> Record:
    """Is one figure larger (or smaller) than another?"""
    subject, unit = rng.choice(
        [("entregas concluídas", integer), ("chamados abertos", integer), ("custo de frete", money),
         ("horas extras", integer), ("faturamento", money), ("devoluções", integer)]
    )
    a_name, b_name = rng.sample(_DEPOTS + _REGIONS, 2)
    a = rng.uniform(200, 90000)
    b = a * rng.choice([rng.uniform(0.55, 0.97), rng.uniform(1.03, 1.7)])
    direction = rng.choice(["maior", "menor"])
    truth = a > b if direction == "maior" else a < b
    month = rng.choice(_MONTHS)
    lines = [(a_name, a), (b_name, b)]
    extra = rng.choice([n for n in _DEPOTS if n not in (a_name, b_name)])
    lines.append((extra, a * rng.uniform(0.5, 1.6)))
    rng.shuffle(lines)
    state = f"{subject.capitalize()} em {month}:\n" + "\n".join(f"- {n}: {unit(v)}" for n, v in lines)
    return _record(
        state, "comparacao",
        {"type": "noul",
         "instructions": f"O valor de {a_name} foi {direction} que o de {b_name}?",
         "label": truth},
        "comparacao",
    )


# -- band of a change ---------------------------------------------------------

_BANDS = [
    ("Queda: o valor diminuiu", "Estável: aumentou até {a}", "Alta moderada: aumentou mais de {a} e até {b}",
     "Alta forte: aumentou mais de {b}"),
    ("Redução", "Variação pequena, de até {a}", "Crescimento entre {a} e {b}", "Crescimento acima de {b}"),
]


def change_band(rng: random.Random) -> Record:
    """Place a before/after change in one of four bands."""
    low, high = rng.choice([(0.05, 0.15), (0.10, 0.25), (0.03, 0.10), (0.08, 0.20)])
    level = rng.randrange(4)
    growth = [rng.uniform(-0.4, -0.02), rng.uniform(0.005, low - 0.01),
              rng.uniform(low + 0.01, high - 0.01), rng.uniform(high + 0.01, high + 0.6)][level]
    before = rng.uniform(800, 400000)
    after = before * (1 + growth)
    subject = rng.choice(["Volume expedido", "Custo de manutenção", "Número de atendimentos",
                          "Valor da carteira", "Consumo de energia"])
    fmt = money if "Custo" in subject or "Valor" in subject else integer
    period = rng.choice([("mês anterior", "mês atual"), ("ano passado", "este ano"),
                         ("primeiro semestre", "segundo semestre")])
    template = rng.choice(_BANDS)
    levels = [t.format(a=percent(low), b=percent(high)) for t in template]
    return _record(
        f"{subject}:\n- {period[0]}: {fmt(before)}\n- {period[1]}: {fmt(after)}", "faixa",
        {"type": "score",
         "instructions": f"Classifique a variação de {subject.lower()} entre os dois períodos.",
         "criteria": levels, "label": level},
        "faixa_variacao",
    )


# -- threshold rules ----------------------------------------------------------


def reorder(rng: random.Random) -> Record:
    """Stock rule: reorder when cover falls below a minimum number of days."""
    item = rng.choice(_ITEMS)
    daily = rng.randint(5, 400)
    minimum = rng.choice([5, 7, 10, 15, 20])
    days = rng.choice([rng.uniform(1, minimum - 0.6), rng.uniform(minimum + 0.6, minimum * 3)])
    stock = int(daily * days)
    state = {
        "regra": f"Emitir pedido de reposição quando o estoque cobrir menos de {minimum} dias de consumo.",
        "item": item,
        "estoque_atual_unidades": stock,
        "consumo_medio_diario_unidades": daily,
    }
    return _record(
        state, "repor",
        {"type": "noul", "instructions": "Pela regra, é preciso emitir pedido de reposição agora?",
         "label": stock / daily < minimum},
        "reposicao",
    )


def expense_approval(rng: random.Random) -> Record:
    """Approval ladder by amount, with a hard block."""
    a, b = rng.choice([(500, 5000), (1000, 10000), (2000, 20000), (300, 3000)])
    amount = rng.choice([rng.uniform(20, a * 0.95), rng.uniform(a * 1.05, b * 0.95),
                         rng.uniform(b * 1.05, b * 2.5)])
    # One request in four exceeds what is left in the budget.
    budget_left = amount * (rng.uniform(0.3, 0.95) if rng.random() < 0.25 else rng.uniform(1.1, 4.0))
    criteria = {
        "coordenador": f"Despesa de até {money(a)}, dentro do saldo do orçamento",
        "gerente": f"Despesa acima de {money(a)} e até {money(b)}, dentro do saldo do orçamento",
        "diretoria": f"Despesa acima de {money(b)}, dentro do saldo do orçamento",
        "recusada": "Despesa maior que o saldo disponível no orçamento: não pode ser aprovada",
    }
    label = ("recusada" if amount > budget_left else "coordenador" if amount <= a
             else "gerente" if amount <= b else "diretoria")
    state = (f"Solicitação de despesa de {rng.choice(_PEOPLE)}.\n"
             f"Valor solicitado: {money(amount)}\nSaldo disponível no orçamento do centro de custo: {money(budget_left)}")
    return _record(
        state, "aprovador",
        {"type": "choice", "instructions": "Quem deve aprovar esta despesa, segundo a política de alçadas?",
         "criteria": criteria, "label": label},
        "alcada_despesa",
    )


def sla(rng: random.Random) -> Record:
    """Was a ticket resolved inside the deadline for its priority?"""
    limits = {"crítica": rng.choice([2, 4]), "alta": rng.choice([8, 12]), "normal": rng.choice([24, 48])}
    priority = rng.choice(list(limits))
    limit = limits[priority]
    hours = rng.choice([rng.uniform(0.3, limit * 0.9), rng.uniform(limit * 1.1, limit * 2.5)])
    state = ("Prazos de solução por prioridade: " + "; ".join(f"{p}: {h} horas" for p, h in limits.items()) + ".\n\n"
             f"Chamado de prioridade {priority}, resolvido {str(round(hours, 1)).replace('.', ',')} horas após a abertura.")
    return _record(
        state, "no_prazo",
        {"type": "noul", "instructions": "O chamado foi resolvido dentro do prazo da sua prioridade?",
         "criteria": {"true": "O tempo de solução não ultrapassou o prazo", "false": "O tempo de solução ultrapassou o prazo"},
         "label": hours <= limit},
        "prazo_chamado",
    )


def eligibility(rng: random.Random) -> Record:
    """Several conditions must all hold."""
    min_months = rng.choice([6, 12, 24])
    max_absences = rng.choice([2, 3, 5])
    min_score = rng.choice([70, 75, 80])
    # Start from a profile that passes, then break one condition half the time,
    # so a failure usually hinges on a single fact.
    months = rng.randint(min_months, 60)
    absences = rng.randint(0, max_absences)
    score = rng.randint(min_score, 100)
    if rng.random() < 0.5:
        broken = rng.randrange(3)
        if broken == 0:
            months = rng.randint(1, min_months - 1)
        elif broken == 1:
            absences = rng.randint(max_absences + 1, max_absences + 5)
        else:
            score = rng.randint(40, min_score - 1)
    state = (f"Regra do programa de bônus: pelo menos {min_months} meses de casa, no máximo {max_absences} "
             f"faltas no ano e nota de desempenho de {min_score} ou mais.\n\n"
             f"Funcionário: {rng.choice(_PEOPLE)}\nTempo de casa: {months} meses\nFaltas no ano: {absences}\n"
             f"Nota de desempenho: {score}")
    return _record(
        state, "elegivel",
        {"type": "noul", "instructions": "O funcionário cumpre todas as condições para receber o bônus?",
         "label": months >= min_months and absences <= max_absences and score >= min_score},
        "elegibilidade",
    )


# -- counting and extremes ----------------------------------------------------


def count_above(rng: random.Random) -> Record:
    """How many of four entries exceed a limit?"""
    names = rng.sample(_DEPOTS, 4)
    values = [rng.randint(40, 990) for _ in names]
    target = rng.randrange(5)
    ordered = sorted(values, reverse=True)
    if target == 0:
        limit = ordered[0] + rng.randint(5, 60)
    elif target == 4:
        limit = max(1, ordered[-1] - rng.randint(5, 30))
    else:
        if ordered[target - 1] - ordered[target] < 2:
            return count_above(rng)
        limit = (ordered[target - 1] + ordered[target]) // 2
    count = sum(v > limit for v in values)
    state = "Pedidos em atraso por centro de distribuição:\n" + "\n".join(
        f"- {n}: {v}" for n, v in zip(names, values, strict=True))
    return _record(
        state, "quantos",
        {"type": "score",
         "instructions": f"Quantos centros de distribuição têm mais de {limit} pedidos em atraso?",
         "criteria": ["Nenhum", "Um", "Dois", "Três", "Quatro"], "label": count},
        "contagem",
    )


def extreme(rng: random.Random) -> Record:
    """Which entry has the largest or smallest value?"""
    metric, fmt = rng.choice([("tempo médio de entrega (dias)", lambda v: str(round(v, 1)).replace(".", ",")),
                              ("custo por pedido", money), ("taxa de devolução", lambda v: percent(v / 1000, 1)),
                              ("pedidos processados", integer)])
    names = rng.sample(_DEPOTS, rng.randint(3, 6))
    values = rng.sample(range(11, 990), len(names))
    direction, pick = rng.choice([("maior", max), ("menor", min)])
    best = pick(range(len(names)), key=values.__getitem__)
    slug = lambda n: n.lower().replace("ã", "a").replace("é", "e").replace("â", "a")  # noqa: E731
    state = f"{metric.capitalize()} por centro de distribuição:\n" + "\n".join(
        f"- {n}: {fmt(v)}" for n, v in zip(names, values, strict=True))
    return _record(
        state, "centro",
        {"type": "choice",
         "instructions": f"Qual centro de distribuição tem o {direction} valor de {metric}?",
         "criteria": {slug(n): (n if rng.random() < 0.6 else None) for n in names},
         "label": slug(names[best])},
        "extremo",
    )


def trend(rng: random.Random) -> Record:
    """Does a short series move in one direction throughout?"""
    start = rng.uniform(100, 5000)
    rising = rng.random() < 0.5
    steps = [rng.uniform(0.03, 0.2) for _ in range(rng.choice([2, 3]))]
    if not rising:
        steps[rng.randrange(len(steps))] *= -1
    values = [start]
    for step in steps:
        values.append(values[-1] * (1 + step))
    direction = rng.choice(["cresceu", "caiu"])
    if direction == "caiu":
        values = values[::-1]
        truth = all(b < a for a, b in zip(values, values[1:], strict=False))
    else:
        truth = all(b > a for a, b in zip(values, values[1:], strict=False))
    first = rng.randrange(0, 12 - len(values))
    state = "Tempo de fila no atendimento (segundos):\n" + "\n".join(
        f"- {_MONTHS[first + i]}: {integer(v)}" for i, v in enumerate(values))
    return _record(
        state, "tendencia",
        {"type": "noul",
         "instructions": f"O valor {direction} em todos os meses, cada mês em relação ao anterior?",
         "label": truth},
        "tendencia",
    )


# -- routing ------------------------------------------------------------------

_QUEUES = {
    "financeiro": ("Cobranças, boletos, notas fiscais e reembolsos",
                   ["Recebi um boleto em duplicidade referente a {m}.", "Preciso da segunda via da nota fiscal de {m}.",
                    "O reembolso do pedido cancelado ainda não caiu na conta.", "A cobrança de {m} veio com valor errado."]),
    "logistica": ("Entregas, prazos de envio, rastreamento e devoluções de mercadoria",
                  ["Meu pedido saiu de {d} e não chegou até hoje.", "O código de rastreamento não mostra nenhuma atualização.",
                   "Quero devolver um produto que chegou com a caixa amassada.", "Qual o prazo de entrega para a região {r}?"]),
    "suporte_tecnico": ("Erros no sistema, acesso, senhas e falhas de equipamento",
                        ["Não consigo entrar no sistema desde ontem, a senha não é aceita.", "O aplicativo fecha sozinho ao abrir o relatório.",
                         "A impressora do setor parou de responder.", "Aparece erro 500 ao salvar o cadastro."]),
    "recursos_humanos": ("Férias, folha de pagamento, benefícios e contratações",
                         ["Gostaria de marcar minhas férias para {m}.", "Meu vale-refeição não foi creditado este mês.",
                          "Como faço para incluir um dependente no plano de saúde?", "O holerite de {m} veio sem as horas extras."]),
    "comercial": ("Propostas, preços, descontos e novos contratos",
                  ["Podem enviar uma proposta para 200 unidades?", "Existe desconto para pagamento à vista?",
                   "Queremos renovar o contrato por mais um ano.", "Qual o preço para revenda na região {r}?"]),
}


def routing(rng: random.Random) -> Record:
    """Send a message to the team that handles it."""
    keep = rng.sample(list(_QUEUES), rng.randint(3, 5))
    queue = rng.choice(keep)
    message = rng.choice(_QUEUES[queue][1]).format(m=rng.choice(_MONTHS), d=rng.choice(_DEPOTS), r=rng.choice(_REGIONS))
    state: Any = message if rng.random() < 0.7 else {"canal": rng.choice(["e-mail", "chat", "formulário"]), "mensagem": message}
    return _record(
        state, "fila",
        {"type": "choice", "instructions": "Qual equipe deve tratar esta mensagem?",
         "criteria": {q: _QUEUES[q][0] for q in keep}, "label": queue},
        "roteamento",
    )


GENERATORS: dict[str, Callable[[random.Random], Record]] = {
    "comparacao": comparison,
    "faixa_variacao": change_band,
    "reposicao": reorder,
    "alcada_despesa": expense_approval,
    "prazo_chamado": sla,
    "elegibilidade": eligibility,
    "contagem": count_above,
    "extremo": extreme,
    "tendencia": trend,
    "roteamento": routing,
}


def generate(n: int, rng: random.Random) -> list[Record]:
    """``n`` records, cycling through the families so each gets an equal share."""
    families = list(GENERATORS)
    return [GENERATORS[families[i % len(families)]](rng) for i in range(n)]
