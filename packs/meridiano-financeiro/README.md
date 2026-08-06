# meridiano-financeiro

Federação de FP&A sobre o modelo financeiro de três demonstrações do **Grupo
Meridiano Varejo Ltda**, entidade sintética com cinco unidades de negócio e 36
meses de histórico (jan/2025 a dez/2027).

Diferente do `demo-enterprise`, este pack roda com **modelos reais** — Ollama
local, sem API key, nada saindo da máquina.

```bash
alm pack install packs/meridiano-financeiro
alm ask "A receita cresceu mas o lucro caiu. Explique o que aconteceu." --trace
```

## Origem do corpus

Os 12 documentos em `corpus/finance/` são **gerados**, não escritos à mão. A
fonte é a planilha de 16 abas em `dataset/Grupo_Demo_Dataset_Financeiro.xlsx`,
na raiz do repositório. Para regenerar:

```bash
python -m venv /tmp/xlsxenv && /tmp/xlsxenv/bin/pip install openpyxl
/tmp/xlsxenv/bin/python tools/build_corpus.py dataset/Grupo_Demo_Dataset_Financeiro.xlsx
```

Edite `tools/build_corpus.py`, nunca os `.md` — eles são sobrescritos.

O conteúdo da planilha é integralmente fictício. O Grupo Meridiano Varejo não
existe, e os números — inclusive os que a aba `Leia-me` descreve como derivados
de notas fiscais — são sintéticos, gerados para demonstração.

## Duas decisões de desenho

**O corpus é narrativo e pré-agregado.** A recuperação entrega seis chunks ao
expert, e qwen2.5:3b cita o que recebe muito melhor do que calcula sobre séries
mensais cruas. Todo número que uma pergunta possa pedir está afirmado no texto,
já agregado.

**As keywords do expert e da ontologia estão em português.** O embedder padrão é
`hash:signed-hashing:384`, determinístico e essencialmente lexical: não há ponte
semântica entre idiomas. Um expert declarado em inglês sobre um corpus em
português cai em fallback e parece falha de modelo quando é falha de declaração.

## Estado conhecido

`alm eval run --pack packs/meridiano-financeiro`, 20 casos:

| métrica | antes do reequilíbrio | atual |
|---|---|---|
| domain accuracy | 0,70 | **0,80** |
| fallback rate | 0,15 | 0,15 |
| consistency | 0,22 | 0,28 |
| latency p95 (ms) | 18.931 | 32.940 |
| auditability | 1,00 | 1,00 |

O fallback de 15% contrasta com os 39% do `demo-enterprise` e é o efeito da
declaração em português. A acurácia ainda abaixo dele é o outro lado da mesma
moeda: o expert carrega 85% da carga em vez de escalar, e os limites de um
modelo de 3B aparecem.

O reequilíbrio corrigiu os dois casos que retornavam a **unidade errada** —
maior margem bruta e prazo de recebimento mais alongado. Eram os mais graves,
porque saíam com confiança 0,74-0,80.

A latência quase dobrou no mesmo movimento. Afirmar cada número explicitamente
engorda os documentos, e os mesmos seis chunks recuperados passam a carregar
mais texto. É o custo direto da estratégia; vale medir antes de estendê-la.

Das quatro falhas restantes, duas são o modelo encerrando a geração antes dos
números — as respostas estão corretas e truncadas, e o grader por keywords as
pune como erro total. Uma é recusa pura, com o dado recuperado em primeiro
lugar. A última é o vazamento entre packs descrito abaixo.

Quatro ressalvas honestas sobre esses números:

1. O corpus e as perguntas do eval saíram da mesma planilha, escritos na mesma
   sessão. Isso mede recuperação e citação, não raciocínio financeiro.
2. Os experts não estão calibrados — a confiança reportada não é confiável.
   Respostas erradas saíram com confiança 0,74-0,80.
3. As políticas IBAC em `policies/ibac.yaml` **não surtem efeito**. O ALM carrega
   as políticas do pack na engine em memória durante a instalação, mas não as
   persiste como nós de grafo, e `load_policies_from_graph()` não encontra nada
   num processo novo. Vale igualmente para o `demo-enterprise`.
4. O caminho de fallback **vaza entre packs**. Quando o roteador escala, a
   recuperação do orquestrador não fica restrita ao domínio da pergunta: um caso
   sobre alíquota de IR foi respondido em inglês, com o corpus de risco de
   fornecedor do `demo-enterprise`. Instalar os dois packs no mesmo tenant
   expõe isso.

Reinstalar o pack depois de editar o corpus **duplica** os documentos alterados:
eles ganham `document_id` novo e as versões antigas ficam órfãas no índice, ainda
ativas na recuperação. Não há comando para removê-las — confira `alm cmrag stats`
depois de reinstalar e limpe à mão se a contagem crescer.

## Uma inconsistência na fonte

A aba `Leia-me` afirma que Utilidades Domésticas é "a maior unidade em volume".
Por receita não é: Vestuário faz R$ 1.439.064 em 2025 contra R$ 912.346. O corpus
gerado não repete essa afirmação. A conclusão do gabarito continua válida por
outro caminho: Utilidades Domésticas é a única unidade cujo lucro bruto encolhe
em valor absoluto no período.
