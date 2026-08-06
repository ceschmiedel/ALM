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
fonte é uma planilha de 16 abas (`Grupo_Demo_Dataset_Financeiro.xlsx`), que não
está versionada aqui. Para regenerar:

```bash
python -m venv /tmp/xlsxenv && /tmp/xlsxenv/bin/pip install openpyxl
/tmp/xlsxenv/bin/python tools/build_corpus.py /caminho/para/Grupo_Demo_Dataset_Financeiro.xlsx
```

Edite `tools/build_corpus.py`, nunca os `.md` — eles são sobrescritos.

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

Última avaliação (`alm eval run --pack packs/meridiano-financeiro`), antes do
reequilíbrio do corpus por unidade de negócio:

| métrica | valor |
|---|---|
| domain accuracy | 0,70 |
| fallback rate | 0,15 |
| consistency | 0,22 |
| auditability | 1,00 |

O fallback de 15% contrasta com os 39% do `demo-enterprise` e é o efeito da
declaração em português. A acurácia mais baixa é o outro lado da mesma moeda: o
expert carrega 85% da carga em vez de escalar, e os limites de um modelo de 3B
aparecem.

Três ressalvas honestas sobre esses números:

1. O corpus e as perguntas do eval saíram da mesma planilha, escritos na mesma
   sessão. Isso mede recuperação e citação, não raciocínio financeiro.
2. Os experts não estão calibrados — a confiança reportada não é confiável.
   Respostas erradas saíram com confiança 0,74-0,80.
3. As políticas IBAC em `policies/ibac.yaml` **não surtem efeito**. O ALM carrega
   as políticas do pack na engine em memória durante a instalação, mas não as
   persiste como nós de grafo, e `load_policies_from_graph()` não encontra nada
   num processo novo. Vale igualmente para o `demo-enterprise`.

## Uma inconsistência na fonte

A aba `Leia-me` afirma que Utilidades Domésticas é "a maior unidade em volume".
Por receita não é: Vestuário faz R$ 1.439.064 em 2025 contra R$ 912.346. O corpus
gerado não repete essa afirmação. A conclusão do gabarito continua válida por
outro caminho: Utilidades Domésticas é a única unidade cujo lucro bruto encolhe
em valor absoluto no período.
