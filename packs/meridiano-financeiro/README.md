# meridiano-financeiro

Federação de FP&A sobre o modelo financeiro de três demonstrações do **Grupo
Meridiano Varejo Ltda**, entidade sintética com cinco unidades de negócio e 36
meses de histórico (jan/2025 a dez/2027).

Diferente do `demo-enterprise`, este pack roda com **modelos reais** — Ollama
local, sem API key, nada saindo da máquina.

```bash
declm pack install packs/meridiano-financeiro
declm ask "A receita cresceu mas o lucro caiu. Explique o que aconteceu." --trace
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

`declm eval run --pack packs/meridiano-financeiro`, 20 casos:

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
lugar. A última era o vazamento entre packs descrito abaixo (item 4), hoje corrigido.

Duas ressalvas honestas sobre esses números continuam de pé:

1. O corpus e as perguntas do eval saíram da mesma planilha, escritos na mesma
   sessão. Isso mede recuperação e citação, não raciocínio financeiro.
2. Os experts não estão calibrados — a confiança reportada não é confiável.
   Respostas erradas saíram com confiança 0,74-0,80.

Três comportamentos encontrados durante a iteração — e já corrigidos no núcleo do `alm`:

3. **Políticas IBAC não persistiam.** O DecLM carregava as políticas do pack na
   engine em memória durante a instalação, mas não as persistia como nós de
   grafo, e `load_policies_from_graph()` não encontrava nada num processo novo
   — a governança virava um no-op fora do processo único de `declm pack install`.
   `apply_pack_policies()` agora grava cada política como um nó `NodeKind.POLICY`,
   e `from_database()` volta a encontrá-las.
4. **O fallback vazava entre packs.** Quando o roteador escalava sem nenhum
   domínio identificado, a recuperação do orquestrador não ficava restrita ao
   domínio da pergunta: um caso sobre alíquota de IR foi respondido em inglês,
   com o corpus de risco de fornecedor do `demo-enterprise`. A recuperação do
   fallback agora usa os domínios candidatos da classificação quando existem e,
   na ausência total de sinal com mais de um domínio instalado, não recupera
   contexto nenhum — uma lacuna honesta em vez de uma citação errada.
5. **Reinstalar o pack depois de editar o corpus órfã documentos.** Um arquivo
   alterado ganhava `document_id` novo e a versão antiga ficava órfã no índice,
   ainda ativa na recuperação, sem comando para removê-la.
   `ChunkStore.upsert_document()` agora identifica o documento por
   `(tenant, domínio, uri)` antes de decidir por hash de conteúdo, e substitui
   os chunks antigos no lugar em vez de duplicá-los.

## Benchmark de decisões

`decisions/fpa.jsonl` traz 265 decisões tipadas em português sobre este mesmo
modelo financeiro — 115 de escolha, 106 de sim/não e 44 de escala, em dez
famílias. É gerado por `tools/build_decisions.py` a partir da planilha; edite o
script, nunca o `.jsonl`.

```bash
declm decide bench packs/meridiano-financeiro/decisions/fpa.jsonl --split test --model <modelo> --backend readout
```

Três propriedades do conjunto:

- **O gabarito é calculado, não julgado.** Cada resposta esperada sai de
  aritmética sobre a planilha. A exceção é `roteamento`, cujas 30 perguntas foram
  escritas à mão, cada uma para uma área.
- **As famílias são balanceadas.** Responder sempre o rótulo mais comum rende no
  máximo 50% em qualquer família, e 20% no conjunto.
- **O estado traz os insumos, nunca o número derivado.** Duas receitas, não a
  variação; preço e custo, não a margem.

71 itens são `dev`, para ajustar temperaturas; 194 são `test`, para reportar.

### Estado conhecido

Divisão `test`, 194 itens, uma rodada por modelo, sem nenhum treino:

| família | Qwen2.5 0,5B | Qwen2.5 1,5B | Qwen3.5 9B | Qwen2.5 72B |
|---|---|---|---|---|
| alcada_desconto | 23% | 23% | 59% | 73% |
| cenarios | 18% | 27% | 55% | 82% |
| ciclo_de_caixa | 47% | 47% | 47% | 47% |
| comparacao_anual | 65% | 54% | 92% | 92% |
| faixa_crescimento | 18% | 27% | 86% | 100% |
| meta_representante | 33% | 33% | 62% | 71% |
| roteamento | 83% | 87% | 87% | 83% |
| tendencia_trimestre | 57% | 57% | 100% | 79% |
| unidade_extrema | 21% | 47% | 95% | 100% |
| venda_permitida | 42% | 58% | 68% | 100% |
| **acurácia** | **42,3%** | **46,9%** | **76,3%** | **83,5%** |
| Brier | 0,706 | 0,756 | — | — |
| ECE | 0,163 | 0,334 | — | — |
| modo | leitura | leitura | generativo | generativo |
| onde rodou | local, CPU | local, CPU | Hugging Face | Hugging Face |
| latência p50 | 1,3 s | 3,7 s | 0,8 s | 3,1 s |

O piso — sempre o rótulo mais comum — é 20,1%.

Como ler a tabela:

- **O conjunto discrimina.** Vai de 42% a 83% conforme o porte do modelo, então
  serve de régua para um modelo treinado.
- **Modelos pequenos roteiam, mas não calculam.** Abaixo de 2B, só `roteamento`
  e `comparacao_anual` ficam acima do acaso; as famílias com regra numérica não.
- **`ciclo_de_caixa` derrota todos.** Os quatro ficam em 47%, abaixo do piso de
  53% da família. É a única que exige somar e subtrair três números com decimais
  antes de comparar — o tipo de decisão que deve ser decomposta em código.
- **Calibração só existe na leitura.** Os modelos hospedados não devolvem
  `logprobs`: sondados 13 pares de modelo Qwen e provedor pelo roteador do
  Hugging Face, nenhum devolveu. Sem distribuição, não há Brier nem ECE.

Ressalvas: a comparação mistura porte e modo — os locais são pequenos e usam
leitura, os hospedados são grandes e usam JSON —, a latência local é de CPU, e o
Qwen3.5 rodou com o raciocínio desligado (`--param reasoning_effort=none`).

## Uma inconsistência na fonte

A aba `Leia-me` afirma que Utilidades Domésticas é "a maior unidade em volume".
Por receita não é: Vestuário faz R$ 1.439.064 em 2025 contra R$ 912.346. O corpus
gerado não repete essa afirmação. A conclusão do gabarito continua válida por
outro caminho: Utilidades Domésticas é a única unidade cujo lucro bruto encolhe
em valor absoluto no período.
