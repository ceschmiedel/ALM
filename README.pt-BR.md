<h1 align="center">ALM · Agent Language Model</h1>

<p align="center">
  <strong>Uma federação governada de modelos de domínio, orquestrada pelo Context Graph.</strong><br>
  Dê a cada Expert Agent o seu próprio modelo em vez de compartilhar um LLM monolítico — e mova a
  inteligência do modelo para a orquestração.
</p>

<p align="center">
  <a href="README.md">Read in English</a>
</p>

---

## 🧭 Por quê

A arquitetura de agentes hoje dominante coloca um único LLM de fronteira no centro: todos os agentes
chamam o mesmo modelo, variando apenas o prompt. Funciona, e encosta em quatro limites estruturais
no cliente enterprise em vertical regulada — custo agregado de token, latência e dependência
externa, soberania de dados, e comportamento genérico que nunca internaliza a ontologia do cliente.

**A tese do ALM:** num domínio, uma federação de especialistas bem roteada supera o generalista —
mas essa superioridade **não é automática**. Ela depende quase inteiramente da qualidade das camadas
de contexto e roteamento, não dos modelos individuais. Por isso este projeto investe onde está a
alavanca:

| Os modelos | A orquestração |
|---|---|
| Substituíveis | Onde o valor se acumula |
| Um adaptador LoRA, um SLM, uma API | O Context Graph que os roteia, a memória que os conecta, a governança que os torna auditáveis |

O ALM não nega o LLM. Ele o reposiciona: de executor onipresente para orquestrador eventual e
professor dos modelos de domínio. Onde soberania, latência e volume pesam, o especialista local
assume; onde o raciocínio aberto é essencial, o LLM continua no circuito.

**O fallback para o LLM é projeto, não fracasso.** Ele garante que a federação nunca fique pior que
o baseline monolítico: no pior caso, ela devolve a tarefa ao LLM. No caso comum, ela resolve local,
barato e dentro do perímetro do cliente.

---

## 🏗️ Arquitetura

Cinco camadas funcionais e duas transversais. Cada uma mapeia para um pacote Python.

```
                    ┌──────────────────────────────────────────┐
   intenção ──────► │ L1  Interface e ingestão      alm.protocol
                    │     Liquid Interface Protocol (LIP)      │
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ L2  Roteador cognitivo         alm.router │
                    │     classifica · decompõe · casa · confia │
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ L4  Orquestração de dependências  alm.orchestration
                    │     plano ──► DAG · paralelo quando independente
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ L3  Federação de experts        alm.experts
                    │     modelo próprio (ALM) + recuperação própria (CMRAG)
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ L5  Arbitragem e síntese     alm.arbitration
                    │     verificador · confiança · autoridade · juiz
                    └───────────────────┬──────────────────────┘
                                        ▼
                              resposta + trilha completa

   CX  Contexto e memória  alm.graph + alm.cmrag  ── alimenta roteamento, execução, síntese
   GV  Governança (IBAC)   alm.governance         ── nenhum acesso a contexto ocorre fora dela
```

### O que é realmente novo aqui

Mixture-of-Experts não é novidade; a diferença está em **onde vive o roteamento**. Dentro de um LLM
com MoE o roteador é interno, implícito, aprendido e não auditável. No ALM o roteamento é **externo,
explícito e governável**: o Context Graph decide qual expert é ativado, em que ordem e como os
resultados são combinados — e registra por quê.

Isso torna explícito o que o monolito faz implicitamente. É ao mesmo tempo o maior custo de
engenharia da federação e a sua maior vantagem: um monolito não consegue mostrar como decidiu; uma
federação, sim. Em setores onde auditabilidade é obrigatória, isso deixa de ser diferencial e vira
pré-requisito.

---

## 🚀 Início rápido

### Instalação

Requer **Python ≥ 3.11** e **pip ≥ 21.3**. Nenhum dos dois pisos é firula, e as ferramentas padrão
do Mac não atendem a nenhum deles:

- **Python 3.11** — o código usa `StrEnum`, que não existe antes do 3.11. O macOS traz o Python 3.9
  e o `python3 -m venv` o pega silenciosamente, resultando em
  `requires a different Python: 3.9.6 not in '>=3.11'`.
- **pip 21.3** — este projeto tem apenas `pyproject.toml`, e instalação editável de um projeto sem
  `setup.py` depende da PEP 660. Um pip mais antigo falha com
  `File "setup.py" or "setup.cfg" not found`.

Então nomeie a versão do interpretador explicitamente em vez de confiar no `python3`:

```bash
git clone https://github.com/ceschmiedel/ALM.git
cd ALM

python3.12 -m venv venv           # macOS: instale antes com brew install python@3.12
source venv/bin/activate          # Windows: py -3.12 -m venv venv && venv\Scripts\activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Confira com `python --version` depois de ativar. Um `pip install .` simples funciona também em pip
antigo, se você não precisa de instalação editável.

<details>
<summary><strong>Problemas na instalação</strong> — os três erros que um macOS de fábrica produz</summary>

<br>

**`Package 'alm-orchestrator' requires a different Python: 3.9.6 not in '>=3.11'`**

O venv foi criado com o `python3` da Apple (3.9). Apague e nomeie a versão explicitamente:

```bash
rm -rf venv && python3.12 -m venv venv && source venv/bin/activate
```

**`File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode`**

O pip é anterior à 21.3 e não conhece a PEP 660. Atualize dentro do venv já ativado:

```bash
python -m pip install --upgrade pip
```

**`Command '[...venv/bin/python3', '-Im', 'ensurepip', ...]' returned non-zero exit status 1`**

O `python -m venv` apontou para um diretório que já continha um venv de outro interpretador; ele
tenta atualizar no lugar e falha. Remova em vez de reaproveitar:

```bash
rm -rf venv && python3.12 -m venv venv
```

Se ainda falhar num diretório limpo, instale o pip à parte:

```bash
python3.12 -m venv --without-pip venv && source venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python
```

</details>

### Rode a demo — sem API key, sem GPU

```bash
alm init
alm pack install packs/demo-enterprise
alm ask "Qual é nossa exposição financeira máxima sob o teto de responsabilidade da cláusula 7.2, e qual risco residual permanece?" --trace
```

Você recebe a resposta, o plano que a produziu, qual expert contribuiu com o quê, como o conflito
entre os experts jurídico e financeiro foi arbitrado, e o custo e a latência de cada chamada.

> ⚠️ O pack demo usa o backend `heuristic` — um respondedor **extrativo determinístico**, não um
> modelo de linguagem. Ele existe para que roteamento, recuperação, DAG, arbitragem, governança e
> avaliação sejam todos exercitados de verdade sem nenhuma dependência externa. Aponte para
> `ollama` ou `vllm` para rodar com modelos reais.

### Conecte modelos reais

```bash
alm install                   # assistente interativo, escreve o .env
alm model add router      --tier micro_slm    --backend ollama --model qwen2.5:0.5b
alm model add finance-slm --tier slm          --backend ollama --model qwen2.5:3b
alm model add orquestrador --tier orchestrator --backend openai --model gpt-4o
```

Serving multi-adaptador — muitos experts, uma base carregada, uma GPU:

```bash
alm model add legal-expert --tier slm --backend vllm \
  --base-model meta-llama/Llama-3.2-3B-Instruct --adapter legal-lora-v3
alm model list        # mostra bases compartilhadas, adaptadores e cargas evitadas
```

**Prefere uma interface visual?** `alm serve` sobe a API em `http://localhost:8800` e
monta um painel web em **`/ui`** — sem etapa de build, sem fontes externas, roda
offline como o resto do stack. Nele dá para ver todo modelo que seu daemon Ollama já
baixou (e o que está carregado em memória agora), atribuir um a uma camada com dois
cliques, e rodar/comparar avaliações de performance da federação contra a baseline
monolítica com o histórico completo por caso.

---

## 🎯 Escolhendo o modelo de cada agente

Não comece treinando modelos. A pergunta certa nunca é "qual modelo treinar", e sim **"este agente
precisa mesmo de um modelo próprio, ou basta recuperação de contexto sobre um modelo
compartilhado?"**

| Tier | Porte | Usar para |
|---|---|---|
| `micro_slm` | < 1B | Classificação, extração, roteamento, normalização. CPU basta. |
| `slm` | 1–4B | O especialista de domínio padrão. Uma GPU. |
| `small` | 4–8B | Raciocínio mais longo, ainda viável on-premise. |
| `orchestrator` | fronteira | Planejamento, casos ambíguos, arbitragem difícil, geração de dados de treino. Com parcimônia. |

E suba de degrau só quando o anterior não resolver — medido, não intuído:

| Técnica | Esforço | Resolve |
|---|---|---|
| RAG sobre modelo compartilhado | Baixo | Conhecimento factual que muda com frequência. **Primeira escolha sempre.** |
| Adaptador LoRA | Médio | Formato, tom, vocabulário, padrões de raciocínio. Barato de treinar **e de servir**. |
| Fine-tuning completo (SFT) | Médio-alto | Especialização profunda quando o adaptador não basta. |
| Continued pretraining | Alto | Domínios cuja linguagem o modelo base desconhece. |

```bash
alm expert promote-check finance-expert --sovereignty --calls 120000
```

avalia os quatro gatilhos de negócio — soberania, volume, latência, acurácia — contra dados reais.
**A maioria dos agentes para no LoRA.**

---

## 📊 Avaliação

> A tese do ALM é falsificável e é tratada assim. Sem harness de avaliação, o ALM é fé; com ele, é
> engenharia.

O harness roda a mesma bateria em duas configurações — **(a)** o LLM monolítico com bom prompt e
RAG, e **(b)** a federação — e compara no **conjunto** das dimensões, não em uma métrica só:

```bash
alm eval run --pack packs/demo-enterprise --compare
```

| Métrica | O que revela |
|---|---|
| Acurácia de domínio | Qualidade na tarefa real. A métrica soberana. |
| Custo por consulta | Custo total de infraestrutura, não só preço de token. |
| Latência p95 | É a cauda que o cliente sente, não a média. |
| Taxa de fallback | Cobertura dos especialistas. Reportada, não pontuada. |
| Consistência | Variação de comportamento — risco específico da federação. |
| Auditabilidade | Completude da trilha. Onde a federação vence por construção. |

**No pack demo, a federação perde em acurácia — e isso está correto.** Ambos os braços rodam no
backend extrativo, o que remove exatamente aquilo que a federação compra (raciocínio de domínio por
um modelo especializado) mantendo todo o seu overhead. O que sobressai é estrutural:
**auditabilidade 1,00 contra 0,00**.

Se a federação não vencer no conjunto, a decisão honesta é manter aquele agente como RAG sobre o
modelo compartilhado. **O ALM se aplica onde vence, não em todo lugar por princípio** — e o
`alm eval run` diz isso explicitamente em vez de deixar você ler o número que preferir.

---

## 🔐 Governança e soberania

Onde o ALM realmente brilha não é no custo, é na soberania. Com os experts e seus modelos rodando
on-premise ou na VPC do cliente, os dados sensíveis não saem do ambiente — o que transforma
exigências de NIS2, EU AI Act e LGPD de obstáculo em recurso de arquitetura.

O IBAC governa cada **agente**, não só cada usuário: um expert de um domínio não enxerga dados de
outro sem autorização explícita, e todo acesso é registrado.

```bash
alm audit tail --session ses_a1b2c3
alm audit export > audit.jsonl
```

---

## 🔄 Destilação

O gargalo real do ALM não é treinar, é ter dados de domínio de qualidade.

```bash
alm distill seed     --domain legal    # 1 · semear com o corpus real
alm distill generate --domain legal    # 2 · gerar com o professor
alm distill filter   --domain legal    # 3 · filtrar e validar
alm distill export   --domain legal --out .alm/datasets/legal.jsonl
alm distill feedback --domain legal    # 5 · falhas de produção de volta ao ciclo
```

Dados ruins ensinam erros de forma persistente, então a etapa de filtro não é opcional, e o conjunto
de avaliação nunca é visto no treino. A etapa 5 é onde a federação **melhora com o uso**: cada caso
em que um expert falhou ou escalou é capturado e realimentado.

---

## ⚠️ Limites conhecidos

Uma arquitetura honesta declara onde pode falhar.

| Risco | Mitigação neste repositório |
|---|---|
| **Carga de MLOps** | Serving multi-adaptador (menos artefatos), versionamento com promoção travada por avaliação e detecção de drift desde o primeiro agente |
| **Fragmentação de conhecimento** | Memória e roteamento centralizados no Context Graph: os modelos executam, o grafo integra |
| **Inconsistência entre agentes** | Base compartilhada, guidelines comuns e a camada L5 uniformizando a saída |
| **Modelo pequeno derrapando** | Fallback projetado para o orquestrador: a federação nunca fica pior que o baseline |
| **Curva de preço do LLM caindo** | A proposta de valor está ancorada em soberania, latência, auditabilidade e defensibilidade — que não caem com o preço do token |

O ALM não é a arquitetura padrão para tudo. É uma camada premium para verticais reguladas e de alto
volume, adotada de forma híbrida e incremental.

---

## 📚 Documentação

| Documento | Conteúdo |
|---|---|
| [docs/architecture.md](docs/architecture.md) | A arquitetura em camadas em profundidade |
| [docs/domain-packs.md](docs/domain-packs.md) | Schema dos packs e guia de autoria |
| [docs/routing.md](docs/routing.md) | Decomposição, casamento, confiança, fallback |
| [docs/arbitration.md](docs/arbitration.md) | Estratégias de resolução de conflito e síntese |
| [docs/serving.md](docs/serving.md) | Tiers, backends e serving multi-adaptador LoRA |
| [docs/evaluation.md](docs/evaluation.md) | O experimento de controle e suas métricas |
| [docs/distillation.md](docs/distillation.md) | Pipeline professor→aluno e o ciclo de realimentação |
| [docs/governance.md](docs/governance.md) | IBAC, soberania e a trilha de auditoria |
| [docs/deployment.md](docs/deployment.md) | Docker, PostgreSQL, vLLM, on-premise |
| [docs/adoption.md](docs/adoption.md) | O caminho de adoção em três ondas |

---

## 📖 Fundamentação

- **SLMs por agente são viáveis e preferíveis** — Belcak et al. (2025), [arXiv:2506.02153](https://arxiv.org/abs/2506.02153)
- **A inteligência migra para o sistema, não o modelo** — Zaharia et al. (2024), BAIR; Jain et al. (2024), [arXiv:2412.01868](https://arxiv.org/abs/2412.01868)
- **A soma de especialistas pode superar o generalista** — Jacobs et al. (1991); Shazeer et al. (2017), [arXiv:1701.06538](https://arxiv.org/abs/1701.06538); Jiang et al. (2023), *LLM-Blender*
- **Roteamento com fallback reduz custo** — Chen, Zaharia & Zou (2023), *FrugalGPT*, [arXiv:2305.05176](https://arxiv.org/abs/2305.05176)
- **Como construir os modelos de domínio** — Lewis et al. (2020), *RAG*; Hu et al. (2021), *LoRA*; Hinton et al. (2015), *Distillation*; Gururangan et al. (2020)
- **A economia do serving** — Sheng et al. (2023), *S-LoRA*, [arXiv:2311.03285](https://arxiv.org/abs/2311.03285)

---

## 📄 Licença

MIT — veja [LICENSE](LICENSE).

<p align="center">Feito com ❤️ por <a href="https://github.com/ceschmiedel">Carlos Schmiedel</a></p>
