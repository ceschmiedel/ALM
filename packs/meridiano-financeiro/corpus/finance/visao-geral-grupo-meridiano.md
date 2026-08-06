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
