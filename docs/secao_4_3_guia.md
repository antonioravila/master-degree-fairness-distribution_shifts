# Guia: onde cada análise entra na Seção 4.3 ("Expansion to Non Linear Models")

Documento de design e de redação. Lista as análises adicionais geradas em
`src/notebooks/03_artigo_figuras_tabelas.ipynb` (seção "4.3 — análises
adicionais" no fim do notebook), onde cada uma entra no texto, qual
mensagem ela sustenta e o que precisa ser corrigido no texto atual.

**Configuração de todas as saídas** (a mesma da `fig04c`/Fig. 6):
`kamiran_rw`, `train_window = none`, `forecast_window = 5`
(`FORECAST_WINDOW`), SMA; ABCInML = média sobre os 11 alphas; monthly e
weekly com pesos iguais. Declarar isso uma vez na seção 3 do artigo.

## 1. Estrutura proposta da 4.3

| Subseção | Mensagem | Material |
|---|---|---|
| 4.3.1 Visão geral por modelo | A correção reduz o DP em ~74–82% nos três modelos; o ganho relativo do RF é o menor | **Tab. 6c** (`tab06c_baselines_models.tex`) |
| 4.3.2 Efeito do alpha (já existe, ampliar) | O DP piora e o TPR melhora com alpha nos três modelos; a sensibilidade cai do LR ao RF | Fig. 5 (existente) + **Tab. 4b** (`tab04b_alpha_models.tex`) |
| 4.3.3 Antecipar vs. apenas retreinar | ABCInML < Dynamic no DP (−0,4 a −0,8 p.p.), mas > Dynamic no TPR (+2,9 a +3,8 p.p.) | **Tab. 6c** (última linha de cada bloco) |
| 4.3.4 Estabilidade temporal | ABCInML é competitivo, mas não uniformemente melhor que o Dynamic | **Tab. 5c** (`tab05c_temporal_baselines_models.tex`), substitui a Tab. 6 atual |

Ordem de leitura sugerida dentro da seção: Fig. 6 (existente) → Tab. 6c →
Tab. 4b e Fig. 5 → Tab. 5c.

## 2. Roteiro parágrafo a parágrafo

### 4.3.1 Visão geral (Tab. 6c)

- Abrir com a pergunta do objetivo (3) e dizer que a comparação é entre os
  quatro baselines para os três modelos.
- Reportar a redução do DP relativa ao vanilla: LR 81,8%, XGBoost 81,6%,
  RF 74,1% (coluna "DP red.").
- **Corrigir a leitura atual:** o RF tem o menor DP absoluto (2,36%), mas o
  vanilla dele já é muito mais justo (9,08% vs. 21,97% no LR e 17,91% no
  XGBoost), então o *ganho* relativo é o menor. Reescrever "o ABCInML
  melhora sua capacidade de correção no RF" como "o RF atinge o menor DP
  absoluto, em parte porque parte de um vanilla mais justo".
- Acurácia: Dynamic e ABCInML ficam em 86,6–87,0%, contra 82,6–86,1% de
  vanilla/static. A diferença vem de retreinar (vanilla e static são
  treinados uma vez, num batch pré-pandemia), não da correção de fairness.

### 4.3.2 Alpha por modelo (Tab. 4b e Fig. 5)

- O DP aumenta com alpha nos três modelos: α=0 → 1 muda +1,48 p.p. (LR),
  +1,20 (XGBoost) e +0,62 (RF). Em termos relativos: +45%, +41% e +29%.
- O TPR cai com alpha (−6,1; −7,3; −5,7 p.p.): o oposto do DP.
- O alpha ótimo do FPR cresce com a flexibilidade do modelo: 0,5 (LR),
  0,7 (XGBoost), 0,9 (RF).
- **Corrigir:** o texto atual diz "0,6 para LR"; a Tab. 4 do LR (fw 5,
  monthly) tem o melhor FPR em 0,5, e a Tab. 4b (monthly+weekly) também.
  Alinhar texto e tabela.

### 4.3.3 Antecipar vs. retreinar (Tab. 6c, linhas "ABCInML − Dynamic")

- DP: −0,75 (LR), −0,81 (XGBoost), −0,39 (RF) p.p.
- TPR: +2,99, +3,84, +2,88 p.p. FPR e acurácia: empate (|Δ| ≤ 0,17 p.p.).
- Explicação teórica a acrescentar: o reweighing de Kamiran & Calders
  equaliza P(S,Y), isto é, mira a paridade demográfica por construção, sem
  garantia sobre TPR/FPR. Isso explica por que a correção anticipada
  melhora o DP e degrada o TPR, mais do que apenas citar o trade-off de [14].
- O Static é o melhor em TPR nos três modelos (8,9 / 6,6 / 3,2%): a
  correção "mais lenta" preserva o TPR. Mencionar como ressalva.

### 4.3.4 Temporais (Tab. 5c)

- Substitui a Tab. 6 (que só tem ABCInML). Com os 4 baselines, a conclusão
  correta é: ABCInML e Dynamic são os melhores e muito próximos.
- Exceção a registrar: XGBoost weekly, onde o ABCInML é pior que o Dynamic
  em Max. Bias (14,40% vs. 10,98%) e Max. Bias Diff. (10,94% vs. 8,13%).
- RF weekly: ABCInML melhor em todas as três (7,13 / 0,74 / 4,86%).

## 3. Reconciliação com as tabelas atuais do artigo

- **Decisão do autor: `forecast_window = 5` em todo o artigo.** O notebook
  filtra `assessment_df`, `absolute_df` e `temporal_df` por
  `FORECAST_WINDOW = 5` na célula de setup, e `TABLE_WINDOW` passou de
  `{monthly: 3, weekly: 5}` para 5 nos dois. Efeito nas saídas existentes:
  Figs. 3, 4, 5–7 (e variantes), Tabs. 3, 4, 5 e 6 foram regeneradas com
  fw 5; as Tabs. 5 e 6 do PDF (monthly fw 3) precisam ser substituídas
  pelas novas (ex.: ABCInML LR monthly Max. Bias 9,13% → 7,98%; Tab. 4 do
  LR α=0: DP 3,22% → 3,14%).
- Antes, as Tabs. 3 e 4 e as Figs. 4–7 tiravam a média sobre os fws 3, 5 e 10
  (sem filtro), o que misturava modelos vanilla/static de referência
  diferentes (`docs/fig4_modelos_plan.md`, seção 3). Com fw 5 isso deixa de
  ocorrer. O texto do artigo deve declarar "forecast window = 5" uma vez, na
  seção 3 (Materials & Methods), em vez de repetir por tabela.
- A Tab. 1 do PDF lista SMA com janelas (3, 5, 10): continuam rodadas, mas
  a análise principal reporta só a janela 5; mencionar que 3 e 10 ficam
  como sensibilidade (não reportada).
- A Tab. 6 atual (ABCInML por modelo) fica redundante com a Tab. 5c e pode
  ser removida ou movida ao apêndice.
- A Tab. 3 (taxas absolutas) existe só para o LR; para XGBoost e RF não foi
  gerada. Se quiser sustentar a frase "FPR cai porque o grupo sensível
  melhora", gerar as duas outras versões (pendente, não incluída aqui).

## 4. Numeração sugerida no artigo

Mantendo as figuras/tabelas existentes e inserindo as novas na ordem de
leitura da seção 4.3:

| Novo | Arquivo | Posição |
|---|---|---|
| Table 6 (nova) | `tab06c_baselines_models.tex` | 4.3.1 |
| Table 7 (nova) | `tab04b_alpha_models.tex` | 4.3.2, após a Fig. 5 |
| Table 8 (nova) | `tab05c_temporal_baselines_models.tex` | 4.3.4, no lugar da Tab. 6 atual |

As tabelas posteriores (a Tab. 7 atual do MAE e a Tab. 8 da 10d) são
renumeradas automaticamente pelo LaTeX.

## 5. Ressalvas que devem aparecer no texto

1. **Configuração:** dizer em cada legenda `forecast window = 5`, ABCInML =
   média sobre alphas, monthly e weekly com pesos iguais.
2. **Alpha:** a média sobre alphas mistura valores onde a antecipação pesa
   pouco (α próximo de 1, equivalente ao Dynamic). As diferenças ABCInML −
   Dynamic da Tab. 6c são, portanto, uma média entre configurações; a
   Tab. 4b mostra o DP em cada alpha.
3. **Random Forest:** a análise de forecasting (4.4) não inclui o RF
   porque o Oracle dele não foi rodado para os 11 alphas em monthly e
   weekly; a 4.3 não é afetada.

## 6. Saídas geradas (nome → célula do notebook)

| Arquivo | Conteúdo |
|---|---|
| `tab06c_baselines_models.tex` | 4 baselines × 3 modelos; Acc, AUC, DP, TPR, FPR; redução do DP; ABCInML − Dynamic |
| `tab04b_alpha_models.tex` | alpha × (DP, TPR, FPR) × 3 modelos; linha Δ (α=1 − α=0) |
| `tab05c_temporal_baselines_models.tex` | Max. Bias, Temporal Stability, Max. Bias Diff.; 4 baselines × 3 modelos; M e W |

Todas em `src/notebooks/figures/artigo/`; reproduzíveis com
`concatenate_results.py` seguido da execução do notebook.
