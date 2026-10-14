# Pendências de validação do artigo (primeira versão para o orientador)

Registro das inconsistências e decisões em aberto levantadas durante a
revisão do artigo "Beyond Logistic Regression". **Nenhuma delas bloqueia a
primeira versão**: a decisão do autor (14/10/2026) é escrever a Introdução
e os Final Remarks como se não fossem problema e validá-las nos dias
seguintes. Este arquivo é a fonte única; o plano de escrita
(`docs/introducao_final_remarks_plan.md`) aponta para cá.

Convenção: **[N]** = afeta números citados no texto; **[T]** = só texto ou
figura; **[D]** = decisão do autor; **[E]** = exige novo experimento.

## 1. Estado dos números usados na Introdução e nos Final Remarks

| Número | Fonte | Estado |
|---|---|---|
| DP/TPR/FPR/Acc/AUC por baseline e modelo, redução do DP | Tab. 6 (`tab06c`) | Estável (fw 5, regenerável pelo notebook) |
| Efeito do α por modelo, α ótimo do FPR | `tab04b` | Estável (ver P05 para o RF) |
| Métricas temporais, contagem 9/18 | Tab. 7 (`tab05c`) | Estável |
| Oracle vs. SMA agregado (Fig. 7) | `fig10c` | Provisório (P06) |
| Oracle − SMA por período | Tab. 10d | Provisório (P06) |
| MAE do EMA vs. SMA | Tab. 8 | **Provisório; ver P01 e P02** |

## 2. Pendências

### P01 — Tab. 8 (MAE) não parece estar sobre os mesmos lotes das rodadas **[N][E]**
- **Evidência:** o SMA com janela 5 não depende de `ema_alpha`, mas o MAE
  semanal difere entre pastas: `weekly_forecast_window_5_ema_alpha_0.3`
  = 0,0196 (374 lotes, gerada em 14/09/2026) e
  `weekly_forecast_window_5_ema_alpha_0.5` = 0,0362 (357 lotes, gerada em
  12/10/2026). As rodadas de fairness têm 357 lotes semanais (fw 5) e 83
  mensais; as pastas de 14/09 têm 374, 369 e 376 (semanais) para fw 5, 10 e 3.
- **Impacto:** o MAE da Tab. 8 (e a Fig. 6) pode não corresponder às
  previsões efetivamente usadas pelas rodadas de fairness; "EMA reduz o MAE
  em ≈ 10%" pode mudar (com a pasta de 12/10: monthly −15%, weekly −61%).
- **Como validar:** regenerar as pastas com `src/tools/analyze_forecast_quality.py`
  (README, seção de ferramentas; rodar para `YEAR-MONTH` e `YEAR-WEEK`) com o
  mesmo `--forecast_window` e `--ema_alpha` das rodadas e comparar o número
  de lotes com `assessment_all_baselines.csv`. Não apagar as pastas antigas
  (CLAUDE.md: `src/results/` é versionado).
- **Texto afetado:** Tab. 8, parágrafo "In Table 8…", C4 do plano.

### P02 — α_ema = 0,3 no MAE vs. 0,5 nas rodadas de fairness **[N][D]**
- Tab. 8 e Fig. 6 usam `ema_alpha 0,3` (janelas 3, 5 e 10 agrupadas);
  Fig. 7 usa `ema_alpha 0,5` (a mesma das rodadas). Escolher um valor e
  usar em todos, ou declarar os dois.
- **Texto afetado:** 4.4, legendas da Tab. 8, Fig. 6 e Fig. 7; seção 3
  (Materials & Methods) deve definir α_ema.

### P03 — "main relative gain" do EMA no *acute pandemic 1* **[T]**
- Monthly: ganho relativo do EMA sobre o SMA = −10,5% (pre-pandemic),
  −11,7% (acute 1), −3,4% (acute 2), −12,0% (transition), −12,1%
  (endemic). Só o ganho **absoluto** é maior em acute 1 (0,0059).
- **Ação:** trocar "relative" por "absolute" na 4.4 (ou reformular).

### P04 — 4.3.1 × 4.3.2 (Random Forest) **[T]**
- A 4.3.2 diz que o RF é "o melhor em acurácia e DP" e que "melhora sua
  capacidade de correção"; a 4.3.1 diz, corretamente, que o ganho relativo
  é o menor.
- **Fatos:** RF tem o menor DP absoluto (2,36%) e o **menor** ganho relativo
  em DP (74,1% vs. 81,8% e 81,6%) e em Max. Bias; não em Max. Bias Diff.
  (maior ganho no monthly, −63,8%). Acurácia: RF 86,86% vs. XGBoost 86,83%
  (0,03 p.p.); AUC: XGBoost 83,17% é melhor que RF 82,77%.
- **Ação:** reescrever a 4.3.2 (nível absoluto vs. ganho relativo).

### P05 — α ótimo do FPR para o RF: 0,7 (texto) vs. 0,9 (dados) **[N]**
- Texto da 4.3.2: "0,7 para RF e XGBoost, 0,5 para LR". Nos dados (fw 5,
  monthly + weekly): LR 0,5; XGBoost 0,7; **RF 0,9** (FPR 0,50% em 0,9 vs.
  0,52% em 0,7). A diferença é pequena (0,02 p.p.).
- **Como validar:** `tab04b_alpha_models.tex` e a Fig. 5 (`fig05c_*`).

### P06 — Recortes de α diferentes entre Fig. 7 e Tab. 10d **[N][D]**
- Fig. 7: todos os α (média dos 5 períodos). Tab. 10d: α ≤ 0,3. Os sinais
  por período valem só para α ≤ 0,3; a média sobre α dilui o efeito.
- **Ação:** escolher um recorte ou rotular os dois.

### P07 — Tab. 1 ("full factorial") vs. oracle do RF incompleto **[T][E]**
- ABCInML: 313 configurações rodadas de 330 previstas. O oracle do RF só
  tem monthly e 5 dos 11 α (não existe weekly). A 4.4 usa só LR e XGBoost
  (coerente com o objetivo 5), mas a Tab. 1 sugere grade completa.
- **Ação:** reformular a nota da Tab. 1 ou completar a grade do RF.

### P08 — Objetivo (3) e pergunta de pesquisa **[T][D]**
- Objetivo (3) promete "better accuracy": a correção não muda a acurácia
  (|Δ| ≤ 0,17 p.p. vs. Dynamic; o que melhora é o retreino).
- A pergunta cita "more advanced forecasting" e "robust bias mitigation
  approaches": só SMA/EMA/oracle e um mecanismo (reweighing) foram avaliados.
- **Ação:** ajustar a redação ou declarar o escopo em 5.1.

### P09 — Tabs. 3, 4 e 5 do PDF versus as regeneradas em fw 5 **[N]**
- O notebook agora usa `FORECAST_WINDOW = 5` em todo o artigo. O PDF ainda
  tem a Tab. 5 com "monthly (forecast window = 3)" (ABCInML LR monthly
  Max. Bias 9,13%) e a Tab. 7 com fw 5 (7,98%) para o mesmo modelo: o
  próprio artigo se contradiz. A Tab. 4 do PDF (α = 0, DP 3,22%) difere
  da regenerada (3,14%).
- **Ação:** substituir as Tabs. 3, 4 e 5 e conferir as Figs. 3, 4 e 5 com
  as saídas atuais de `src/notebooks/figures/artigo/`.

### P10 — Tradução e acabamento **[T]**
- Parágrafos em português na 4.3.1 e 4.3.3; "modelos lineares" (deveria
  ser "não lineares"); "nos 4 modelos" (são 3); "TRP" em vez de TPR;
  referência `??` na 4.4 (a figura do oracle, hoje a Fig. 10d); a Fig. 8
  aparece depois das referências; "Escrever aqui texto das métricas
  temporais" (1.1.1); "Escrever Abstract…".

### P11 — Configuração do experimento ausente em Materials & Methods **[T]**
- A Seção 3 não diz quais classificadores, hiperparâmetros, sementes nem
  protocolo (treino/teste por lote, warm-up). Hiperparâmetros reais
  (`src/utils/model_utils.py`; sklearn 1.8.0, xgboost 3.2.0): LR
  `lbfgs`, `C=1`, `max_iter=1000`; XGBoost padrão com `tree_method='hist'`,
  GPU, sem `random_state`; RF `n_estimators=100`, `max_depth=12`,
  `random_state=42`. Sem busca de hiperparâmetros; uma execução por
  configuração (sem repetição por semente, sem intervalos).
- **Ação:** subseção "Models and training" na Seção 3 e limitação em 5.1.

### P12 — Estatística **[T]**
- Os p-valores (Wilcoxon pareado, Holm) usam lotes adjacentes, que são
  autocorrelacionados: servem como consistência, não como inferência. Só
  4 de 60 células da Tab. 10d passam em Holm.

### P13 — Tab. 3 (taxas absolutas) existe só para o LR **[E?]**
- Para XGBoost e RF não foi gerada. Necessária só se o texto quiser
  sustentar a explicação "FPR cai porque o grupo sensível melhora" nos
  modelos não lineares.

### P14 — Decisões do autor ainda em aberto **[D]**
- Código aberto: sim, a decidir (link/licença).
- Limite de páginas do FAccT (define o tamanho dos dois trechos).
- Formulação final da contribuição 4 ("o limitante é o reweighting"):
  mantida pelo autor; marcar o mecanismo como interpretação.

## 3. Ordem sugerida de validação

1. P01 e P02 (regenerar `forecast_quality` e decidir α_ema): afetam a
   Tab. 8 e C4.
2. P09 (substituir Tabs. 3, 4 e 5) e P04–P05 (texto da 4.3).
3. P03, P07, P08, P10 (texto).
4. P11–P12 (Methods e Limitations), P13 e P14.
