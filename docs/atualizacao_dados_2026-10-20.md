# Atualização dos dados (20/10/2026): o que mudou, o que atualizar e o que não muda

Referência: artigo "Beyond Logistic Regression" (versão enviada em 20/10/2026) comparado com o
notebook `03_artigo_figuras_tabelas.ipynb` após rodar `concatenate_results.py` e executar o
notebook inteiro. Tudo com `kamiran_rw`, `train_window = none`, `forecast_window = 5`.

## 1. O que mudou nos dados

- `src/results/all/`: 356.262 linhas (antes 317.640).
- **330 de 330** configurações do ABCInML presentes (3 modelos × SMA com janelas 3, 5 e 10 + EMA +
  oracle × monthly/weekly × 11 α). O **oracle do Random Forest** estava incompleto (monthly, 5 α) e
  agora tem os 11 α em monthly e weekly.
- Cada modelo tem agora o mesmo número de linhas por baseline (ABCInML 23.925; Vanilla, Static e
  Dynamic 1.299), então a comparação entre modelos é balanceada.

## 2. O que precisa ser atualizado no artigo

### 2.1 Tabelas e números (não batem mais com o notebook)

| Onde | No artigo | No notebook (substituir) |
|---|---|---|
| Tab. 3 (taxas absolutas, LR) | vanilla 20,34 / 48,27 / 2,58 / 20,56; static 29,48 / 25,88 / 4,32 / 7,66; dynamic 37,65 / 27,39 / 3,75 / 4,30; abcinml 39,14 / 25,95 / 3,94 / 4,03 | vanilla **27,30 / 45,82 / 3,71 / 20,02**; static **33,92 / 28,48 / 5,57 / 9,03**; dynamic **37,78 / 27,51 / 3,77 / 4,32**; abcinml **39,47 / 26,18 / 4,00 / 4,06** (`tab03b_…`) |
| Tab. 4 (α, LR) | α = 0: 86,49 / 77,80 / 3,22 / 16,39 / 0,88 ... α = 1: DP 4,75, TPR 10,17 | α = 0: **86,48 / 77,79 / 3,14 / 16,49 / 0,86** ... α = 1: **DP 4,68, TPR 10,30** (todas as linhas mudam um pouco; `tab04_alpha_lr.tex`) |
| Tab. 5 (temporais LR) — colunas M | MB 33,73 / 10,10 / 8,25 / 9,13; TS 1,29 / 1,14 / 0,99 / 1,17; MBD 7,66 / 6,19 / 4,78 / 5,00 | **MB 24,54 / 7,43 / 8,25 / 7,98; TS 1,25 / 1,04 / 0,96 / 1,01; MBD 6,24 / 5,80 / 4,78 / 4,66** (as colunas W já batem) |
| Tab. 5, legenda | "monthly (forecast window = 3)" | **5** (e a tabela passa a ser idêntica ao bloco LR da Tab. 7: remover uma das duas) |
| Seção 5.2.2 (α ótimo do FPR) | "0,9 para RF **e XGBoost**, 0,5 para LR" | LR **0,5**, XGBoost **0,7**, RF **0,9** |
| Resumo e Introdução | "nearly four million" / "4 million" reports | ≈ **4,44 milhões** (soma dos `n` da Fig. 2) |
| Resumo | TPR piora "by about 3 percentage points" | **+2,88 a +3,84 p.p.** nos três modelos |
| Seção 5.2.1 | "across the four models"; "extended to linear models" | **three** models; **non-linear** models |

### 2.2 Texto que passou a ser possível (ou ficou incorreto) com o RF completo

- **Tab. 9 e Seção 5.3:** o oracle do RF existe agora. Valores que entrariam na Tab. 9 (MAE igual aos
  demais): RF SMA 2,23% / 12,76% / 0,62%; EMA 2,23% / 12,83% / 0,62%; Oracle 2,26% / 12,98% /
  0,64% (DP / TPR / FPR). O texto da 5.3 diz "on all the models", mas a tabela só tem LR e XGBoost.
  Para incluir o RF, basta acrescentar `'random_forest'` a `MODELS_10C` no notebook.
- **Objetivo (5) e Final Remarks:** hoje dizem "Logistic Regression and XGBoost".
- **Tab. 2 ("full factorial"):** agora é verdadeira para os 3 modelos.

### 2.3 Afirmações que merecem ajuste

- **Final Remarks:** "across all metrics, the best-performing baselines remain ABCInML and Dynamic".
  Na Tab. 7, o Static é o melhor em 3 das 18 combinações (MB monthly de LR e RF, TS monthly do RF).
- **5.2.2:** "RF é o melhor em acurácia e DP": em DP sim; em acurácia a vantagem é 86,86% contra
  86,83% do XGBoost (0,03 p.p.), e o XGBoost é melhor em AUC.
- **5.3:** "main relative gain of the EMA ... acute pandemic 1": o ganho relativo no monthly é maior
  em transition (−12,0%) e endemic (−12,1%) do que em acute 1 (−11,7%); só o absoluto é maior.
- **Tab. 4:** o notebook gera só o monthly; a legenda do artigo não diz isso.

## 3. O que não muda

- **Tab. 6** (baselines por modelo) e **Tab. 7** (temporais por modelo): todos os valores batem com o
  notebook, para os 3 modelos; o RF dos 4 baselines já estava completo.
- Todos os números do Resumo e dos Final Remarks tirados da Tab. 6: 81,8% / 81,6% / 74,1% de redução do
  DP e −0,75 / −0,81 / −0,39 p.p. vs. Dynamic ("0,4 a 0,8").
- **Fig. 5** (frontier), Fig. 3 e Fig. 4 (já em `forecast_window = 5`).
- **Tab. 8** (MAE) e **Fig. 6** (forecast vs. real): vêm das pastas `forecast_quality`, não dos resultados
  concatenados; **não foram afetadas** por esta rodada. Continuam valendo as pendências P01 e P02 de
  `docs/pendencias_validacao.md` (lotes semanais diferentes entre as pastas de 14/09 e 12/10; α_ema
  0,3 vs. 0,5).
- **Tab. 9** com LR e XGBoost: valores idênticos aos do artigo.

## 4. Notebook enxuto (feito em 21/10/2026)

O notebook passou a gerar apenas os 13 itens do artigo (6 figuras e 7 tabelas; ver o README). Foram
removidas as células das figuras e tabelas que o artigo não usa (Fig. 2 original, 4c, 5/5b, 6/6b, 7/7b,
8/8b, 9 original, 10/10b/10c/10d; Tabs. 3, 4b, 5, 6/6b, 10c, 10d, 10e). Os **arquivos** dessas figuras e
tabelas continuam em `src/notebooks/figures/artigo/` e não foram apagados; basta excluí-los, se quiser.
