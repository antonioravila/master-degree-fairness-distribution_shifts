# Plano de implementação: `--train_window` + migração do schema existente

Branch de trabalho: `apr_all_regions` (todas as regiões, kamiran-only,
schema já migrado pela rodada anterior — sem `adaptative_rw`, sem coluna
`train_window`). Este documento assume os valores **`TRAIN_WINDOWS = [None, 3,
6, 12]`** (meses) — a linha da tabela de crescimento combinatório da análise
anterior com 4 valores. Se a intenção era outra linha/outros valores, ajustar
a seção 2 antes de implementar; o resto do plano não depende do número exato
de valores.

Complementa `docs/rolling_window_plan.md` (design geral, já existente) — este
arquivo foca no que aquele deixou em aberto: a divisão do cache
`vanilla+static+dynamic` (§7.1 de lá) e a migração dos 224 resultados que já
existem nesta branch sem essa dimensão.

## 1. Recap da decisão de design

- Parâmetro novo: `--train_window` (meses, mesma convenção de
  `--forecast_window`), `default=None` = comportamento atual (histórico
  completo).
- Afeta só `dynamic` e `abcinml`. `vanilla`/`static` são ignorados (treinam
  uma vez, num único batch).
- Reponderação (Kamiran/APR) não muda — já opera sobre o que for passado
  como treino.

## 2. Valores a testar

```python
TRAIN_WINDOWS = [None, 3, 6, 12]  # meses; None = histórico completo (atual)
```

`None` entra explicitamente na lista (não é "ausência de flag") porque
precisamos dele como ponto de comparação da validação (seção 8) e ele
também precisa aparecer, de forma explícita, no nome de pasta e na coluna do
CSV — ver seção 4.

## 3. Mudança estrutural obrigatória: separar `dynamic` do cache `vanilla+static`

Hoje (`src/main.py`, bloco `non_abc_to_run`), `vanilla`+`static`+`dynamic`
rodam juntos e caem num único arquivo
(`baseline_non_abc_{model}_{batch_type}_{fairness_correction}`). Confirmado
nos dados desta branch: essa pasta tem uma única `assessment_all_baselines.csv`
com as 3 baselines misturadas (83 linhas cada). Como só `dynamic` passa a
depender de `train_window`, esse cache único vira ambíguo assim que a janela
existir (§7.1 de `rolling_window_plan.md`).

**Mudança**: dividir o bloco `non_abc_to_run` em dois caminhos de
salvamento:

- `vanilla` + `static` continuam juntos, numa pasta que não leva sufixo de
  `train_window` (eles nunca dependem disso):
  `baseline_non_abc_{model}_{batch_type}_{fairness_correction}` — **mesmo
  nome de hoje**, só que agora contém apenas essas duas baselines.
- `dynamic` ganha pasta própria, com sufixo explícito de `train_window`:
  `baseline_dynamic_{model}_{batch_type}_{fairness_correction}_trainwindow_{W}`
  (`W` = `none` ou o número de meses).

Isso exige:
- Nova função `metrics_utils.baseline_dynamic_folder_name(model_name,
  batch_type, fairness_correction, train_window)`, irmã de
  `baseline_non_abc_folder_name` (que passa a cobrir só vanilla+static).
- Em `main.py`, `_NON_ABC_BASELINES` deixa de incluir `dynamic`; um novo
  bloco roda `dynamic` isoladamente (mesmo padrão de cache/skip-se-existe que
  o bloco atual já tem), passando `train_window_batches`.
- `_attach_metadata` grava `train_window` (valor em meses ou `'none'`) nos 3
  CSVs de **todas** as pastas — inclusive vanilla/static (coluna vazia/`none`
  sempre, já que não se aplica) e abcinml (que também ganha a mesma coluna,
  ver seção 4).

## 4. Nome de pasta: sufixo sempre explícito (não condicional)

O rascunho original de `rolling_window_plan.md` §7 dizia "sufixo só quando
`train_window` não é `None`". **Revisão**: seguir o mesmo padrão já adotado
para `fairness_correction` (sufixo sempre presente, mesmo no valor default) —
foi essa explicitação que evitou ambiguidade na migração anterior, e o mesmo
raciocínio vale aqui: uma pasta sem sufixo de `train_window` seria ambígua
entre "rodada antes dessa mudança existir" e "rodada com `train_window=None`
depois dela".

Convenção final:
- `dynamic`: `baseline_dynamic_{model}_{batch_type}_{fc}_trainwindow_{W}`
- `abcinml` (pastas `sma`/`ema`/`oracle`, uma por alpha): sufixo adicional no
  fim de `metrics_utils._run_folder_name`, ex.:
  `logistic_regression_alpha_0.1_sma_window_5_monthly_kamiran_rw_trainwindow_12`
- `vanilla`+`static`: sem sufixo (nunca variam com `train_window`).

`W` formatado como `none` (string) ou o inteiro de meses — mesmo tratamento
que `_fmt_alpha` já dá para números.

## 5. Migração dos resultados já existentes nesta branch (224 pastas)

Todas as 224 pastas atuais desta branch foram rodadas com histórico completo
(comportamento pré-`train_window`) — ou seja, correspondem a
`train_window=None` na nova convenção. A migração não descarta nada, só
reclassifica e, num caso, separa:

**5.1. As 4 pastas `baseline_non_abc_{model}_{batch_type}_kamiran_rw`**
(vanilla+static+dynamic combinados hoje):
1. Ler `assessment_all_baselines.csv`, `temporal_results_df.csv`,
   `absolute_results_df.csv`.
2. Separar linhas por `baseline`: `vanilla`/`static` ficam nos 3 CSVs
   originais, na mesma pasta (nome não muda).
3. Linhas de `dynamic` são extraídas para uma nova pasta
   `baseline_dynamic_{model}_{batch_type}_kamiran_rw_trainwindow_none`
   (3 CSVs próprios).
4. Adicionar coluna `train_window` = `'none'` nas linhas de `dynamic`
   extraídas; `''`/`'none'` (irrelevante, mas presente) nas de
   `vanilla`/`static` que ficam para trás.

**5.2. As 220 pastas `{model}_alpha_{a}_{technique}_window_{w}_{batch}...`**
(só `abcinml`, confirmadas 1-baseline-só):
1. Renomear a pasta acrescentando `_trainwindow_none` no fim.
2. Adicionar coluna `train_window='none'` nos 3 CSVs de cada uma.

**5.3. Script**: estender/copiar o padrão de `migrate_legacy_schema.py`
(já usado nesta branch para a migração anterior) — nova versão
`migrate_train_window_schema.py`, com:
- Regex sobre os nomes de pasta atuais (sem o novo sufixo) para localizar as
  224 pastas-alvo.
- Split por `baseline` só para as 4 pastas `baseline_non_abc_*`.
- Rename + column backfill para as outras 220.
- Idempotente (se already-migrated, isto é, se já existe `_trainwindow_`
  no nome, pula) — mesmo cuidado que o script anterior teve.
- Rodar com `--dry-run` primeiro (listar o que seria renomeado/dividido sem
  escrever nada), só aplicar de fato depois de revisão manual — mesmo
  procedimento cauteloso da migração de `fairness_correction`.

## 6. Ferramentas que dependem do nome/coluna (mesma lista do estudo anterior, agora com ação)

- **`concatenate_results.py`**: mais um grupo opcional no regex
  (`_ABC_RE`/`_BASELINE_RE`) para `_trainwindow_{W}`; `_BASELINE_RE` também
  precisa reconhecer o novo prefixo `baseline_dynamic_` (hoje só reconhece
  `baseline_non_abc_`).
- **`vanilla_shift_chart.py`**: continua apontando para
  `baseline_non_abc_folder_name` sem mudança (vanilla não se move). Se algum
  dia precisar comparar `dynamic` aqui também, vai precisar do novo
  `baseline_dynamic_folder_name` + escolher um `train_window`.
- **`alpha_analysis.py`**: hoje já não filtra por `fairness_correction`
  (lacuna pré-existente, seção 11 do estudo anterior); com `train_window`
  como mais uma coluna, precisa de filtro explícito por ambos antes de
  qualquer `groupby('alpha')` — senão mistura `none`/`3`/`6`/`12` sem avisar.
- **`run_parallel.py`**: adicionar `TRAIN_WINDOWS` como eixo **opt-in**, não
  no grid default — ex. `--train_windows none 3 6 12` como flag nova,
  default `['none']` (grid idêntico ao de hoje se ninguém passar a flag).
  Evita a explosão 4x descrita na análise anterior até você decidir rodar a
  exploração de propósito.

## 7. Ordem de execução recomendada

1. Implementar o parâmetro + slicing em `run_baseline` (sem mexer em nomes
   de pasta ainda) — validar com `--test_run` que `dynamic`/`abcinml` com
   `train_window` passado produzem menos linhas de treino por batch.
2. Implementar a separação `dynamic` vs. `vanilla+static` (seção 3) e a
   nova convenção de nomes (seção 4) — sem rodar nada novo ainda.
3. Rodar `migrate_train_window_schema.py --dry-run`, revisar a lista, então
   aplicar de fato — isso deixa as 224 pastas existentes no novo schema,
   equivalente a `train_window=none`.
4. Atualizar `concatenate_results.py`/`alpha_analysis.py` e regenerar os
   agregados para confirmar que os dados migrados continuam batendo com os
   números já publicados (checagem de regressão antes de rodar qualquer
   coisa nova).
5. Só então rodar as combinações novas (`train_window` em `[3, 6, 12]`) —
   inicialmente num subconjunto reduzido (1 modelo, 1 `batch_type`), seguindo
   a recomendação de não embutir no grid completo de cara.

## 8. Validação (reforça §9 de `rolling_window_plan.md`)

Antes de considerar qualquer `W` "oficial": comparar `dynamic`/`abcinml` com
`train_window=none` (dado migrado) contra os novos `train_window` candidatos,
mesmo `model_name`/`batch_type`/`fairness_correction` — accuracy + as 3
métricas de fairness lado a lado. Só promover um valor depois dessa
comparação, documentando a escolha.
