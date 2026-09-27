# Plano: incluir Random Forest como terceiro `--model_name`

Documento de design — descreve **como** adicionar `random_forest` ao eixo de
modelos, antes de qualquer mudança de código. Não altera o comportamento de
`logistic_regression` nem de `xgboost`: nenhum resultado existente é invalidado,
renomeado ou migrado por esta mudança.

Ao contrário das mudanças anteriores de eixo (`--train_window`,
`--forecast_window`), que **mudaram o schema de pastas** e exigiram os scripts
`migrate_*.py`, esta não: o eixo `model_name` já existe em todos os nomes de
pasta e em todas as CSVs desde sempre, e só ganha um valor novo.

## 1. Motivação

O pipeline compara hoje um modelo linear (`logistic_regression`) e um de
boosting (`xgboost`). Random Forest cobre uma terceira família — bagging de
árvores — com um perfil de viés/variância distinto dos dois. Como toda a
análise de fairness é feita *por modelo* (`alpha_analysis.py` já gera uma
coluna por `model_name`), incluir RF amplia a comparação sem custo de
reestruturação.

## 2. Escopo

**Dentro do escopo:**
- Novo valor `random_forest` para `--model_name`, aceito por `src/main.py` e
  por `src/tools/vanilla_shift_chart.py`.
- Entrada correspondente em `model_utils.initialize_model`.
- Inclusão de RF na grade paralela (`run_parallel.py`), com pool próprio.
- Correção do roteamento de jobs em `run_parallel.py` (seção 6) — hoje ele
  descartaria jobs de RF silenciosamente.
- Decisão explícita sobre a chave de cache de hiperparâmetros (seção 5).

**Fora do escopo (não fazer nesta mudança):**
- Não tunar hiperparâmetros por busca automatizada. A configuração inicial é
  fixa e justificada na seção 4; tuning, se necessário, vira trabalho
  posterior sob a estratégia (b) da seção 5.
- Não mudar `metrics_utils.py`, nem os nomes de pasta, nem o schema das CSVs.
- Não migrar nada. Não rodar os `migrate_*.py` (apenas verificá-los em
  `--dry-run`, seção 8).
- Não mexer no `SOURCE_DIR` fixo em `IDADE_BIN` de `concatenate_results.py`
  (limitação pré-existente, independente de RF — ver seção 10).

## 3. Dependências

**Nenhuma dependência nova.** `RandomForestClassifier` vem de
`scikit-learn>=1.8.0`, já declarado em `pyproject.toml`. Não há `uv add`,
nem alteração de lockfile.

O que muda é o **perfil de recurso**, e isso importa para a seção 6:

| Modelo | Recurso dominante | Observação |
|---|---|---|
| `logistic_regression` | CPU | pool de CPU atual |
| `xgboost` | GPU (`device='cuda'`) | não disputa CPU com LR |
| `random_forest` | **CPU + RAM** | disputa diretamente com LR |

RAM é o gargalo mais provável: o scikit-learn converte `X` para um array
`float32` contíguo antes de treinar, duplicando em memória o DataFrame de
treino. Em `dynamic`/`abcinml` sem `--train_window`, esse DataFrame é o
histórico acumulado inteiro (milhões de linhas × ~60 colunas após o one-hot
de `data_utils._convert_types`). É por isso que a seção 9 recomenda começar o
sweep com `--train_window` limitado.

## 4. Interface do modelo e hiperparâmetros

### 4.1 Compatibilidade — nada a adaptar

Todo o pipeline downstream já é agnóstico ao modelo. Os três contratos que
`RandomForestClassifier` precisa cumprir, e cumpre:

| Contrato | Usado em |
|---|---|
| `fit(X, y, sample_weight=...)` | `model_utils.fit_fair_model` (os 3 caminhos) |
| `predict(X)` | `main.py:146`, `model_utils.adaptive_priority_reweighing` |
| `predict_proba(X)` | `main.py:147`, `model_utils.compute_margins` (Eq. 19) |

Consequência importante: **APR (`adaptative_rw`) funciona com RF sem nenhuma
alteração**. As margens da Eq. 19 saem de `predict_proba`, que a floresta
fornece como fração de votos das árvores. O custo, porém, é alto — ver 9.3.

### 4.2 Configuração inicial proposta

```python
'random_forest': RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    n_jobs=-1,
    random_state=42,
)
```

> **Revisado na Fase 2 (2026-09-26):** a proposta original incluía
> `min_samples_leaf=50`. Foi removido — ver §12. Com ele, o RF treinado no
> batch de referência de `vanilla`/`static` (6.765 linhas no mensal, 1.783 no
> semanal) **nunca prevê a classe positiva** (`max predict_proba` = 0,43),
> produzindo `dp = tpr_diff = fpr_diff = 0` artificiais em todos os batches.

Justificativas, uma a uma:

- **`max_depth=12`** — sem limite de profundidade, uma floresta sobre o
  histórico acumulado cresce árvores até folhas puras (130M nós medidos no
  histórico completo). A profundidade limita cada árvore a ≤ 4.096 folhas e
  torna o custo por `fit` previsível. `min_samples_leaf` não contribui para a
  memória (§12) e foi deixado no default (1).
- **`n_jobs=-1`** — paraleliza a construção das árvores. Interage com o
  paralelismo de processos do `run_parallel.py`: ver 6.2.
- **`random_state=42`** — LR e XGB não fixam semente hoje, mas RF é
  visivelmente estocástico (amostragem bootstrap + amostragem de features).
  Sem semente, dois runs idênticos produzem métricas de fairness diferentes,
  o que é inaceitável para os números da dissertação.
- **Sem `class_weight`** — deliberado. O balanceamento entre grupos é papel
  da reponderação (Kamiran & Calders via `reweigh_df`, ou APR via
  `update_apr_sample_weights`), entregue ao modelo por `sample_weight`. Usar
  `class_weight` em paralelo aplicaria duas correções sobrepostas e tornaria
  o efeito do `alpha` do ABCInML impossível de isolar.

### 4.3 Sobre o parâmetro `MAX_ITER`

`initialize_model(model_name, MAX_ITER)` recebe `MAX_ITER` posicionalmente
para qualquer modelo. `xgboost` já o ignora hoje; RF também ignora. Manter a
assinatura como está — refatorar para `**kwargs` seria churn sem ganho.

## 5. Decisão pendente: chave de cache vs. hiperparâmetros

**Este é o risco menos óbvio de toda a mudança.**

`src/main.py` usa a existência da pasta de resultados como cache:

```python
# main.py:250 (vanilla/static) e main.py:323 (dynamic)
if os.path.exists(non_abc_assessment_path):
    print(f'--- Loading cached non-ABC results from {non_abc_results_dir} ---\n')
```

O nome da pasta codifica modelo, alpha, técnica de forecast, `forecast_window`,
`batch_type`, `fairness_correction` e `train_window` — **mas não codifica
hiperparâmetros do modelo**. Isso nunca foi problema porque LR e XGB têm
hiperparâmetros fixos desde o primeiro commit.

Com RF, o cenário realista é: rodar com `max_depth=12`, olhar os resultados,
decidir testar `max_depth=None`. A segunda execução encontra a pasta, carrega
as CSVs da **primeira** configuração, imprime "Loading cached…" e termina com
sucesso. Nenhum erro, nenhum aviso, resultados errados na dissertação.

Duas estratégias — escolher **antes** de gerar qualquer pasta de RF:

**(a) Congelar a configuração.** Fixa-se a configuração da seção 4.2 e não se
mexe mais. Se mudar, apagar manualmente as pastas de `random_forest` em
`src/results/` e `src/figures/` antes de re-rodar. Consistente com LR/XGB,
zero código extra, mas depende de disciplina manual.

**(b) Variante como nome de modelo.** Registrar cada configuração como um
`model_name` distinto (`random_forest`, `random_forest_deep`, …), cada um com
sua entrada no `models_mapping`. O cache passa a discriminar as variantes de
graça, e as duas configurações aparecem lado a lado nos gráficos — o
`alpha_analysis.py` já gera uma coluna por `model_name`, então a comparação
sai sem trabalho adicional.

**Recomendação:** (b) se há intenção de tunar RF; (a) se a configuração da
4.2 vai ser fixada e seguida. Em ambos os casos, registrar a decisão neste
documento antes da Fase 1.

**Decisão registrada (2026-09-26): (b).** A configuração da §4.2 fica
congelada sob o nome `random_forest`; qualquer outra configuração entra como
um `model_name` novo (ex.: `random_forest_deep` para `max_depth=None`). A regra
está documentada no README (seção *Outputs*) e num comentário no
`models_mapping`. Nenhum código extra é necessário para isso agora.

Duas regras do `CLAUDE.md` pesam nessa escolha:
- "Todo parâmetro novo que muda a saída deve estar no nome da pasta" — (b)
  cumpre isso por construção; (a) só cumpre enquanto a configuração nunca
  mudar.
- `src/results/` é versionado e apagar pastas ali exige confirmação explícita
  do usuário. Na estratégia (a), a "limpeza manual" é, portanto, uma decisão a
  ser pedida, não um passo automático; e, depois de qualquer mudança em
  `src/results/`, rodar `concatenate_results.py` para manter `src/results/all/`
  sincronizado.

## 6. `run_parallel.py` — a única mudança estrutural

### 6.1 Bug latente: jobs de RF seriam descartados em silêncio

As linhas 259-260 particionam os jobs por igualdade exata de string:

```python
xgb_jobs = [j for j in jobs if j['model'] == 'xgboost']
lr_jobs  = [j for j in jobs if j['model'] == 'logistic_regression']
```

Um job com `model == 'random_forest'` não entra em nenhuma das duas listas.
Ele não é submetido a pool algum, nenhuma exceção é levantada, e o script
imprime `Finished. N/N succeeded.` normalmente — apenas sem ter rodado RF.

**Correção**: substituir o particionamento por um mapa modelo→pool com
fallback explícito, de forma que um modelo novo caia num pool conhecido em
vez de desaparecer. Algo na linha de:

```python
POOL_BY_MODEL = {
    'xgboost': 'gpu',
    'random_forest': 'rf',
}
DEFAULT_POOL = 'cpu'   # logistic_regression e qualquer modelo futuro
```

Um `assert` de que a soma dos tamanhos dos pools é igual a `len(jobs)` fecha
a classe inteira de bugs.

### 6.2 Pool e oversubscription

RF precisa de pool próprio (`--max_rf_workers`, default `1`), não do pool de
LR, por dois motivos:

- Com `n_jobs=-1`, cada processo de RF já usa todos os núcleos. Dois
  processos simultâneos causam oversubscription e ficam mais lentos que um
  sozinho.
- O `apr_gate` existente (semáforo de `adaptative_rw`) é ortogonal ao modelo
  e não protege contra isso.

Se a Fase 2 mostrar que a memória é o limite antes da CPU, a alternativa é
fixar `n_jobs` num valor menor na seção 4.2 e subir `--max_rf_workers`.

### 6.3 Tamanho da grade

`MODELS` passa de 2 para 3 valores, multiplicando a grade completa por 1,5:
hoje são 160 cenários **já contando os 2 modelos** (10 combinações de
técnica/janela/batch_type × 2 modelos × 2 `fairness_correction` × 4
`train_window`); com RF passam a 240. Não rodar a grade cheia de RF de
imediato — ver 9.4.

## 7. Criação de arquivos e pastas — nada a escrever

Verificado ponto a ponto: **nenhum código de criação de arquivo é
necessário**. `model_name` é apenas uma string interpolada nos nomes de pasta:

- `metrics_utils._run_folder_name`, `baseline_non_abc_folder_name` e
  `baseline_dynamic_folder_name` (linhas 77-101) interpolam `model_name` sem
  validar contra lista alguma.
- As três funções `save_*` (linhas 104-131) chamam
  `os.makedirs(..., exist_ok=True)`.
- `generate_all_charts` monta `subfolder` da mesma forma, sobre `_FIGURES_DIR`.

Na primeira execução nascem sozinhas, por exemplo:

```
src/results/IDADE_BIN/random_forest_alpha_0.5_sma_window_5_monthly_kamiran_rw_trainwindow_none/
src/results/IDADE_BIN/baseline_non_abc_random_forest_monthly_kamiran_rw_fw_5/
src/results/IDADE_BIN/baseline_dynamic_random_forest_monthly_kamiran_rw_trainwindow_none_fw_5/
src/figures/IDADE_BIN/random_forest_alpha_0.5_sma_window_5_monthly_kamiran_rw_trainwindow_none/
```

com as três CSVs de sempre (`assessment_all_baselines.csv`,
`temporal_results_df.csv`, `absolute_results_df.csv`) e os PNGs de sempre.
A coluna `model_name` é preenchida por `main._attach_metadata`, sem mudança.

## 8. Impacto nas ferramentas de `src/tools/`

Cada ferramenta foi verificada contra o nome `random_forest` — que, como
`logistic_regression`, contém underscore, o caso que quebraria um parser
ingênuo.

| Ferramenta | Veredito |
|---|---|
| `concatenate_results.py` | **Sem mudança.** `_ABC_RE` usa `(?P<model>.+)` guloso terminado em `_alpha_`; `_BASELINE_RE` usa `(?P<model>.+)_(?P<batch>monthly\|weekly)`. Ambos já resolvem `logistic_regression`. Verificar na Fase 2 mesmo assim. |
| `regenerate_charts.py` | **Sem mudança.** Lê `model_name` da coluna da CSV (linha 80); nunca compara contra uma lista. |
| `alpha_analysis.py` | **Sem mudança obrigatória.** `sorted(df['model_name'].unique())` ganha a terceira coluna sozinho. Efeito colateral: `figsize=(max(13, 8 * n_models), …)` leva a figura a ~24 polegadas de largura. Quebrar em duas linhas é melhoria opcional, posterior. |
| `vanilla_shift_chart.py` | **Mudança necessária**: `choices` do `--model_name` (linha 164). |
| `migrate_legacy_schema.py` | **Sem mudança.** Só toca pastas do schema antigo. |
| `migrate_train_window_schema.py` | **Sem mudança.** Idem; declara-se idempotente para pastas que já têm `_trainwindow_`. |
| `migrate_forecast_window_schema.py` | **Sem mudança.** Idem. |

Os três `migrate_*` devem mesmo assim ser rodados em `--dry-run` após a
primeira pasta de RF existir (passo 8 da seção 9), para confirmar que a
ignoram em vez de tentar renomeá-la.

## 9. Plano de execução

### 9.1 Fase 1 — código

1. Registrar na seção 5 a decisão (a) ou (b).
2. `src/utils/model_utils.py`: `from sklearn.ensemble import RandomForestClassifier`
   e a entrada do `models_mapping` conforme 4.2.
3. `src/utils/general_utils.py:18`: adicionar `'random_forest'` aos `choices`.
4. `src/tools/vanilla_shift_chart.py:164`: idem.
5. `uv run ruff check src/` sem erros **novos** (há 14 pré-existentes, em
   notebooks, `tools/` e `general_utils.py:74` — não corrigir nesta mudança) e
   `uv run ruff format` **apenas nos arquivos tocados**. Não rodar
   `ruff format src/`: hoje ele reformataria 14 arquivos não relacionados.

Nenhum outro arquivo em `src/utils/` muda. `metrics_utils.py` em particular
**não** muda.

### 9.2 Fase 2 — validação barata (antes de gastar horas)

6. `uv run python src/main.py --test_run True --alpha 0.5 --baseline_type all
   --model_name random_forest --sensitive_attribute IDADE_BIN` — medir tempo
   por batch e **pico de RAM** do bloco `dynamic`. Ajustar `max_depth` /
   `n_estimators` a partir daqui, antes de qualquer run definitivo.
7. Conferir as pastas geradas contra a seção 7 e a coluna `model_name ==
   'random_forest'` nas três CSVs.
8. `uv run python src/tools/concatenate_results.py` e confirmar que as pastas
   de RF foram parseadas (não ignoradas). Rodar os três `migrate_*.py
   --dry-run` e confirmar que não propõem mexer nelas.
9. Repetir 6 com `--fairness_correction adaptative_rw --apr_iterations 3`.

### 9.3 Nota de custo sobre APR + RF

APR treina o modelo `apr_iterations + 1` vezes por batch (um `fit` inicial +
`T` re-treinos; default `T=20`, `model_utils.adaptive_priority_reweighing`).
Com RF isso são 21 florestas completas por batch, sobre o histórico acumulado.

E o número de batches treinados não é pequeno: `dynamic`/`abcinml` treinam em
todo batch após o warm-up — **81** no mensal com `forecast_window=5` (86 − 5),
mas **~357** no semanal (379 − 22). A célula semanal + APR + `train_window
none` é, de longe, a mais cara da grade. O passo 9 existe para dimensionar isso
com `T=3` antes de liberar `T=20` no sweep.

### 9.4 Fase 3 — grade paralela

10. `run_parallel.py`: `'random_forest'` em `MODELS`, roteamento por mapa
    (6.1) e `--max_rf_workers` (6.2).
11. Primeiro sweep **restrito**, não a grade cheia:
    `--fairness_corrections kamiran_rw --train_windows 6 12`. Limitar o
    `train_window` mantém o custo por batch constante e é o que torna RF
    viável antes de saber o perfil real de memória.
12. Avaliar os números do passo 11 antes de decidir se APR + RF e
    `train_window none` entram na grade.

### 9.5 Fase 4 — documentação (no mesmo commit do código)

O `CLAUDE.md` exige que mudanças de CLI/saídas/pipeline atualizem o
`README.md` **na mesma mudança** — esta fase não é um passo posterior. O
`CLAUDE.md` não descreve mais CLI nem `model_utils` (delega ao README), então
não precisa mudar.

13. `README.md`, por seção:
    - **CLI Reference**: `choices` de `--model_name`.
    - **Parallel Grid Run**: descrição dos pools (`--max_rf_workers`), tabela
      da grade e a contagem de cenários (160 → 240).
    - **Installation & Usage**: requisitos (RF é CPU, sem GPU).
    - **Dependencies**: linha de `scikit-learn` passa a citar Random Forest.

## 10. Limitação pré-existente, registrada de passagem

`src/tools/concatenate_results.py:16` fixa
`SOURCE_DIR = RESULTS_ROOT / 'IDADE_BIN'`. Resultados de `CS_RACA` nunca são
agregados. Não tem relação com Random Forest e não deve ser corrigido nesta
mudança, mas vale um item próprio se `CS_RACA` voltar ao escopo.

## 11. Critérios de aceitação (ver §12 para o status)

- `--model_name random_forest` roda as 4 baselines ponta a ponta e grava as
  três CSVs e os PNGs nas pastas da seção 7.
- `concatenate_results.py` agrega as pastas de RF com `model_name` correto.
- `alpha_analysis.py` renderiza três colunas de modelo.
- `run_parallel.py` submete jobs de RF (contagem impressa bate com a grade) e
  nenhum job é silenciosamente descartado.
- Dois runs idênticos de RF produzem métricas idênticas (semente fixa).
- Nenhum resultado pré-existente de LR ou XGB foi alterado, renomeado ou
  apagado.

## 12. Status da implementação (2026-09-26)

Fases 1, 2 e 4 concluídas; Fase 3 com o código pronto e o sweep ainda por rodar.

**Medições da Fase 2** (processo isolado por configuração, 100 árvores):

| Configuração | vanilla mensal: fração prevista positiva | vanilla semanal | histórico completo (3,79M): `fit` | pico de RAM |
|---|---|---|---|---|
| `leaf=50, depth=12` (proposta original) | **0,000** | **0,000** | 163 s | 4,62 GB |
| `leaf=1, depth=12` (**adotada**) | 0,028 | 0,027 | 149 s | 4,61 GB |
| `leaf=10, depth=12` | 0,005 | 0,000 | 151 s | 4,45 GB |
| `leaf=1, depth=None` | 0,064 | 0,054 | 247 s | 6,50 GB |

Taxa real de positivos no batch seguinte: 0,154 (mensal) e 0,107 (semanal);
LR prevê 0,071 no mesmo batch. O pico de RAM é dominado pela cópia dos dados,
não pelas árvores — por isso `min_samples_leaf` não protege memória. Mesmo
adotada, a configuração `depth=12` ainda subprevê positivos nos batches
pequenos; `depth=None` (candidata a `random_forest_deep`, estratégia (b)) é a
alternativa se isso se mostrar relevante nos resultados.

Estimativa de custo derivada: ~150 s por `fit` no histórico completo ⇒
`dynamic` mensal sem `train_window` ≈ 81 `fit`s de histórico crescente ≈ 1–2 h
por run com `kamiran_rw`; semanal (~357 `fit`s) e APR (×21) multiplicam isso.
Reforça o passo 11: começar por `--train_windows 6 12`.

**Critérios de aceitação:**
- ✓ 4 baselines ponta a ponta com `--test_run True --alpha 0.5`, para
  `kamiran_rw` e para `adaptative_rw --apr_iterations 3`; três CSVs com
  `model_name == 'random_forest'` e PNGs gerados. Validação feita com CWD fora
  do repositório — `--test_run` não entra no nome da pasta, então um teste
  dentro do repo deixaria em `src/results/` pastas que o run real carregaria
  como cache.
- ✓ `concatenate_results._parse_folder` reconhece as 16 variantes de nome de
  RF (baseline/dynamic/abcinml × mensal/semanal × kamiran/APR, com e sem ema e
  `train_window`); os três `migrate_*` não as tocam.
- ✓ `run_parallel.py`: 240 cenários, 80 por pool (cpu/gpu/rf); `assert` de
  roteamento.
- ✓ Determinismo: dois runs de teste → 9 CSVs idênticas byte a byte.
- ✓ Nenhum resultado de LR/XGB tocado.
- ☐ `alpha_analysis.py` com três colunas — depende de resultados reais de RF.
- ☐ Sweep da Fase 3 (passos 11–12).
