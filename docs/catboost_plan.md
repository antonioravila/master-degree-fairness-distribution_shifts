# Plano: incluir CatBoost como `--model_name`

Documento de design — descreve **como** adicionar `catboost` ao eixo de modelos,
antes de qualquer mudança de código.

Quarto da série, depois de `docs/random_forest_plan.md`, `docs/svm_plan.md` e
`docs/lightgbm_plan.md`. Tecnicamente integra sem bloqueio, como o LightGBM. Mas
é o plano com a **justificativa científica mais fraca dos quatro** (§1.1) e o
único com **dois efeitos colaterais operacionais concretos** que precisam ser
neutralizados na própria configuração do modelo (§4).

Todos os números foram **medidos** em
`src/data/SRAG/cache/IDADE_BIN_YEARMONTH_51978cb343.parquet` (3.791.353 linhas,
58 features `int8`), com `catboost 1.2.10` num ambiente isolado.

## 1. Motivação

CatBoost é gradient boosting com dois diferenciais de projeto: **ordered
boosting** (permutações para evitar o vazamento de alvo que o boosting clássico
introduz ao estimar gradientes nos mesmos dados em que treina) e **tratamento
nativo de features categóricas** via estatísticas de alvo ordenadas, em vez de
one-hot.

### 1.1 A objeção, que aqui é séria

Este é o **terceiro modelo de boosting de árvores** da grade, depois de XGBoost e
LightGBM. E, pior que isso: **os dois diferenciais do CatBoost estão
estruturalmente indisponíveis neste pipeline.**

**O tratamento categórico não tem em que trabalhar.** `data_utils._convert_types`
(linhas 87-93) já faz one-hot com `drop_first=True` nas 7 colunas
multi-categóricas, produzindo 26 dummies `int8`:

| coluna de origem | dummies geradas |
|---|---|
| `AVE_SUINO` | 2 |
| `CS_GESTANT` | 6 |
| `TP_ANTIVIR` | 2 |
| `SUPORT_VEN` | 2 |
| `RAIOX_RES` | 5 |
| `TOMO_RES` | 5 |
| `CLASSI_FIN` | 4 |
| **total** | **26** |

O CatBoost recebe um `X` onde todas as 58 colunas são numéricas binárias. Sem
`cat_features`, ele é apenas mais um booster de histograma — a §3 discute o que
custaria mudar isso, e a resposta é: mais do que vale.

**O ordered boosting se autodesativa nesta escala.** O default `boosting_type`
escolhe `Plain` para datasets grandes justamente porque `Ordered` é caro. Medido
com 1M de linhas: `Ordered` leva **28,4 s** contra **7,8 s** do default — 3,6×.
Ou seja, ou se aceita o default e não se usa o diferencial, ou se força
`Ordered` e se paga 3,6× por ele.

**Recomendação honesta:** priorizar Random Forest (bagging) e SVM (perda de
margem), que ampliam de verdade o espaço de hipóteses. Entre os boosters, um
segundo já é discutível (ver §1.1 de `docs/lightgbm_plan.md`); um terceiro, cujos
dois diferenciais não se aplicam, é difícil de defender numa banca.

Se ainda assim houver razão para incluí-lo — por exemplo, porque CatBoost é
padrão de comparação na literatura de tabular e sua ausência seria questionada —
então há uma justificativa defensável, **mas ela tem de ser a de §1.2, não a de
marketing da biblioteca.**

### 1.2 A única justificativa que se sustenta aqui

Com `boosting_type='Ordered'` explicitamente ativado, CatBoost passa a testar
uma hipótese concreta e relevante para esta dissertação: **o vazamento de alvo do
boosting clássico afeta a estabilidade das métricas de fairness sob deriva
temporal?** `dynamic` e `abcinml` retreinam a cada batch sobre histórico
acumulado, e um estimador de gradiente enviesado poderia degradar de forma
desigual entre grupos conforme a distribuição muda. Ordered boosting é a correção
direta desse viés, e nenhum outro modelo da grade a oferece.

Isso custa 3,6× o tempo e exige `boosting_type='Ordered'` fixado — ver §5.2. É a
única configuração em que CatBoost acrescenta algo que XGBoost e LightGBM não
acrescentam.

## 2. Escopo

**Dentro do escopo:**
- Nova dependência `catboost`, com a verificação do §3.1.
- Neutralização dos dois efeitos colaterais do §4 — obrigatória, não opcional.
- Decisão explícita entre `Ordered` (justificável, §1.2) e o default (mais um
  booster).
- Entrada em `initialize_model`, `choices` do CLI, pool em `run_parallel.py`.

**Fora do escopo (não fazer nesta mudança):**
- **Não desfazer o one-hot de `_convert_types` para usar `cat_features`** — §3.3
  explica por que o custo é desproporcional.
- Não tunar `learning_rate`/`depth` por busca automatizada.
- Não mexer nos nomes de pasta, no schema das CSVs, nem em `metrics_utils.py`.

## 3. Dependência

### 3.1 Footprint pesado

```bash
uv add catboost
```

Ao montar o ambiente isolado desta análise, `catboost` trouxe: `graphviz`,
`matplotlib`, `numpy`, `pandas`, `plotly`, `narwhals`, `pillow`, `kiwisolver`,
`pyparsing`, `scipy`, `six`, `scikit-learn`, `threadpoolctl`. Conferido contra
o `uv.lock` atual: `pillow`, `kiwisolver` e `narwhals` **já estão** no grafo
(dependências transitivas de `matplotlib` e do grupo `dev`), então os realmente novos são **`plotly` e
`graphviz`** — o stack de visualização próprio do CatBoost, que este projeto não
usa e não vai usar.

Comparação: `lightgbm` traz apenas `numpy` e `scipy`.

**Vale o mesmo alerta da §3.2 de `docs/lightgbm_plan.md`, e com mais força:**
inspecionar o diff de `uv.lock` antes de commitar. O ambiente de benchmark
resolveu `pandas 3.0.6`, `numpy 2.5.3` e `scikit-learn 1.9.1` — todos à frente
do que o projeto usa hoje (`pandas 3.0.1`, `numpy 2.4.3`, `scikit-learn 1.8.0`,
`xgboost 3.2.0`, conferidos no `uv.lock`). Um `uv add catboost` com esse footprint tem chance
real de re-resolver o grafo e mover `pandas`, `numpy`, `scikit-learn` ou
`xgboost`, o que invalidaria a comparabilidade de **todos** os resultados já
versionados sem que o cache de pasta do `main.py` perceba.

Se o resolvedor mover qualquer dessas quatro: ou pinar as versões atuais, ou
aceitar e re-gerar a grade afetada. Registrar a decisão aqui, não no lockfile.

### 3.2 Sem GPU (na prática)

CatBoost tem suporte a GPU (`task_type='GPU'`), mas depende de build e driver
compatíveis, e o ganho é modesto em datasets desta largura (58 features
binárias). Tratar como modelo de **CPU + multithreading**, como LightGBM e RF.
Não investir em GPU sem antes ter os números de CPU.

### 3.3 Por que não desfazer o one-hot

A tentação é óbvia: passar `AVE_SUINO`, `CS_GESTANT`, `TP_ANTIVIR`,
`SUPORT_VEN`, `RAIOX_RES`, `TOMO_RES` e `CLASSI_FIN` crus como `cat_features`,
liberando o diferencial do CatBoost. Três razões para não fazer isso nesta
mudança:

1. **Quebra a comparabilidade.** O `X` do CatBoost passaria a ter um espaço de
   features diferente do de todos os outros modelos. Qualquer diferença de
   fairness observada seria um efeito conjunto de *modelo + codificação*,
   indistinguíveis. A pergunta da dissertação é sobre o modelo.
2. **Invalida o cache de pré-processamento de todo mundo.**
   `data_utils._preprocessing_fingerprint()` faz hash do código-fonte de
   `_convert_types`; mexer nele muda o fingerprint e força o re-processamento
   para **todos** os modelos e ambos os atributos sensíveis.
3. **`drop_first=True` não é reversível sem reescrever `_convert_types`.** A
   categoria de referência foi descartada; não se reconstrói a coluna original a
   partir das dummies sem voltar ao parquet bruto.

Se essa comparação interessar, ela é um experimento **próprio** — "efeito da
codificação de categóricas sobre fairness sob shift" — com plano próprio, não um
detalhe da adição de um modelo.

## 4. Dois efeitos colaterais que precisam ser neutralizados

Nenhum dos três planos anteriores tem equivalente a isto. Ambos foram
reproduzidos, não supostos.

### 4.1 CatBoost escreve `catboost_info/` no diretório de trabalho

Medido: um `fit` sem `allow_writing_files=False` cria um diretório
`catboost_info/` no CWD. O `run_parallel.py` executa `uv run src/main.py` a partir
da raiz do repositório, então:

- o diretório aparece **na raiz do projeto**, não versionado e não esperado;
- com jobs em paralelo, **múltiplos processos escrevem no mesmo caminho**
  simultaneamente;
- em `dynamic`/`abcinml` são 81 `fit`s por run no mensal (`forecast_window=5`)
  e ~357 no semanal, e com APR, 21× isso.

**Correção obrigatória:** `allow_writing_files=False` na configuração do modelo.
Não é preferência de estilo — sem isso, rodar a grade polui o repositório e cria
uma corrida de escrita entre jobs.

### 4.2 Verbosidade real (ao contrário do LightGBM)

Medido: **101 linhas de stdout para um único `fit`** de 100 iterações, sem passar
`verbose`. As primeiras linhas:

```
Learning rate set to 0.114413
0:	learn: 0.6003300	total: 8.47ms	remaining: 839ms
```

No plano do LightGBM eu levantei essa hipótese e o teste a derrubou. Aqui ela se
confirma. O `run_parallel.py` redireciona `stdout`/`stderr` para arquivo de log;
um job de `abcinml` com APR faria ~81 batches × 21 `fit`s × ~101 linhas ≈
**170 mil linhas de log por job** no mensal — e ~760 mil no semanal (~357
batches) —, o que torna os logs inúteis para diagnóstico e consome disco sem
necessidade.

**Correção obrigatória:** `verbose=False`.

## 5. Desempenho medido e hiperparâmetros

### 5.1 Onde CatBoost se posiciona

Todos com `n = 1.000.000` linhas e `sample_weight`, medidos na mesma máquina que
os demais planos:

| Estimador | tempo |
|---|---|
| `XGBClassifier(device='cuda')` | 2,1 s |
| `LGBMClassifier()` | 3,5 s |
| `XGBClassifier(tree_method='hist')` CPU | 5,6 s |
| **`CatBoostClassifier(iterations=100)`** — default `boosting_type` | **7,8 s** |
| `LogisticRegression` | 8,8 s |
| `CatBoostClassifier(iterations=100, boosting_type='Plain')` | 9,0 s |
| `LinearSVC(dual=False)` | 14,3 s |
| **`CatBoostClassifier(iterations=100, boosting_type='Ordered')`** | **28,4 s** |
| **`CatBoostClassifier()`** — **default puro, `iterations=1000`** | **74,7 s** |

Três leituras:

- **O default de `iterations` é 1000, não 100.** XGBoost e LightGBM usam 100.
  Deixar o default é rodar 10× o trabalho dos outros e produzir uma comparação
  sem sentido — 74,7 s contra 3,5 s do LightGBM não mede o algoritmo, mede o
  orçamento de árvores. **`iterations=100` é obrigatório para comparabilidade.**
- Com orçamento igualado, CatBoost é ~2,2× o LightGBM e ~1,4× o XGBoost em CPU.
  Viável na grade.
- `Ordered` (a justificativa do §1.2) custa 3,6× o default. Ainda viável, mas é
  a célula mais cara da grade de boosters.

### 5.2 Hiperparâmetros propostos

```python
'catboost': cb.CatBoostClassifier(
    iterations=100,            # igual a XGB/LGBM — sem isso o default é 1000 (§5.1)
    boosting_type='Ordered',   # a única justificativa que se sustenta (§1.2)
    random_seed=42,
    verbose=False,             # obrigatório (§4.2)
    allow_writing_files=False, # obrigatório (§4.1)
),
```

- **`iterations=100`** — comparabilidade com XGBoost e LightGBM, ver §5.1.
- **`boosting_type='Ordered'`** — decisão a confirmar. Se a justificativa
  escolhida for a de §1.2, precisa ser explícito: o default escolheria `Plain`
  nesta escala e o modelo perderia sua razão de estar na grade. Se a decisão for
  usar o default, **então vale reconsiderar incluir CatBoost** (§1.1).
- **`random_seed=42`** — note o nome: é `random_seed`, não `random_state` como
  nos outros modelos. Verificado como suficiente: dois `fit` com a mesma semente
  produzem `predict_proba` idêntico.
- **Sem `class_weight` / `scale_pos_weight` / `auto_class_weights`** — mesma
  razão dos três planos anteriores: o balanceamento é papel da reponderação
  (Kamiran/APR) via `sample_weight`.

## 6. Compatibilidade com o pipeline — nada a adaptar (verificado)

| Contrato | Status |
|---|---|
| `fit(X, y, sample_weight=...)` | ✓ testado com pesos em [0,5; 2,0] sobre 1M linhas |
| `predict` / `predict_proba` | ✓ array (n, 2), como `compute_snapshot_metrics` exige |
| Entrada `int8` do `_downcast_dtypes` | ✓ aceita o DataFrame direto |
| Determinismo com semente fixa | ✓ `predict_proba` idêntico entre dois `fit` |
| Batches minúsculos (118, 220, 6.765 linhas) | ✓ sem preditor constante, sem avisos |

APR funciona sem alteração (só precisa de `predict_proba`). Nem `main.py` nem
`metrics_utils.py` mudam.

O teste de batches pequenos importa porque `vanilla`/`static` treinam num único
batch, e o menor batch semanal do dataset tem 118 linhas. CatBoost produziu 116
probabilidades distintas nesse tamanho — sem degeneração.

## 7. `run_parallel.py`

### 7.1 Mesmo bug latente dos três planos anteriores

Linhas 259-260: particionamento por igualdade exata de string, então um job
`'catboost'` não entra em pool algum, não roda, e o script reporta
`Finished. N/N succeeded.`. Correção única, descrita na §6.1 de
`docs/random_forest_plan.md`. Quatro planos, o mesmo bug — se algum for
implementado primeiro, resolve para todos.

### 7.2 Pool

CatBoost usa todos os núcleos por default (`thread_count=-1`), então **1 worker**,
pela mesma razão de RF e LightGBM. Como não usa GPU, pode rodar em paralelo com o
XGBoost — o ganho de escalonamento descrito na §6.2 de `docs/lightgbm_plan.md`.

Se RF, LightGBM e CatBoost forem todos adotados, os três disputam CPU entre si e
devem compartilhar **um único pool de CPU-pesada com 1 worker**, não um pool cada.
Caso contrário o paralelismo nominal vira contenção.

## 8. Criação de arquivos e cache

**Pastas de resultados:** idêntico à §7 de `docs/random_forest_plan.md` —
`model_name` é interpolado nos nomes de pasta e as funções `save_*` fazem
`os.makedirs(..., exist_ok=True)`. Nada a escrever. `catboost` não tem underscore,
então é caso simples para as regexes de `concatenate_results.py`.

**Arquivo indesejado:** ver §4.1 — `catboost_info/` na raiz do repo, se
`allow_writing_files=False` for esquecido. Se algum run acontecer antes da
correção, apagar o diretório e considerar adicioná-lo ao `.gitignore` como rede
de segurança.

**Cache de hiperparâmetros:** vale a §5 de `docs/random_forest_plan.md` — o nome
da pasta não codifica hiperparâmetros. Aqui há um risco específico e grave:
**`boosting_type` é a decisão central do §1.2 e não aparece no nome da pasta.**
Rodar com `Ordered`, depois trocar para `Plain`, faria o `main.py` carregar as
CSVs de `Ordered` e reportar sucesso — e a diferença entre as duas é exatamente o
que justifica o modelo.

**Recomendação: estratégia (b)** — registrar as variantes como `model_name`
distintos (`catboost_ordered`, `catboost_plain`), o que resolve o cache e permite
comparar as duas lado a lado, que é o experimento interessante do §1.2.

## 9. Plano de execução

### 9.1 Fase 0 — decidir se vale incluir

1. Ler §1.1. Se nenhuma justificativa se sustentar para esta dissertação,
   **parar aqui** e registrar a decisão de não incluir. Esse é um resultado
   válido deste plano.
2. Se seguir: escolher entre a justificativa de §1.2 (`Ordered`, 3,6× mais caro)
   e "padrão de comparação da literatura" (default). A escolha determina o §5.2.

### 9.2 Fase 1 — dependência e código

3. `uv add catboost`; **inspecionar o diff de `uv.lock`** e resolver o §3.1
   antes de commitar, com atenção a `pandas`, `numpy`, `scikit-learn` e
   `xgboost`.
4. `model_utils.py`: `import catboost as cb` + entrada do §5.2, com
   `verbose=False` e `allow_writing_files=False` — os dois não são opcionais.
5. `choices` em `general_utils.py:18` e `vanilla_shift_chart.py:164`.
6. `uv run ruff check src/` sem erros **novos** (há 14 pré-existentes) e
   `uv run ruff format` só nos arquivos tocados — `ruff format src/` hoje
   reformataria 14 arquivos não relacionados.

### 9.3 Fase 2 — validação

7. **Se o passo 3 moveu `pandas`/`numpy`/`sklearn`/`xgboost`:** re-rodar um
   cenário já existente de `logistic_regression` **e** um de `xgboost`, comparando
   com as CSVs versionadas, antes de qualquer run de CatBoost. Num `git
   worktree` com as pastas desses cenários removidas — ver o cuidado com o
   cache de pasta no passo 8 da §10.3 de `docs/svm_plan.md`. Se `pandas` mudou,
   o cache de pré-processamento **não** é invalidado (o fingerprint cobre só o
   código-fonte das funções de `data_utils`); rodar com `use_cache=False`.
8. `--test_run True --alpha 0.5 --baseline_type all --model_name catboost`.
   Conferir explicitamente que **nenhum `catboost_info/` apareceu** na raiz e que
   o log tem tamanho normal — a verificação dos dois itens do §4.
9. Conferir pastas e coluna `model_name` nas três CSVs; rodar
   `concatenate_results.py` e os três `migrate_*.py --dry-run`.
10. `--fairness_correction adaptative_rw --apr_iterations 3` para dimensionar o
    custo. Com `Ordered`, esta é a célula mais cara de toda a grade (21 `fit`s por
    batch com `T=20`, ~357 batches no semanal): estimar antes de liberar `T=20`.

### 9.4 Fase 3 — grade

11. `'catboost'` em `MODELS`; pool conforme §7.2, atento à contenção de CPU se RF
    e LightGBM também estiverem na grade.
12. Começar por `--fairness_corrections kamiran_rw`. Com `Ordered` a 28,4 s/1M,
    considerar `--train_windows 6 12` para a primeira rodada, em vez de histórico
    ilimitado.

### 9.5 Fase 4 — documentação (no mesmo commit do código)

13. `README.md` — exigido pelo `CLAUDE.md` na mesma mudança (o `CLAUDE.md` em
    si não descreve mais CLI nem `model_utils`): seções **CLI Reference**
    (`--model_name`), **Parallel Grid Run** (pool e contagem de cenários),
    **Installation & Usage** e **Dependencies** (linha nova `catboost`).
14. Na dissertação: registrar a justificativa escolhida no §1.1/§1.2, o
    `boosting_type` usado, o `iterations=100` (e por que não o default de 1000), e
    **explicitamente que as features categóricas chegam one-hot, de modo que o
    tratamento nativo do CatBoost não está em uso** (§1.1). Um avaliador que
    conheça a biblioteca vai perguntar exatamente isso.

## 10. Critérios de aceitação

- A decisão da Fase 0 está registrada — incluir com qual justificativa, ou não
  incluir.
- `pyproject.toml` e `uv.lock` atualizados, com a questão de versões do §3.1
  resolvida explicitamente.
- `--model_name catboost` roda as 4 baselines ponta a ponta, gravando as três
  CSVs e os PNGs nas pastas de sempre.
- **Nenhum diretório `catboost_info/` é criado em lugar algum do repositório.**
- Os logs de `run_parallel.py` para jobs de CatBoost têm tamanho comparável aos
  dos outros modelos (ou seja, `verbose=False` está de fato ativo).
- `iterations` está explicitamente em 100 — nenhum run com o default de 1000.
- Dois runs idênticos produzem métricas idênticas (semente fixa).
- Nenhum resultado de LR, XGB, RF, SVM ou LightGBM foi alterado, renomeado ou
  apagado; se alguma dependência compartilhada mudou de versão, a decisão sobre
  re-gerar está registrada.
