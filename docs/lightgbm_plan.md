# Plano: incluir LightGBM como `--model_name`

Documento de design — descreve **como** adicionar `lightgbm` ao eixo de modelos,
antes de qualquer mudança de código.

Terceiro da série, depois de `docs/random_forest_plan.md` e `docs/svm_plan.md`.
Em termos de integração, é **o mais fácil dos três**: `LGBMClassifier` cumpre
todos os contratos do pipeline sem adaptação (ao contrário do SVM, §4 de lá) e
escala sem restrição (ao contrário do kernel SVM, §3.1 de lá). Em compensação, é
o único que **acrescenta uma dependência nova** (§3) e o único cuja justificativa
científica precisa ser argumentada em vez de assumida (§1.1).

Todos os tempos foram **medidos** em
`src/data/SRAG/cache/IDADE_BIN_YEARMONTH_51978cb343.parquet` (3.791.353 linhas,
58 features `int8`, 86 batches mensais / 379 semanais), num ambiente isolado com
`lightgbm 4.7.0`.

## 1. Motivação

LightGBM é gradient boosting com crescimento **leaf-wise** (guiado pela maior
redução de perda) e binning por histograma, contra o crescimento **level-wise**
do XGBoost. Na prática produz árvores mais profundas e assimétricas para o mesmo
orçamento de nós.

### 1.1 A objeção honesta, e a resposta

Diferente de Random Forest (bagging) e SVM (perda de margem), **LightGBM não
abre uma família nova**: é boosting de árvores, como o XGBoost que já está na
grade. Vale dizer isso explicitamente, porque um avaliador de dissertação vai
perguntar "por que dois modelos de boosting?".

Há duas respostas defensáveis, e convém escolher uma antes de rodar:

1. **Argumento de fairness, não de acurácia.** O crescimento leaf-wise concentra
   capacidade onde a perda é maior. Num dataset com 23,2% de positivos e grupos
   sensíveis desbalanceados, isso significa que LightGBM e XGBoost alocam
   capacidade de forma diferente entre subgrupos, mesmo com acurácia agregada
   parecida. A pergunta "duas implementações de boosting com o mesmo desempenho
   médio produzem o mesmo perfil de fairness sob shift?" é uma pergunta legítima
   — e, se a resposta for não, é um resultado interessante para a dissertação.
2. **Argumento operacional (§6.2).** O XGBoost roda hoje com `device='cuda'`, o
   que limita a grade a 1 worker de GPU. LightGBM é CPU e **não disputa a GPU**,
   então um segundo modelo de boosting pode rodar em paralelo com o primeiro, em
   vez de enfileirar atrás dele.

Se nenhuma das duas convencer, a recomendação sincera é **priorizar RF e SVM**,
que ampliam o espaço de hipóteses de verdade, e deixar LightGBM para depois. O
resto deste plano assume que a decisão foi tomada a favor.

## 2. Escopo

**Dentro do escopo:**
- Nova dependência `lightgbm`, com a verificação de não-regressão do §3.2.
- Entrada em `initialize_model`, `choices` do CLI, pool em `run_parallel.py`.
- Escolha de hiperparâmetros que torne a comparação com XGBoost honesta (§5.2).

**Fora do escopo (não fazer nesta mudança):**
- Não tentar LightGBM em GPU — a wheel do PyPI é CPU-only (§3.3).
- Não tunar `num_leaves`/`learning_rate` por busca automatizada.
- Não mexer nos nomes de pasta, no schema das CSVs, nem em `metrics_utils.py`.
- Não migrar nada.

## 3. Dependência nova — o único dos três planos que tem uma

### 3.1 O que adicionar

```bash
uv add lightgbm
```

`pyproject.toml` hoje lista `xgboost>=3.2.0` mas nenhum `lightgbm`. É a primeira
vez nesta série de planos que `pyproject.toml` e `uv.lock` mudam.

### 3.2 Risco não-óbvio: `uv add` pode subir o XGBoost junto

Ao montar o ambiente isolado de benchmark desta análise, o resolvedor trouxe
**`xgboost 3.4.1`**, enquanto o ambiente do projeto tem **`xgboost 3.2.0`**
(`pyproject.toml` permite `>=3.2.0`, sem teto). Um `uv add lightgbm` pode
re-resolver o grafo e atualizar o XGBoost de carona.

Isso importa porque **todos os resultados de XGBoost já versionados foram
produzidos com a 3.2.0**. Uma mudança de versão pode alterar defaults ou
desempate numérico, e o cache de pasta do `main.py` não perceberia: os resultados
antigos continuariam sendo carregados e comparados com resultados novos gerados
por outra versão da biblioteca.

**Mitigação obrigatória**, na Fase 1:
1. Rodar `uv add lightgbm` e **inspecionar o diff de `uv.lock`** antes de
   commitar.
2. Se o XGBoost tiver subido de versão, ou reverter (`uv add "xgboost==3.2.0"`
   para pinar), ou aceitar conscientemente e **re-rodar os cenários de XGBoost**,
   registrando a decisão aqui.
3. Não deixar a decisão implícita no lockfile.

### 3.3 Sem GPU

A wheel do PyPI é compilada sem suporte a GPU (requer build próprio com
OpenCL/CUDA). Não existe equivalente ao `device='cuda'` do XGBoost. LightGBM é,
para efeitos deste projeto, um modelo de **CPU + multithreading**, como o Random
Forest — o que define o pool do §6.2.

## 4. Compatibilidade com o pipeline — nada a adaptar (verificado)

Ao contrário do SVM, não há bloqueio algum. Cada item abaixo foi testado, não
assumido:

| Contrato | Status |
|---|---|
| `fit(X, y, sample_weight=...)` | ✓ testado com pesos em [0,5; 2,0] sobre 1M linhas |
| `predict(X)` / `predict_proba(X)` | ✓ array (n, 2), como `compute_snapshot_metrics` exige |
| Entrada `int8` do `_downcast_dtypes` | ✓ aceita o DataFrame direto, sem conversão nem aviso |
| Determinismo com semente fixa | ✓ dois `fit` com `random_state=42` → `predict_proba` idêntico |

Consequência: **APR (`adaptative_rw`) funciona sem alteração**, já que
`compute_margins` só precisa de `predict_proba`. E nenhuma linha de
`metrics_utils.py` ou `main.py` muda.

### 4.1 Duas hipóteses de risco que foram testadas e descartadas

Registrado para que ninguém refaça a investigação:

**Batches pequenos não quebram o modelo.** `vanilla`/`static` treinam num único
batch, e os menores do dataset são pequenos de verdade: 981 linhas no menor
batch mensal (2019-01) e **118 linhas** no menor semanal (2019-52). A hipótese
era que o crescimento leaf-wise com `num_leaves=31` e `min_child_samples=20` se
recusaria a dividir, devolvendo um preditor constante — o que daria AUC ≈ 0,5
silenciosamente. Medido em 220, 1.783 e 6.765 linhas: LightGBM constrói as 100
árvores, produz 211/1.491/4.749 probabilidades distintas e **não emite aviso
algum**. XGBoost se comporta praticamente igual nos mesmos tamanhos. Risco
inexistente.

**Não há enxurrada de log.** A hipótese era que os avisos do LightGBM
("No further splits with positive gain") poluiriam os logs de
`run_parallel.py`, que redireciona `stdout`/`stderr` para arquivo em até ~357
batches semanais × 21 `fit`s de APR. Não reproduzido em nenhuma das configurações testadas.
Ainda assim, `verbose=-1` continua recomendado no §5.2 como seguro barato — o
LightGBM escreve pelo stream C++, que nem sempre é capturável do lado Python, e
custa nada.

## 5. Desempenho medido e hiperparâmetros

### 5.1 A tabela que decide o pool

Todos com `n = 1.000.000` linhas e `sample_weight`:

| Estimador | tempo | observação |
|---|---|---|
| `XGBClassifier(device='cuda')` — config atual | **2,1 s** | usa a GPU |
| `LGBMClassifier()` (n_jobs=-1, default) | **3,5 s** | todos os núcleos |
| `XGBClassifier(tree_method='hist')` — CPU | 5,6 s | referência de CPU |
| `LGBMClassifier(n_jobs=1)` | 10,1 s | single-thread |

Duas leituras importantes:

- LightGBM em CPU é **1,6× mais rápido que o XGBoost em CPU**, e apenas 1,6×
  mais lento que o XGBoost em GPU. É perfeitamente viável na grade completa,
  inclusive sem `--train_window`.
- LightGBM **depende fortemente de multithreading**: de 3,5 s para 10,1 s ao cair
  para um núcleo. Isso define o §6.2.

### 5.2 Hiperparâmetros propostos

```python
'lightgbm': lgb.LGBMClassifier(
    n_estimators=100,     # igual ao default do XGBoost — comparação honesta
    num_leaves=31,        # default; leaf-wise, sem max_depth explícito
    n_jobs=-1,
    random_state=42,
    verbose=-1,           # seguro barato, ver §4.1
),
```

- **`n_estimators=100`** — é o default de ambas as bibliotecas. Mantê-lo
  explícito documenta que a comparação LightGBM × XGBoost usa o mesmo orçamento
  de árvores; o que difere é *como* cada árvore cresce, que é exatamente a
  variável de interesse do §1.1.
- **`num_leaves=31` sem `max_depth`** — deliberado. Limitar a profundidade
  aproximaria o LightGBM do comportamento level-wise do XGBoost e dissolveria a
  única diferença que justifica o modelo. Se a Fase 2 mostrar overfit nos batches
  pequenos, o parâmetro a mexer é `min_child_samples`, não `max_depth`.
- **`random_state=42`** — mesma razão dos outros dois planos, e aqui confirmada
  como suficiente para reprodutibilidade (§4).
- **Sem `class_weight` / `scale_pos_weight`** — mesma razão dos outros dois
  planos: o balanceamento é papel da reponderação (Kamiran/APR) via
  `sample_weight`. Com taxa de positivos medida em 0,2323, a tentação existe;
  ceder a ela tornaria impossível isolar o efeito do `alpha` do ABCInML.

## 6. `run_parallel.py`

### 6.1 Mesmo bug latente dos outros dois planos

Linhas 259-260: particionamento de jobs por igualdade exata de string, então um
job `'lightgbm'` não entra em pool nenhum, não roda, e o script reporta sucesso.
A correção é a da §6.1 de `docs/random_forest_plan.md` (mapa modelo→pool com
fallback + `assert` de soma dos pools). Se qualquer um dos três planos for
implementado primeiro, este ponto já vem resolvido — é a mesma correção,
contada uma vez.

### 6.2 Pool: CPU, 1 worker, mas sobreposto ao XGBoost

| Modelo | recurso | `n_jobs` | workers |
|---|---|---|---|
| `xgboost` | GPU | — | 1 (VRAM) |
| `lightgbm` | CPU | `-1` | **1** (oversubscription) |
| `random_forest` | CPU | `-1` | 1 |
| `logistic_regression` / `svm` | CPU | 1 thread | vários |

LightGBM precisa de pool próprio com 1 worker pela mesma razão do RF: com
`n_jobs=-1` cada processo já toma todos os núcleos, e o ganho de 10,1 s → 3,5 s
some se dois jobs disputarem os mesmos núcleos.

**Mas há um ganho real de escalonamento**, e é o argumento operacional do §1.1:
como LightGBM não toca a GPU, um job de LightGBM e um de XGBoost rodam
**simultaneamente** sem se atrapalhar. Hoje a grade de boosting é serializada
pelo gargalo de 1 worker de GPU; com LightGBM em paralelo, a vazão de modelos de
boosting por hora praticamente dobra.

Se RF e LightGBM forem ambos adotados, porém, os dois disputam CPU entre si —
nesse caso o pool de CPU-pesada deve ser **um só**, compartilhado pelos dois, com
1 worker no total.

## 7. Criação de arquivos e pastas — nada a escrever

Idêntico à §7 de `docs/random_forest_plan.md`, e não se repete em detalhe:
`model_name` é interpolado nos nomes de pasta por
`metrics_utils._run_folder_name` / `baseline_non_abc_folder_name` /
`baseline_dynamic_folder_name`, e as três funções `save_*` chamam
`os.makedirs(..., exist_ok=True)`. As pastas de `lightgbm` nascem sozinhas em
`src/results/` e `src/figures/`, com as três CSVs e os PNGs de sempre.

`lightgbm` não tem underscore, então é um caso mais simples que
`logistic_regression` para as regexes de `concatenate_results.py` — nenhuma
mudança lá.

## 8. Cache de pasta vs. hiperparâmetros

Vale o que a §5 de `docs/random_forest_plan.md` descreve: o nome da pasta é a
chave de cache (`main.py:250` e `main.py:323`) e **não codifica
hiperparâmetros**. Trocar `num_leaves` depois de gerar resultados faz o
`main.py` carregar as CSVs antigas e reportar sucesso.

**Para LightGBM a recomendação é (a), congelar** a configuração do §5.2 — ao
contrário do SVM, onde uma varredura de `C` é provável. Aqui os defaults já são
os corretos para a pergunta do §1.1, e mexer em `num_leaves` descaracterizaria a
comparação. Se ainda assim houver varredura, usar (b) e registrar variantes como
`model_name` distintos (`lightgbm`, `lightgbm_leaves127`, …).

Atenção adicional e específica deste plano: **a versão da biblioteca também não
está no nome da pasta** (§3.2). Vale para o LightGBM tanto quanto para o
XGBoost.

## 9. Plano de execução

### 9.1 Fase 1 — dependência e código

1. Decidir e registrar a justificativa do §1.1 (argumento 1, argumento 2, ou
   adiar o modelo).
2. `uv add lightgbm`; **inspecionar o diff de `uv.lock`** e resolver o §3.2
   antes de commitar.
3. `model_utils.py`: `import lightgbm as lgb` + entrada do §5.2 no
   `models_mapping`.
4. `choices` em `general_utils.py:18` e `vanilla_shift_chart.py:164`.
5. `uv run ruff check src/` sem erros **novos** (há 14 pré-existentes) e
   `uv run ruff format` só nos arquivos tocados — `ruff format src/` hoje
   reformataria 14 arquivos não relacionados.

Nenhum outro arquivo muda. `metrics_utils.py` e `main.py` **não** mudam.

### 9.2 Fase 2 — validação

6. **Se o XGBoost subiu de versão no passo 2:** re-rodar um cenário de
   `xgboost` já existente e comparar com a CSV versionada, antes de qualquer
   coisa. Se diferir, decidir explicitamente entre pinar a versão antiga ou
   re-gerar toda a grade de XGBoost. Fazer isso num `git worktree` com as
   pastas do cenário removidas — no repositório principal o cache de pasta
   impede `vanilla`/`static`/`dynamic` de re-executar, e o `abcinml`
   sobrescreveria a CSV versionada (ver passo 8 da §10.3 de
   `docs/svm_plan.md`).
7. `--test_run True --alpha 0.5 --baseline_type all --model_name lightgbm`.
8. Conferir as pastas geradas e a coluna `model_name == 'lightgbm'` nas três
   CSVs; rodar `concatenate_results.py` e os três `migrate_*.py --dry-run`.
9. `--fairness_correction adaptative_rw --apr_iterations 3` para dimensionar o
   custo de APR (21 `fit`s por batch no default: 1 inicial + `T=20`) antes de
   liberá-lo na grade.

Os passos que os planos de RF e SVM dedicam a riscos específicos (memória do RF,
convergência do SVM) **não têm equivalente aqui** — os riscos candidatos foram
testados e descartados no §4.1.

### 9.3 Fase 3 — grade

10. `'lightgbm'` em `MODELS`; pool conforme §6.2, com atenção à interação com RF
    se ambos forem adotados.
11. Com os tempos do §5.1, LightGBM cabe na grade completa sem `--train_window`.
    Ainda assim, começar por `--fairness_corrections kamiran_rw` para ter os
    primeiros números rápido.

### 9.4 Fase 4 — documentação (no mesmo commit do código)

12. `README.md` — exigido pelo `CLAUDE.md` na mesma mudança (o `CLAUDE.md` em
    si não descreve mais CLI nem `model_utils`): seções **CLI Reference**
    (`--model_name`), **Parallel Grid Run** (pool e contagem de cenários),
    **Installation & Usage** e **Dependencies** — que neste plano ganha uma
    linha nova (`lightgbm`).
13. Na dissertação: registrar a justificativa do §1.1 e as **versões** de
    `lightgbm` e `xgboost` usadas, dado o §3.2.

## 10. Critérios de aceitação

- `pyproject.toml` e `uv.lock` atualizados, com a questão da versão do XGBoost
  (§3.2) resolvida e registrada — não deixada implícita no lockfile.
- `--model_name lightgbm` roda as 4 baselines ponta a ponta, gravando as três
  CSVs e os PNGs nas pastas de sempre.
- Nenhum resultado de LR, XGB, RF ou SVM foi alterado, renomeado ou apagado. Se
  o XGBoost mudou de versão, a decisão sobre re-gerar a grade está registrada.
- Dois runs idênticos de LightGBM produzem métricas idênticas (já verificado no
  §4, confirmar ponta a ponta).
- `concatenate_results.py` agrega as pastas de `lightgbm` com `model_name`
  correto.
- `run_parallel.py` submete os jobs de LightGBM, com a contagem impressa batendo
  com a grade, e um job de LightGBM e um de XGBoost rodando de fato em paralelo
  (§6.2).
