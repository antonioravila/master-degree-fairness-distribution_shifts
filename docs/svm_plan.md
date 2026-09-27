# Plano: incluir SVM como `--model_name`

Documento de design — descreve **como** adicionar um SVM ao eixo de modelos,
antes de qualquer mudança de código.

Irmão de `docs/random_forest_plan.md`, mas **não é o mesmo caso**. Random Forest
era uma entrada de dicionário: `RandomForestClassifier` cumpre sozinho todos os
contratos que o pipeline exige. SVM **não cumpre** — `predict_proba` não existe
em `LinearSVC`, e o SVM com kernel é computacionalmente inviável neste dataset.
As seções 3 e 4 são decisões de design reais, não escolhas de hiperparâmetro; as
demais seções dependem delas.

Todos os números de tempo deste documento foram **medidos** em
`src/data/SRAG/cache/IDADE_BIN_YEARMONTH_51978cb343.parquet` (3.791.353 linhas,
58 features + alvo, 86 batches mensais), não estimados de literatura.

## 1. Motivação

O pipeline compara hoje um modelo linear com perda logística
(`logistic_regression`) e um ensemble de boosting (`xgboost`). SVM acrescenta o
eixo da **perda de margem máxima** (hinge). O interesse para a dissertação é
específico: a fronteira de decisão do SVM é determinada pelos pontos próximos da
margem, e os grupos sensíveis não estão distribuídos uniformemente nessa região
— há razão teórica para esperar um perfil de fairness diferente do da regressão
logística, mesmo entre dois modelos lineares.

Isso só se sustenta, porém, se a escolha de variante (seção 3) preservar a perda
de margem. Um SVM calibrado até virar "quase uma regressão logística" não
responde à pergunta.

## 2. Escopo

**Dentro do escopo:**
- Decidir e implementar **uma** variante de SVM viável (seção 3).
- Resolver o bloqueio de `predict_proba` (seção 4) — obrigatório, não opcional.
- Ajustar os parâmetros de APR que dependem da escala de score (seção 5), se
  APR + SVM entrar no escopo.
- Entrada em `initialize_model`, `choices` do CLI, pool em `run_parallel.py`.

**Fora do escopo (não fazer nesta mudança):**
- **SVM com kernel RBF/polinomial** — inviável, com medição na seção 3.1. Não
  tentar "só no vanilla", não tentar com subamostragem.
- Não tunar `C` por busca automatizada.
- Não mudar os nomes de pasta nem o schema das CSVs.
- Não migrar nada.

## 3. Decisão 1 — qual SVM

### 3.1 Kernel SVM está fora, com medição

`SVC(kernel='rbf')` medido neste dataset:

| n (linhas) | tempo de 1 `fit` |
|---|---|
| 5.000 | 0,3 s |
| 20.000 | 8,0 s |

São 27× de custo para 4× de dados — escala empírica ≈ **O(n^2,4)**, coerente com
o comportamento conhecido do libsvm. Extrapolando com esse expoente:

| Cenário | n | 1 `fit` |
|---|---|---|
| batch mediano | 23.000 | ~11 s |
| batch médio | 44.000 | ~53 s |
| maior batch | 260.000 | ~77 min |
| histórico completo (`dynamic` sem `train_window`) | 3.791.353 | **~23 dias** |

E `dynamic`/`abcinml` fazem 81 `fit`s por run no mensal com
`forecast_window=5` (86 batches − 5 de warm-up) e ~357 no semanal, com o
histórico crescendo a cada um. Com APR, multiplique por `apr_iterations + 1 = 21`.

Não há tuning que salve isso: é a complexidade do algoritmo, não a constante.
Kernel SVM está descartado — registrar como decisão, não reabrir.

### 3.2 As três variantes viáveis, medidas

Todas com `n = 1.000.000` linhas, com `sample_weight`, para comparar com a
baseline existente:

| Estimador | tempo | vs. LogReg | `predict_proba` |
|---|---|---|---|
| `LogisticRegression` (baseline atual) | 8,8 s | 1,0× | sim |
| `LinearSVC(dual=False)` | 14,3 s | 1,6× | **não** |
| `CalibratedClassifierCV(LinearSVC, cv=3)` | 25,5 s | 2,9× | sim |
| `SGDClassifier(loss='modified_huber')` | 7,4 s | 0,8× | sim |

`LinearSVC` escala linearmente e converge dentro de `max_iter=1000` em todos os
tamanhos testados (44k: 0,6 s; 200k: 2,1 s; 1M: 14,3 s — sem `ConvergenceWarning`
em nenhum). O custo está na mesma ordem de grandeza da regressão logística, ou
seja, **SVM linear é perfeitamente viável na grade completa**.

**Recomendação: `LinearSVC`**, pela honestidade metodológica — é a perda hinge
de fato. `SGDClassifier(loss='modified_huber')` é mais rápido e resolveria a
seção 4 de graça, mas `modified_huber` é uma hinge *suavizada e quadrática na
região de margem*, escolhida justamente para ser diferenciável e produzir
probabilidades. Chamar isso de "SVM" na dissertação exige uma nota de rodapé que
enfraquece exatamente a comparação da seção 1.

### 3.3 Consequência: `dual=False` força `squared_hinge`

Com 3,79M linhas e 58 features, `n_samples >> n_features`, então a formulação
primal (`dual=False`) é a correta — e é o que foi medido acima. O liblinear só
aceita `dual=False` com `loss='squared_hinge'`; `loss='hinge'` exige `dual=True`,
que é muito mais lento nesse regime.

Ou seja: a variante viável é **SVM linear com hinge quadrática**. Isso é
defensável (é o default do `LinearSVC` e continua sendo uma perda de margem),
mas precisa ser dito explicitamente no texto da dissertação, não deixado
implícito.

> Nota: o default de `dual` no scikit-learn 1.8 é `'auto'`, que provavelmente
> escolheria `False` aqui sozinho. Fixar explicitamente `dual=False` para que o
> comportamento não dependa de heurística de versão.

## 4. Decisão 2 — o bloqueio de `predict_proba`

### 4.1 O problema

Três pontos do pipeline chamam `predict_proba` incondicionalmente:

| Local | Uso |
|---|---|
| `main.py:147` | `y_pred_proba = model.predict_proba(X_test)` |
| `metrics_utils.compute_snapshot_metrics` | `roc_auc_score(y_test, y_pred_proba[:, 1])` — exige array de 2 colunas |
| `model_utils.compute_margins` | `predict_proba(X)[:, 1]` — a margem da Eq. 19 do APR |

Verificado no scikit-learn 1.8.0:

```
LinearSVC has predict_proba:      False
LinearSVC has decision_function:  True
SGDClassifier(loss='hinge') predict_proba: False
SGDClassifier(loss='modified_huber') predict_proba: True
```

Com `LinearSVC`, `main.py:147` levanta `AttributeError` no primeiro batch. Não é
um caso de degradação silenciosa — falha alto e cedo, o que é bom. Mas **tem que
ser resolvido antes de qualquer run**.

### 4.2 Opção A — `CalibratedClassifierCV` (nenhuma mudança no pipeline)

Envolver o `LinearSVC` em `CalibratedClassifierCV(cv=3)`, que fornece
`predict_proba` via Platt scaling. Verificado: `fit` aceita e repassa
`sample_weight` (assinatura `fit(self, X, y, sample_weight=None, **fit_params)`),
e as probabilidades saem bem distribuídas (faixa medida: 0,008–0,933).

- **A favor:** zero mudança em `main.py` e `metrics_utils.py`. A entrada em
  `initialize_model` resolve tudo, exatamente como foi com RF.
- **Contra:** custo 2,9× o da LogReg; e a calibração de Platt ajusta uma
  sigmoide sobre os scores — as *probabilidades* passam a ser logísticas, ainda
  que a *fronteira de decisão* continue sendo a do SVM. Como AUC e as métricas
  de fairness dependem ou do ranking (invariante à sigmoide monótona) ou de
  `predict` (inalterado), o impacto nas métricas é nulo. A objeção é de
  interpretação, não de número.
- **Efeito colateral sério com APR:** o APR faz `T + 1 = 21` `fit`s por batch
  (um inicial + `T=20` re-treinos); com `cv=3`, são **63 `fit`s de `LinearSVC`
  por batch**.

### 4.3 Opção B — abstrair o score no `model_utils`

Introduzir um helper único:

```python
def positive_scores(model, X):
    """Score da classe positiva: predict_proba quando existe, decision_function
    caso contrário. AUC e margens dependem só da ordenação, que é preservada."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
```

e passar a trabalhar com um vetor de scores (n,) em vez do array (n, 2) em
`main.py:147`, `compute_snapshot_metrics` e `compute_margins`.

- **A favor:** mantém o `LinearSVC` puro (1,6× LogReg, sem os 63 fits do APR).
  `roc_auc_score` aceita qualquer score monotônico, então **o AUC de LR e XGB
  não muda em um único dígito** — o branch `hasattr` devolve exatamente o
  `predict_proba[:, 1]` de hoje para eles.
- **Contra:** é a primeira mudança em `metrics_utils.py` (assinatura de
  `compute_snapshot_metrics`) desde que o eixo de modelos existe, e toca o
  caminho compartilhado por LR e XGB. Exige a verificação de regressão do passo
  7 da seção 10.
- **Contra (importante):** não resolve sozinho o problema de escala da seção 5.

### 4.4 Recomendação

**Opção B**, se APR + SVM estiver no escopo — os 63 fits por batch da Opção A
tornam essa célula da grade cara sem necessidade, e a verificação de regressão é
barata e definitiva (AUC idêntico byte a byte para runs de LR já existentes).

**Opção A**, se o SVM for rodar só com `kamiran_rw`. Aí o custo de 2,9× é
aceitável, e a virtude de não tocar em código compartilhado vale mais.

Decidir **antes** da Fase 1 e registrar aqui.

## 5. APR + SVM: dois parâmetros que deixam de fazer sentido

Se a Opção B for escolhida e APR entrar no escopo, dois defaults do APR estão
calibrados para probabilidades e quebram com `decision_function`.

**`apr_decision_boundary=0.5`** — a Eq. 19 calcula `phi = |score - d|`. Para
probabilidades, a fronteira é 0,5; para `decision_function`, é **0**. Rodar SVM
sem passar `--apr_decision_boundary 0` mede a distância até um ponto arbitrário
do espaço de scores, e todas as margens ficam erradas.

**`apr_eta=1.0`** — a Eq. 18 usa `exp(-eta * phi)`, que é sensível à escala de
`phi`. Medido com `LinearSVC` neste dataset, `decision_function` varia em
**[-2,61; 1,70]**, contra `[0; 0,5]` de uma margem de probabilidade. Com
`eta=1.0`:

| | `phi` máximo | `exp(-eta·phi)` |
|---|---|---|
| probabilidade | 0,50 | 0,61 |
| `decision_function` | 2,61 | 0,074 |

O mesmo `eta` produz uma priorização ~8× mais agressiva no SVM. Os resultados de
APR não seriam comparáveis entre modelos, e a diferença apareceria como "efeito
do modelo" sem ser.

**Saída:** normalizar `phi` dentro de `compute_margins` (dividir pelo desvio
padrão dos scores do batch, por exemplo) ou varrer `--apr_eta` para SVM em vez de
herdar `1.0`. A primeira é preferível — mantém um único `eta` comparável entre
modelos — mas **muda o comportamento de APR para LR e XGB** se aplicada sem
guarda. Se for por esse caminho, aplicar só quando o modelo não tem
`predict_proba`, e documentar a assimetria.

Se APR + SVM ficar fora do escopo inicial, esta seção inteira fica adiada — é
mais um argumento para começar só com `kamiran_rw` (seção 10.4).

## 6. Escala de features — já resolvida (verificado)

O aviso padrão sobre SVM ("normalize as features ou a regularização fica
dominada pela de maior escala") **não se aplica aqui**, e isso foi confirmado no
dado em cache, não assumido:

```
rows 3791353  cols 60
int8         59
period[M]     1
```

Depois de `data_utils._convert_types` + `_downcast_dtypes`, todas as features
são 0/1 (`int8`): flags binárias, dummies do one-hot e os próprios
`EVOLUCAO`/`IDADE_BIN`. A única coluna contínua, `NU_IDADE_N`, já é min-max para
[0, 1] na linha 112 — e, para `sensitive_attribute='IDADE_BIN'`, é descartada
por completo (daí as 58 features do benchmark, não 59).

Ou seja: **nenhum `StandardScaler` é necessário**, e nenhum `Pipeline` precisa
ser introduzido. Todas as features já vivem em [0, 1]. Vale um comentário na
entrada do `models_mapping` registrando *por que* não há scaler, para que ninguém
o adicione depois "por precaução" — isso alteraria silenciosamente todos os
resultados.

## 7. Hiperparâmetros propostos

Assumindo Opção B da seção 4:

```python
'svm': LinearSVC(
    C=1.0,
    dual=False,          # n_samples >> n_features; força loss='squared_hinge' (§3.3)
    max_iter=MAX_ITER,   # 1000, vindo de main._MAX_ITER
    random_state=42,
)
```

- **`max_iter=MAX_ITER`** — pela primeira vez desde que o eixo de modelos
  existe, o parâmetro `MAX_ITER` de `initialize_model` volta a ser usado por um
  segundo modelo (hoje só `LogisticRegression` o usa; `xgboost` o ignora, e RF
  também ignoraria). Ligar os dois em vez de fixar um literal.
- **`random_state=42`** — mesma razão do plano de RF: reprodutibilidade dos
  números da dissertação.
- **Sem `class_weight`** — mesma razão do plano de RF, e aqui ainda mais forte:
  a taxa de positivos medida é 0,2323, e `class_weight='balanced'` seria uma
  segunda correção sobreposta à reponderação (Kamiran/APR), tornando impossível
  isolar o efeito do `alpha` do ABCInML.

### 7.1 Armadilha: `ConvergenceWarning` está suprimido

`main.py:202` faz `warnings.filterwarnings('ignore', category=ConvergenceWarning)`.
`LinearSVC` emite exatamente essa categoria quando estoura `max_iter` — e o
resultado de um SVM não convergido é um modelo silenciosamente ruim, não um erro.

Nos benchmarks desta análise **não houve** `ConvergenceWarning` em nenhum tamanho
(44k, 200k, 1M), então o risco é baixo com `dual=False`. Ainda assim, o passo 6
da seção 10 verifica convergência explicitamente com o filtro desativado, antes
de confiar na grade inteira.

## 8. Cache de pasta vs. hiperparâmetros

Vale aqui **exatamente** o que a §5 de `docs/random_forest_plan.md` descreve, e
não se repete em detalhe: o nome da pasta é a chave de cache
(`main.py:250` e `main.py:323`) e **não codifica hiperparâmetros**. Trocar `C` ou
trocar a variante de SVM depois de gerar resultados faz o `main.py` carregar as
CSVs antigas e reportar sucesso.

Para SVM isso é mais perigoso que para RF, porque `C` é o parâmetro que se quer
mexer. Duas saídas, as mesmas de lá:

- **(a)** congelar `C=1.0` e apagar as pastas manualmente se mudar;
- **(b)** registrar variantes como `model_name` distintos (`svm`, `svm_c0.1`, …),
  o que faz o cache discriminar sozinho e ainda coloca as variantes lado a lado
  nos gráficos.

**Para SVM a recomendação é (b)**, ao contrário de RF — porque uma varredura de
`C` é muito mais provável de acontecer do que uma varredura de profundidade de
floresta.

## 9. `run_parallel.py`

### 9.1 Mesmo bug latente

As linhas 259-260 particionam os jobs por igualdade exata de string, então um job
`'svm'` não entra em nenhum pool, não roda, e o script imprime
`Finished. N/N succeeded.`. A correção é a mesma proposta na §6.1 do plano de RF
(mapa modelo→pool com fallback + `assert` de soma dos pools) — se aquele plano
for implementado primeiro, este ponto já vem resolvido.

### 9.2 Paralelismo: SVM é o caso fácil

Contraste com RF, e a favor do SVM:

| | `n_jobs` | workers recomendados |
|---|---|---|
| `random_forest` | `-1` (usa todos os núcleos) | 1 — mais que isso é oversubscription |
| `LinearSVC` | **não tem** — single-threaded | vários, limitado por RAM |

`LinearSVC` não paraleliza internamente, então cada job usa um núcleo. Pode-se
rodar tantos quanto houver núcleos e memória. O pool de SVM pode simplesmente
reutilizar o pool de CPU do `logistic_regression`, com `--max_lr_workers`
cobrindo os dois — é a opção mais simples e provavelmente a certa.

RAM por job: 3,79M × 58 features em `float64` (o liblinear converte a entrada)
≈ **1,8 GB** por job no pior caso (`dynamic` sem `train_window`). É isso que
limita o número de workers, não a CPU.

## 10. Plano de execução

### 10.1 Decisões a registrar antes de escrever código

1. Variante: `LinearSVC` (§3.2) — confirmar.
2. Opção A ou B para `predict_proba` (§4.4).
3. APR + SVM entra no escopo inicial? Se sim, §5 vira trabalho obrigatório.
4. Estratégia de cache (a) ou (b) (§8).

### 10.2 Fase 1 — código

5. `model_utils.initialize_model`: import + entrada conforme §7.
   - Se **Opção B**: acrescentar `positive_scores` em `model_utils.py`; ajustar
     `main.py:147`, `compute_snapshot_metrics` e `compute_margins` para o vetor
     de scores.
   - Se **Opção A**: envolver em `CalibratedClassifierCV(cv=3)`; nenhum outro
     arquivo muda.
6. `choices` em `general_utils.py:18` e `vanilla_shift_chart.py:164`.
7. `uv run ruff check src/` sem erros **novos** (há 14 pré-existentes) e
   `uv run ruff format` só nos arquivos tocados — `ruff format src/` hoje
   reformataria 14 arquivos não relacionados, incluindo `metrics_utils.py` e
   `model_utils.py`, e poluiria o diff de regressão do passo 8.

### 10.3 Fase 2 — validação

8. **Regressão (só se Opção B):** re-rodar um cenário de `logistic_regression`
   já existente e conferir que `assessment_all_baselines.csv` sai **idêntica** à
   versionada. É a verificação que fecha o risco de tocar em código
   compartilhado. Fazer antes de qualquer run de SVM.

   Cuidado com o cache de pasta ao fazer isso: no repositório principal,
   `vanilla`/`static`/`dynamic` **não re-executam** (as pastas existem e são
   carregadas), e o `abcinml` re-executa **sobrescrevendo a CSV versionada**.
   Forma segura: rodar num `git worktree` separado (copiando para ele
   `src/data/SRAG/raw/` e `src/data/SRAG/cache/`, que não são versionados),
   apagar lá as pastas do cenário escolhido (a cópia é descartável) e comparar
   as CSVs geradas com as do repositório principal. Alternativa mínima: rodar só `--baseline_type
   abcinml --alpha 0.5` no repositório principal e confirmar com `git status`
   que a CSV não mudou — cobre `abcinml`, mas não as outras três baselines.
9. `--test_run True --alpha 0.5 --baseline_type all --model_name svm`, com o
   filtro de `ConvergenceWarning` temporariamente desativado (§7.1), conferindo
   que nenhum batch estoura `max_iter`.
10. Conferir pastas e coluna `model_name == 'svm'` nas três CSVs; rodar
    `concatenate_results.py` e os três `migrate_*.py --dry-run` (as regexes
    aceitam `svm` — sem underscore, é um caso mais simples que
    `logistic_regression`).
11. Só então, se §5 estiver resolvida: `--fairness_correction adaptative_rw
    --apr_iterations 3 --apr_decision_boundary 0`.

> Nota: `--test_run` limita o dataset às primeiras 100.000 linhas
> (`main.py:231`, e assim documentado no README). Como o DataFrame está ordenado
> por data, isso cobre só os primeiros meses — suficiente para validar o
> pipeline, não para medir convergência nos batches grandes. O passo 9 é um
> teste de fumaça; convergência no histórico completo só se confirma na grade.

### 10.4 Fase 3 — grade

12. `'svm'` em `MODELS`; pool conforme §9.2.
13. Primeiro sweep restrito a `--fairness_corrections kamiran_rw`. Com os tempos
    da §3.2, o SVM linear cabe na grade completa sem `--train_window` — é o único
    dos três modelos novos considerados de que isso se pode dizer.

### 10.5 Fase 4 — documentação (no mesmo commit do código)

14. `README.md` — exigido pelo `CLAUDE.md` na mesma mudança (o `CLAUDE.md` em
    si não descreve mais CLI nem `model_utils`): seções **CLI Reference**
    (`--model_name`), **Parallel Grid Run** (pool e contagem de cenários) e
    **Installation & Usage**. Se Opção B, documentar também `positive_scores`,
    a nova assinatura de `compute_snapshot_metrics` e, na seção **Adaptive
    Priority Reweighing**, o `--apr_decision_boundary 0` para SVM (§5).
15. Na dissertação: registrar que a variante é **hinge quadrática, primal**
    (§3.3) e que kernel foi descartado por inviabilidade medida (§3.1). Ambos
    são escolhas metodológicas que um avaliador vai perguntar.

## 11. Critérios de aceitação

- `--model_name svm` roda as 4 baselines ponta a ponta, gravando as três CSVs e
  os PNGs nas pastas de sempre.
- **Se Opção B:** um cenário de `logistic_regression` re-rodado produz CSV
  idêntica à versionada (nenhum resultado existente perturbado).
- Nenhum batch da grade final emite `ConvergenceWarning` com o filtro desativado.
- Dois runs idênticos de SVM produzem métricas idênticas (semente fixa).
- `concatenate_results.py` agrega as pastas de SVM com `model_name` correto.
- `run_parallel.py` submete os jobs de SVM, com a contagem impressa batendo com
  a grade.
- Se APR + SVM foi ao escopo: `--apr_decision_boundary` e a escala de `phi`
  tratadas conforme §5, e a escolha documentada.
