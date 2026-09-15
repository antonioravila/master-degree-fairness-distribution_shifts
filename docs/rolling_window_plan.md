# Plano: janela deslizante (`--train_window`) para `dynamic`/`abcinml`

Documento de design — descreve **como** implementar uma janela de treino limitada,
antes de qualquer mudança de código. Não altera o comportamento padrão do
pipeline: sem passar o novo parâmetro, `dynamic`/`abcinml` continuam treinando
sobre todo o histórico acumulado, exatamente como hoje.

## 1. Motivação (recap)

Medido em `feat/adaptative_priority_rw` (perfil real, `IDADE_BIN`/monthly,
LogisticRegression): o tempo de `fit` cresce mais que linear com o tamanho do
treino acumulado — 44K linhas → 0,3 s; 369K linhas → ~5 s. `dynamic` e
`abcinml` re-treinam do zero a cada batch sobre **todo** o histórico até ali,
e o APR multiplica esse custo por `apr_iterations` (=20) refits por batch. Uma
janela deslizante limita o tamanho do treino a uma constante (≈ W meses de
dados), em vez de deixá-lo crescer até o dataset inteiro.

## 2. Escopo

**Dentro do escopo:**
- Novo parâmetro `--train_window` (em meses, mesma convenção do
  `--forecast_window` existente), aplicado só a `dynamic` e `abcinml`.
- `vanilla`/`static` não são afetados (já treinam uma vez só, num único
  batch).
- Registro do valor usado nos CSVs/nomes de pasta, para nunca misturar runs
  com e sem janela na mesma agregação.

**Fora do escopo (não fazer nesta mudança):**
- Não redefinir o significado de `dynamic`/`abcinml` quando `train_window`
  não é passado — o default é `None` (sem limite), idêntico ao
  comportamento atual.
- Não mudar a janela usada pelo **forecast** do `abcinml` (`forecast_window`
  já existe e continua controlando só a previsão SMA/EMA da distribuição
  futura — ver seção 4).
- Não mudar `vanilla`/`static`, nem a lógica de warm-up.

## 3. Interface (CLI e propagação)

Novo argumento em `general_utils.parse_arguments`, ao lado de
`--forecast_window`:

```python
parser.add_argument('--train_window', type=int, default=None)
```

- Unidade: **meses**, igual a `--forecast_window`. Convertido para número de
  batches com a mesma função já existente:
  `general_utils.effective_forecast_window(train_window, batch_type)` — sem
  duplicar a lógica de conversão monthly/weekly.
- `default=None` → comportamento atual (histórico completo), zero mudança
  para quem não passar a flag.
- Propagado por `main()` → `run_baseline(...)` → junto dos demais parâmetros
  já passados hoje (`forecast_window_batches`, `apr_iterations`, etc.).

## 4. Onde a janela corta, no `run_baseline`

Hoje (`src/main.py`), `dynamic`/`abcinml` treinam em `accumulated_train_df`,
que cresce por `pd.concat` a cada batch e nunca é truncado. Como `df` já vem
ordenado por `window_column` (`data_utils.pre_process_df` faz
`sort_values(...)`), o corte da janela é feito por slicing sobre os `batches`
já calculados no início de `run_baseline`, sem reprocessar o DataFrame:

```python
if train_window_batches is not None:
    lower = max(0, i - train_window_batches)
    train_batches = batches[lower:i]
else:
    train_batches = batches[0:i]  # comportamento atual
```

Dois pontos de atenção na integração com o que já existe:

- **Warm-up (`forecast_window_batches`)**: o limite superior do warm-up
  continua igual (`i < forecast_window_batches` → não treina/testa). O
  `train_window_batches` só entra no limite **inferior** de quais batches
  compõem o treino — os dois cortes convivem sem conflito.
- **`accumulated_train_df` deixa de ser "tudo desde o batch 0"**. Em vez de
  só fazer `pd.concat` e nunca descartar, a manutenção da janela precisa
  também **remover** as linhas do batch que saiu da janela (`lower` avançou).
  Isso é uma mudança estrutural na função `_accumulate()` interna de
  `run_baseline` — hoje ela só acrescenta; com janela, teria que acrescentar
  o batch novo e descartar o(s) batch(es) mais antigos que ficaram fora de
  `train_batches`. A forma mais simples e menos arriscada: **não manter
  `accumulated_train_df` incremental quando `train_window` está ativo** —
  recomputar a fatia via slicing direto sobre o `df` original a cada batch
  (`df[df[window_column].isin(train_batches)]`), já que a fatia é pequena e
  constante em tamanho (diferente do histórico completo, que era grande
  demais para reprocessar do zero a cada batch — daí o acumulador existir em
  primeiro lugar). Ou seja: com janela ativa, o próprio motivo de existir do
  acumulador incremental desaparece.

## 5. Reponderação (Kamiran e APR) recalculada só na janela

`reweigh_df`/`weight_estimation` (Kamiran) e o `p_{y,a}` fixo do
`adaptive_priority_reweighing` (APR) são hoje computados sobre
`X_train`/`y_train` — ou seja, já operam sobre o que quer que seja passado
como treino. **Nenhuma mudança de código é necessária nessas funções**: como
a janela corta o `X_train`/`y_train` antes de chegar em `fit_fair_model`, a
reponderação automaticamente passa a refletir só a janela, sem viés de
histórico fora dela. Only a validação empírica muda (as proporções passam a
oscilar mais entre batches, já que a base é menor — ver seção 8).

## 6. `forecast_window` do `abcinml` continua desacoplado

O `historical_proportions` (usado para SMA/EMA da distribuição futura) **não
é limitado pela nova janela** — continua olhando para trás pelo
`forecast_window_batches` já existente (que pode ser igual, maior ou menor
que `train_window_batches`; são dois hiperparâmetros independentes). Isso é
importante porque estimar a distribuição futura é uma agregação barata
(médias de proporções), diferente do custo de treinar o modelo — não há
motivo para forçar os dois no mesmo valor, e fazer isso perderia a
possibilidade de comparar "forecast de longo prazo + treino de curto prazo".

## 7. Rastreabilidade nos resultados (evitar misturar runs)

Seguindo o mesmo padrão já usado para `fairness_correction`/
`apr_fairness_measure`:

- **Nome de pasta**: `metrics_utils._run_folder_name` ganha um sufixo
  condicional, só quando `train_window` não é `None` — ex.:
  `..._monthly_kamiran_rw_trainwindow12`. Sem a flag, nome de pasta
  idêntico ao atual (nenhuma pasta existente é afetada).
- **Coluna nos CSVs**: `main._attach_metadata` grava `train_window` (o valor
  em meses, ou vazio/`None` quando não usado) nos três CSVs
  (`assessment_all_baselines.csv`, `temporal_results_df.csv`,
  `absolute_results_df.csv`), mesmo padrão do `fairness_correction`.
- **`concatenate_results.py`**: os regex de reconhecimento de pasta
  precisariam do mesmo tratamento que já existe para o sufixo de
  `fairness_correction` — um grupo opcional a mais no fim do nome.
- **Notebooks**: qualquer agregação (`groupby`, filtro) que hoje assume
  implicitamente "só existe uma configuração de treino" passa a precisar
  filtrar também por `train_window`, do mesmo jeito que hoje já filtra por
  `fairness_correction` — mesma lição da última mudança no notebook do
  seminário.

### 7.1. Conflito real: cache compartilhado de `vanilla+static+dynamic`

Em `main.py` (bloco `non_abc_to_run`), `vanilla`, `static` e `dynamic` são
executados juntos e seus resultados combinados são salvos em **um único**
arquivo, via `metrics_utils.baseline_non_abc_folder_name(model_name,
batch_type, fairness_correction)` — chaveado só por
`(model_name, batch_type, fairness_correction)`. `train_window` só afeta
`dynamic`; `vanilla`/`static` treinam uma vez, num único batch, e o ignoram.
Isso quebra a premissa implícita de que essa pasta representa uma única
configuração de treino:

- Se o nome da pasta ganhar o sufixo de `train_window`, `vanilla`/`static`
  seriam recalculados e duplicados a cada valor de W testado — resultado
  idêntico, armazenado várias vezes (desperdício, não incorreção).
- Se o nome **não** ganhar o sufixo, uma run com `train_window=12` sobrescreve
  o cache de uma run com `train_window=None` para a mesma combinação
  model/batch_type/fc — `vanilla`/`static` continuam corretos, mas `dynamic`
  dentro do mesmo CSV passaria a refletir só o último valor de W rodado,
  poluindo qualquer comparação "sem janela vs. com janela" feita depois.

**Direção recomendada**: separar o armazenamento de `dynamic` do de
`vanilla`/`static` assim que `train_window` for implementado — `dynamic` passa
a ter seu próprio arquivo/pasta (sufixado por `train_window` quando ativo),
enquanto `vanilla`/`static` continuam compartilhando a pasta atual, sem
sufixo, já que nunca são afetados por essa janela. Isso evita tanto a
duplicação quanto a sobrescrita silenciosa. Essa separação deve ser decidida
e implementada **antes** de qualquer rodada de validação da seção 9 — sem
ela, os números de `dynamic` coletados ali não são confiáveis.

## 8. Como escolher o valor de W (procedimento, não só o valor)

1. **Âncora inicial pelos períodos epidemiológicos** já usados no projeto
   (pré-pandemia, pandemia aguda 1/2, transição, endêmico). Não ultrapassar
   a duração do regime mais curto (pandemia aguda 2 ≈ 12 meses) como teto de
   partida — acima disso, a janela já reintroduz dado de regime passado, o
   problema que o `abcinml` existe para mitigar.
2. **Checar densidade de casos positivos no subgrupo sensível dentro da
   janela candidata**, antes de rodar qualquer coisa — para cada W candidato,
   contar quantos óbitos existem no subgrupo sensível nos batches que
   comporiam a janela no pior caso (o batch mais antigo/mais esparso da
   série). Um TPR/FPR calculado sobre poucas dezenas de casos é ruído, não
   sinal, independente de quão bem o modelo ajustou.
3. **Curva de aprendizado empírica** — usando dados que já existem (sem
   rodar nada novo): para uma fatia de batches, plotar accuracy e as 3
   métricas de fairness em função do tamanho do treino acumulado (que já
   temos registrado por batch nos resultados atuais) e observar onde a curva
   satura. Onde a métrica para de melhorar é o indício mais forte de que
   história adicional além daquele ponto é custo sem benefício.
4. **Grid search formal**, reaproveitando a infraestrutura existente —
   adicionar `train_window` como mais um eixo em `run_parallel.py`
   (`TRAIN_WINDOWS = [3, 6, 12, 24, None]`, análogo a `WINDOWS` do
   `forecast_window`), escolher o valor final pela mesma régua já usada para
   `alpha`/`forecast_window` (accuracy × DP × TPR diff × FPR diff), em um
   corte temporal de validação separado da avaliação final.
5. **Teto por orçamento de tempo**, se a motivação principal continuar sendo
   velocidade: escolher o maior W (em meses) cujo pior caso de linhas
   acumuladas fique abaixo do teto de tempo de fit aceitável — usando a
   mesma curva de "linhas → tempo de fit" já medida neste projeto.

## 9. Plano de validação antes de adotar

Como a janela **muda os números** (é uma alteração real de dados de treino,
não uma otimização transparente como o downcast/cache), a adoção não deve
ser silenciosa:

1. Rodar `dynamic`/`abcinml` com `train_window=None` (baseline atual) e com
   2–3 valores candidatos de W, no mesmo `sensitive_attribute`/`model_name`.
2. Comparar accuracy e as 3 métricas de fairness lado a lado (mesmo padrão
   dos gráficos "Kamiran vs APR" já feitos no notebook do seminário — aqui
   seria "sem janela vs. com janela W").
3. Só promover um valor de W para os resultados "oficiais" da dissertação
   depois dessa comparação — documentando explicitamente que é uma escolha
   de design, com a justificativa dos passos 1–4 da seção 8.

## 10. Impacto em `run_parallel.py`

`run_parallel.py` mantém um mirror manual de `_run_folder_name`/
`_fairness_correction_suffix` (comentado no próprio código como "keep in
sync") — o mesmo tratamento dado a `fairness_correction` precisa ser
replicado para `train_window`:

- **`_run_folder_name`** (e, por consequência, `_folder_exists`/
  `_missing_alphas`) precisa reconhecer o novo sufixo. Se ficar
  dessincronizado de `metrics_utils._run_folder_name`, o script passa a
  comparar contra o nome de pasta errado — ou pula scenarios com
  `train_window` que ainda não rodaram (achando que já existem), ou tenta
  re-rodar scenarios que já estão completos.
- **Novo eixo no grid** (`TRAIN_WINDOWS`, análogo a `WINDOWS`): não deve ser
  simplesmente adicionado ao produto cartesiano de `build_commands` sem
  ressalva. Esse eixo só é relevante para `dynamic`/`abcinml`, mas
  `build_commands` gera comandos por `(model, technique, window, batch_type,
  fairness_correction)` sem saber quais baselines vão rodar dentro de cada
  um (`main.py` roda `--baseline_type all` por padrão) — multiplicar por
  `TRAIN_WINDOWS` recalcularia `vanilla`/`static` à toa para cada W, o mesmo
  problema da seção 7.1. Multiplicar o grid inteiro por mais um eixo de
  tamanho ~5 (incluindo `None`) também infla o número de jobs/logs em ~5x
  (hoje ~40 scenarios → ~200).
- **Recomendação**: não embutir `TRAIN_WINDOWS` no grid principal de
  `build_commands`. Rodar a exploração de W (2–3 valores candidatos, seção 9)
  como um grid separado e menor — e só depois de resolver a separação de
  armazenamento da seção 7.1, senão os resultados de `dynamic` coletados por
  `run_parallel.py` sofrem o mesmo problema de sobrescrita/duplicação.
- `_scenario_cmd`/`run_command`/`apr_gate` não precisam de mudança de lógica
  além de propagar `--train_window` quando aplicável — o gate de
  concorrência do APR é ortogonal a essa janela.

## 11. Achados colaterais (mesma classe de risco, não bloqueantes)

Durante o levantamento de dependências, dois scripts de análise já hoje têm
uma lacuna análoga à de `train_window`, mas causada por `fairness_correction`
— não são bloqueantes para implementar a janela, mas valem registro porque a
mesma correção (filtrar/parametrizar explicitamente) resolve os dois:

- **`alpha_analysis.py`**: `groupby('alpha')` não filtra por
  `fairness_correction` — se o CSV agregado tiver kamiran e APR juntos, os
  números já se misturam silenciosamente hoje. Ganhar mais uma coluna
  (`train_window`) sem tratamento equivalente reproduziria o mesmo problema
  para essa nova dimensão.
- **`vanilla_shift_chart.py`**: chama `baseline_non_abc_folder_name(model_name,
  batch_type)` sem passar `fairness_correction`, sempre resolvendo para a
  pasta `kamiran_rw` por default e ignorando APR silenciosamente. Precisaria
  do mesmo tratamento explícito para não ficar preso ao `train_window`
  default (`None`) por omissão.

## 12. Riscos e perguntas em aberto

- **Comparabilidade com resultados já publicados** (ex.: seminário de
  andamento): qualquer resultado gerado com `train_window` ativo precisa ser
  claramente rotulado como tal — não deve ser comparado diretamente contra
  os resultados de histórico completo já apresentados sem essa ressalva.
- **`abcinml` com janela pequena**: se `train_window` for menor que
  `forecast_window`, o modelo treina em menos dados do que os usados para
  prever o futuro — não é necessariamente errado, mas é uma combinação que
  vale nomear e testar explicitamente (treino "míope" + forecast de longo
  prazo).
- **Interação com o APR**: como o APR já é o baseline mais caro, ele é o
  maior beneficiário da janela — mas também o que mais precisa de validação
  cuidadosa (Eq. 17/18 do método dependem de contagens de subgrupo; com
  janela pequena essas contagens ficam mais instáveis batch a batch).
