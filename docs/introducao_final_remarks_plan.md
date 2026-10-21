# Plano: escrita da Introdução (Seção 1) e dos Final Remarks (Seção 5)

Documento de design e de redação (em português; o artigo é em inglês).
Cobre `1 Introduction` (o texto que antecede `1.1 Theoretical Background`),
`5 Final Remarks`, `5.1 Limitations` e `5.2 Future Works` do PDF
"Beyond Logistic Regression".

## 0. Princípio de projeto: o teste do leitor que só lê dois trechos

Quem ler **apenas** a Introdução e os Final Remarks deve sair sabendo:

| # | O leitor deve saber | Onde aparece |
|---|---|---|
| 1 | Qual é o problema (correção de fairness sob distribution shift) | Intro P1 |
| 2 | Qual é a lacuna do ABCInML (só LR, só SMA, cenários restritos) | Intro P2 |
| 3 | Pergunta de pesquisa e objetivos | Intro P3 |
| 4 | O que foi feito: dados, modelos, baselines, previsão, métricas | Intro P4 |
| 5 | Contribuições, com números | Intro P5; retomadas no FR P2–P3 |
| 6 | Resposta à pergunta de pesquisa, objetivo a objetivo | FR P2 |
| 7 | Por que importa (implicações) | FR P3 |
| 8 | O que **não** vale (limitações) e o que vem depois | FR 5.1 e 5.2 |

Regras:
- Os **mesmos números** nos dois trechos e no corpo (tirar todos da
  Tab. 6/7 do PDF e da Tab. 8, ver seção 4 deste plano).
- Nada de número de figura/tabela no argumento da Introdução; os Final
  Remarks podem citar "Table 6" porque o leitor já terá a tabela à mão.
- Terminologia fixa: *Vanilla, Static, Dynamic, ABCInML*; *DP difference*,
  *TPR difference*, *FPR difference*; *forecast window*; *shift*.
- A Introdução anuncia as respostas em uma frase por objetivo; os Final
  Remarks as repetem com a **interpretação** e as ressalvas.

## 1. Introdução (≈ 1–1,5 página, antes de 1.1)

O Section 2 já repete a pergunta de pesquisa e os cinco objetivos; na
Introdução a pergunta aparece **em uma frase** e os objetivos em uma
linha cada (ou agrupados), sem duplicar o Section 2.

| § | Função | Conteúdo | Frase-guia (EN) |
|---|---|---|---|
| P1 | Problema | Decisões automatizadas de alto impacto (saúde, crédito, justiça); métodos de fairness assumem treino e produção com a mesma distribuição; em produção as populações mudam | *Fairness-aware models are typically trained and evaluated as if the population they serve were static, yet in practice the data they see drifts.* |
| P2 | Lacuna | ABCInML antecipa a distribuição futura de (S,Y) e a incorpora nos pesos; mas foi avaliado só com Logistic Regression, só com média móvel simples e em cenários restritos | *ABCInML was evaluated only on a linear model, with a single simple forecaster, under a few settings; whether anticipation generalises is open.* |
| P3 | Pergunta e objetivos | Pergunta de pesquisa em 1 frase; objetivos (1)–(5) compactados | *We ask to what extent anticipatory bias correction extends to non-linear models and whether its forecasting component is the limiting factor.* |
| P4 | Metodologia (resumo) | SRAG 2019–2026 (≈ 4,4 milhões de registros, soma dos `n` da Fig. 2); alvo = óbito; atributo sensível = idade ≥ 65 (binário); LR, XGBoost, RF; 4 baselines; SMA, EMA e oracle; lotes mensais e semanais; 11 valores de α; métricas snapshot (DP, TPR, FPR, acurácia, AUC) e temporais (MB, TS, MBD); `forecast_window = 5` | *We reproduce the four ABCInML settings on three classifiers over a decade-long, naturally shifting health dataset.* |
| P5 | Contribuições | bullets C1–C4 com os números reais (seção 3 deste plano) | *Our contributions are…* |
| P6 | Resultado em uma frase | Antecipar reduz o DP além do retreino em todos os modelos, ao custo do TPR; melhorar a previsão não resolve | *Anticipation lowers demographic-parity difference below retraining alone in all three models, but raises the TPR gap; a perfect forecast does not change this picture.* |
| P7 | Roteiro | Uma frase: Seção 2 objetivos; 3 método; 4 resultados (4.1 dados, 4.2 reprodução LR, 4.3 modelos não lineares, 4.4 previsão); 5 conclusões | — |

Cuidados:
- Evitar "provamos"/"mostramos que generaliza": usar *consistent with*,
  *in this dataset*.
- O texto atual de 1.1/1.2 já tem a teoria; a Introdução **não** redefine
  DP, TPR nem reweighing.
- Declarar na P4 que só o `forecast_window = 5` é reportado (a Tab. 1 lista
  3, 5 e 10).

## 2. Final Remarks (≈ 1 página) + 5.1 + 5.2

### 2.1 Estrutura de "considerações finais"

| § | Função | Conteúdo |
|---|---|---|
| P1 | Retomada | Problema, lacuna e o que foi feito, em 3–4 frases (sem repetir a Introdução inteira) |
| P2 | Respostas aos objetivos | Um bloco por objetivo, na ordem (1)–(5), cada um com **um número** e **uma ressalva** (tabela abaixo) |
| P3 | Valor da contribuição | Implicações para quem implanta um modelo (guia de decisão) e para a pesquisa (o que o oracle revela) |

### 2.2 Respostas aos objetivos (números do PDF atual; `forecast_window = 5`)

| Obj. | Resposta curta | Evidência | Ressalva obrigatória |
|---|---|---|---|
| (1) Reproduzir em LR | DP melhora gradualmente de Vanilla a ABCInML; FPR ≈ 0 em Dynamic e ABCInML; TPR é melhor no Static | Figs. 3–4, Tabs. 3–5 | A antecipação (ABCInML) soma pouco ao retreino em FPR e temporais |
| (2)–(3) Estender e comparar | Redução do DP vs. Vanilla: LR 81,8%, XGBoost 81,6%, RF 74,1%; ABCInML < Dynamic no DP (−0,75; −0,81; −0,39 p.p.); acurácia ≈ igual (|Δ| ≤ 0,17 p.p.) | Tab. 6 | TPR sobe (+2,99; +3,84; +2,88 p.p.); o menor DP do RF vem de um Vanilla já mais justo |
| (3) Estabilidade temporal | ABCInML e Dynamic são os melhores e próximos; RF mais estável | Tab. 7 | XGBoost weekly: ABCInML pior que Dynamic em MB (14,40% vs. 10,98%) e MBD |
| (4) Sensibilidade a α | DP piora com α; TPR melhora com α; FPR tem ótimo interior | Fig. 5, Tab. 4 | Não existe α que otimize todas as métricas (trade-off entre métricas) |
| (5) Previsão limita? | EMA reduz o MAE do SMA em todos os períodos; o oracle não melhora a fairness de forma consistente (pior no *acute pandemic 1* no DP) | Tab. 8, Fig. 7, análise por Δ | O MAE do EMA na Tab. 8 usa α_ema = 0,3; as rodadas de fairness usam 0,5 (alinhar) |

### 2.3 Mensagem central (a "tese" de uma frase)

> *Anticipating the future (S,Y) distribution buys a measurable reduction in
> demographic-parity difference beyond retraining, and this holds for
> non-linear models; but the gain is bounded by the reweighting mechanism
> itself, not by forecast accuracy, and it trades off against TPR parity.*

Interpretação a oferecer (marcada como hipótese): o ABCInML antecipa apenas
P(S,Y), enquanto o shift do SRAG também altera P(X) e P(Y|X) (Seção 4.1);
por isso prever melhor P(S,Y) não corrige o que o reweighting não alcança.

### 2.4 5.1 Limitations (agrupar por natureza)

| Grupo | Limitação | Observação |
|---|---|---|
| Validade externa | Um único conjunto (SRAG/Brasil), um atributo sensível binário (idade ≥ 65), um limiar | O shift é de natureza epidemiológica, extremo e único |
| Método | Só pré-processamento (reweighing de Kamiran & Calders); os objetivos mencionam "robust bias mitigation approaches", mas só um foi avaliado | O repositório tem a variante `adaptative_rw` (APR) não analisada no artigo |
| Previsão | Só SMA, EMA e oracle; o oracle antecipa apenas P(S,Y), não P(X\|Y) | O oracle é incompleto para o RF (não foi rodado nos 11 α, monthly e weekly), por isso a 4.4 usa só LR e XGBoost |
| Configuração | Janela de previsão fixa em 5; α_ema fixo (0,3 no MAE e 0,5 nas rodadas de fairness; alinhar); **hiperparâmetros fixos, sem busca/ajuste**: LR (`max_iter=1000`), XGBoost (padrões da biblioteca, `tree_method='hist'`, GPU) e RF (`n_estimators=100`, `max_depth=12`, `random_state=42`) | Os valores do RF foram escolhidos à mão para evitar um caso degenerado (com `min_samples_leaf=50` o Vanilla/Static nunca previa positivo), não por desempenho; sem `class_weight`. Janelas 3 e 10 existem nos resultados, mas não são reportadas |
| Estatística | **Uma única execução por configuração** (nenhum par duplicado nos resultados; não há repetição por semente nem intervalos); Wilcoxon pareado sobre lotes autocorrelacionados (p-valores otimistas); médias entre α mistura configurações onde a antecipação pesa pouco | LR é determinístico; RF usa semente fixa 42; XGBoost não define `random_state` (usa o padrão da biblioteca) e roda em GPU; por isso diferenças muito pequenas (≈ 0,1 p.p.) não devem ser interpretadas |
| Interpretação | Correlação, não causalidade: "reweighting é o limitante" é inferido por exclusão; o paralelo com TPR sugere, mas não demonstra, o mecanismo | Usar linguagem de hipótese |

### 2.5 5.2 Future Works (um item por limitação)

| Item | Resolve | Esboço |
|---|---|---|
| Outros mecanismos de correção (in-processing, `adaptative_rw`/APR, equalized odds) | Método; TPR | Testar se o custo em TPR some quando o objetivo de fairness inclui as taxas |
| Antecipar mais que P(S,Y): P(X\|S,Y) ou o risco | Previsão; oracle | Previsão condicional; oracle sobre features |
| Previsores mais ricos (ARIMA, ETS, aprendizado) e oracle completo para RF | Previsão; limitação do RF | Completar a grade (11 α, monthly e weekly) |
| Outros conjuntos e atributos sensíveis (raça, múltiplos atributos, interseccionalidade) | Validade externa | Datasets com shift gradual vs. abrupto |
| Análise por período e inferência robusta (bootstrap por blocos, sementes) | Estatística | Intervalos para ΔABCInML−Dynamic |
| Seleção de α adaptativa (por métrica-alvo ou online) | Trade-off | α variável no tempo, guiado por *drift* detectado |

### 2.6 Fecho

Uma frase final voltada à prática (o que fazer ao implantar) e uma ao campo
(a antecipação é útil quando o objetivo é DP; para TPR/FPR precisa-se de outro
objetivo).

## 3. P5 — Contribuições (bullets para a Introdução, com os números reais)

Usar estes mesmos itens, na mesma ordem, na Introdução (P5) e nos Final
Remarks (P3). Configuração dos números: `kamiran_rw`, `train_window = none`,
`forecast_window = 5`, ABCInML = média sobre os 11 α, monthly e weekly com
pesos iguais, salvo indicação. Fonte entre colchetes.

**Frase de abertura (P5):** *This paper makes the following contributions:*

- **C1 — Primeira avaliação do ABCInML em modelos não lineares, sobre um
  conjunto real com shift natural.** XGBoost e Random Forest, além da
  Logistic Regression reproduzida como referência, sobre o SRAG
  (2019–2026): **≈ 4,44 milhões de registros** (23.721 + 2.918.344 +
  582.621 + 284.365 + 630.796 = 4.439.847; soma dos `n` da Fig. 2), em
  **5 períodos** e **83 lotes mensais / 366 semanais**. A letalidade
  passa de < 15% para até ≈ 35% e o shift envolve P(Y), P(X) e P(Y|X)
  (Seção 4.1). [Fig. 1–2; resultados `src/results/all/`]
- **C2 — Protocolo experimental mais amplo que o original.**
  **313 configurações do ABCInML** executadas (3 modelos × SMA com janelas
  3, 5 e 10 + EMA + oracle × monthly/weekly × 11 valores de α), das 330
  previstas: o oracle do RF só foi rodado em monthly e com 5 dos 11 α.
  Quatro baselines (Vanilla, Static, Dynamic, ABCInML), métricas de
  desempenho (acurácia, AUC-ROC), de fairness (DP, TPR, FPR) e temporais
  (Max. Bias, Temporal Stability, Max. Bias Diff.), mais as taxas
  absolutas de TPR e FPR. [Tab. 1; `assessment_all_baselines.csv`]
- **C3 — O ganho da antecipação sobre o retreino existe nos três modelos,
  é pequeno e custa TPR.**
  - DP vs. Vanilla: LR 21,97% → 4,01%, XGBoost 17,91% → 3,30%, RF
    9,08% → 2,36%, isto é, reduções de **81,8%, 81,6% e 74,1%**. [Tab. 6]
  - DP vs. Dynamic: ABCInML é menor em **−0,75, −0,81 e −0,39 p.p.**
    (−15,8%, −19,7% e −14,2% em termos relativos). [Tab. 6]
  - TPR vs. Dynamic: ABCInML é maior em **+2,99, +3,84 e +2,88 p.p.**
    (10,39 → 13,38%; 14,77 → 18,61%; 10,26 → 13,13%). [Tab. 6]
  - Acurácia vs. Dynamic: −0,12, −0,17 e −0,10 p.p.; AUC-ROC: −1,16,
    −0,74 e −0,74 p.p. [Tab. 6]
  - Efeito do α (0 → 1): DP sobe de 3,27 → 4,76% (LR), 2,91 → 4,11%
    (XGBoost) e 2,12 → 2,74% (RF); TPR cai de 16,49 → 10,39%, 22,07 →
    14,77% e 15,94 → 10,26%; o α ótimo do FPR é 0,5, 0,7 e 0,9. [Tab. 4b]
  - Estabilidade temporal: em 18 combinações (3 modelos × 3 métricas ×
    monthly/weekly), o ABCInML é o melhor em **9**, o Dynamic em 6, o
    Static em 3 e o Vanilla em 0. [Tab. 7]
- **C4 — Diagnóstico do componente de previsão: prever melhor não é o
  que limita.**
  - EMA vs. SMA (MAE): monthly **0,0382 → 0,0344** (−9,9%) e weekly
    **0,0212 → 0,0189** (−10,8%), menor em todos os 5 períodos e nos dois
    batch types. [Tab. 8; ver pendência 2 abaixo]
  - Oracle (MAE = 0) vs. SMA, agregando os 5 períodos: DP
    **3,86 → 3,91%** (LR) e **3,24 → 3,25%** (XGBoost); TPR 13,75 → 13,86%
    e 18,25 → 18,49%. Ou seja, zerar o erro de previsão muda o DP em
    **+0,05 e +0,01 p.p.** [Fig. 7; todos os α, α_ema = 0,5]
  - Por período (Oracle − SMA, α ≤ 0,3, LR / XGBoost): DP **+0,79 / +0,89**
    (pre-pandemic) e **+0,51 / +0,60** (acute 1), mas **−0,33 / −0,52**
    (acute 2), **−0,24 / −0,24** (transition) e **−0,23 / −0,32**
    (endemic); TPR em acute 2: **+0,76 / +1,30**. [Tab. 10d; apenas
    4 de 60 células passam em Holm: citar como consistência de sinais]
- **C5 — (reservado) Reprodutibilidade.** Código aberto: decisão
  adiada; não listar até confirmar.

**Frase de fecho (P5):** *Taken together, the results indicate that the
benefit of anticipation is bounded by the reweighing mechanism rather than
by forecast accuracy.*

## 4. Pendências de validação (ignoradas na primeira versão)

As inconsistências e decisões em aberto estão em
`docs/pendencias_validacao.md` (P01–P14). A decisão do autor é escrever
a Introdução e os Final Remarks **sem** resolvê-las e validá-las depois.

Números marcados como provisórios no arquivo de pendências, e que devem
aparecer com cautela no texto enviado ao orientador:

- MAE do EMA vs. SMA (C4, Tab. 8): P01 e P02.
- Oracle vs. SMA agregado e por período (C4, Fig. 7 e Tab. 10d): P06.
- α ótimo do FPR do RF (C3): P05.

## 5. Ordem de trabalho sugerida

1. (Adiado) Validar as pendências de `docs/pendencias_validacao.md`; a primeira versão segue com os números atuais.
2. Congelar a lista de números (Tabs. 6, 7, 8 e Δ da 4.4).
3. Escrever **Final Remarks primeiro** (força a clareza da mensagem), depois
   a Introdução, que antecipa a mesma mensagem.
4. Revisar com o teste da seção 0: ler só os dois trechos e conferir os 8
   itens da tabela.
5. Traduzir/revisar o restante (as partes em português da 4.3).

## 6. Decisões do autor (registradas em 14/10/2026) e pendências

| # | Decisão | Efeito no texto |
|---|---|---|
| 1 | **Manter** "o limitante é o reweighting" | Contribuição 4 e tese de uma frase ficam como estão; na Discussão, marcar o mecanismo (apenas P(S,Y) é antecipado) como interpretação |
| 2 | **Código aberto: sim, a decidir depois** | Deixar um marcador no FR (5.2 ou Reproducibility) e não listar como contribuição por enquanto; decidir link/licença antes da submissão |
| 3 | **Hiperparâmetros/sementes: analisado no código** | Ver a linha "Configuração" e "Estatística" de 2.4: sem ajuste de hiperparâmetros e uma única execução por configuração; declarar em 3.0.3 (Materials & Methods) e em 5.1 |
| 4 | Esclarecimento sobre 4.3.1 × 4.3.2 | Ver o item 1 da seção 4: o ganho do RF é menor em DP e Max. Bias, não em Max. Bias Diff. |

Pendente: limite de páginas do FAccT (define o quanto do plano cabe).
