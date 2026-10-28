# spyno-service

Serviço de decisão para as notificações adaptativas do Spyno. Uma interface,
várias políticas: trocar Thompson Sampling por PPO ou por uma política de
referência é mudar uma variável de ambiente, sem tocar em quem consome.

## Por que não há fila nem broker

Na escala do Spyno — algumas dezenas a centenas de usuários, pontos de decisão
horários, algo como duas mil decisões por dia — o Postgres é a fila. O estado
das janelas de atribuição vive na própria tabela de decisões e um agendador em
processo fecha o que vence.

O caminho de decisão é **síncrono** de propósito. O aplicativo pergunta "notifico
agora?" e precisa da resposta agora, com tempo limite e regra de contingência.
Uma fila na frente disso acrescentaria latência e quebraria a idempotência de
que a propensão registrada depende.

O que o trabalho precisa da implementação não é vazão, são quatro propriedades
de correção: propensão registrada corretamente, decisão reproduzível a partir do
log, atribuição de desfecho auditável e modo de observação que impeça qualquer
envio acidental. Fila distribuída torna essas quatro mais difíceis de verificar.

## Contrato (versão 2.3, esquema de política `v2`)

Todos os `/v1/*`, exceto `GET /v1/contract`, exigem
`Authorization: Bearer <SPYNO_DECISION_TOKEN>` (comparação em tempo constante).
Token errado ou ausente: 401. Sem `SPYNO_DECISION_TOKEN` configurado o serviço
responde 503 (fail closed), salvo `SPYNO_DECISION_AUTH=disabled`, só para
desenvolvimento. `GET /health` também é aberto.

| | |
|---|---|
| `GET /v1/contract` | `policy_version`, `outcome_window_min`, features, braços, clipping (inclusive `arm_floor` e `max_importance_weight`, desde a 2.1), guardrails, desde a 2.2, `posterior_init` (`warmstart` \| `prior`) e `warmstart_sha256` e, desde a 2.3, `learning_frozen` e `shadow_learn`. O backend lê na subida e recusa iniciar se divergir da configuração dele |
| `POST /v1/decide` | contexto do ponto de decisão → ação recomendada e executada, `decision_id`, versão, propensões, distribuição clipada, `update_eligible` e, desde a 2.3, `learning_frozen` e `shadow_override` |
| `POST /v1/feedback` | `decision_id` + desfecho, aplicado uma única vez |
| `POST /v1/end-day` | `{"date": "AAAA-MM-DD"}` (data local). O backend chama uma vez por dia; ajusta a posterior com o buffer do dia, o esvazia e grava a posterior no banco na mesma transação do fechamento. Idempotente por data. Com a posterior congelada (shadow) só fecha o dia (`learning_frozen=true`, 0 updates) |
| `POST /v1/close-windows` | fecha como censura as janelas vencidas, sem treino |
| `GET /v1/policy-state` | hash da posterior, updates pendentes, dias fechados e a origem da posterior (`posterior_source`, `posterior_init`, `persisted_day`, `warmstart`, `skipped_persisted`) e, desde a 2.3, `learning_frozen`: confirma de fora que o end-day mudou a posterior e que um reinício a recuperou (sem expor os pesos) |
| `GET /health` | política ativa, versão, decisões, desfechos pendentes, desde a 2.2, `posterior_source` e, desde a 2.3, `learning_frozen` |
| `GET /v1/policies`, `GET /v1/features` | políticas disponíveis; ordem do vetor |

Garantias:

- **Idempotência por hora.** `decision_id = sha256(user_key | decision_point
  truncado na hora, em UTC | policy_version)`. Ticks repetidos do mesmo usuário
  na mesma hora devolvem a decisão já gravada, sem nova amostragem. O
  `user_key` chega pseudonimizado pelo backend e é tratado como opaco.
- **Ação recomendada e executada são campos distintos.** Um envio bloqueado por
  guardrail não é uma abstenção escolhida pela política.
- **Probability clipping e propensão exata.** Nos pontos elegíveis, P(enviar)
  fica em `[SPYNO_CLIP_MIN, SPYNO_CLIP_MAX]` (padrão [0,05; 0,9], "p05", desde
  2026-10-26; antes [0,1; 0,9]) e cada braço de envio tem probabilidade de pelo
  menos `SPYNO_ARM_FLOOR` (padrão 0,05/3 ≈ 0,0167; antes 1/30; ver abaixo). A
  ação é sorteada dessa distribuição, e a propensão registrada é
  exatamente a dela, nunca 0. O esquema completo está em
  `spyno_service/policies/clipping.py`.
  `recommended_propensity` é P(ação recomendada); `propensity` é P(ação
  executada) sob o processo de coleta, que vale 1 quando guardrail ou shadow
  forçam a abstenção.
- **Treino.** `responded` → y=1, `not_responded` → y=0, sempre sobre a
  `executed_action` e para qualquer braço, inclusive o 0. `censored`,
  `delivery_failed` e respostas fora da janela só fecham o pendente. Só treinam
  pontos com `update_eligible`: elegíveis pelo backend, sem guardrail que
  bloquearia um envio e sem contingência. Decisões de outra `policy_version`
  ficam no log, mas não treinam. Com a posterior congelada (shadow, abaixo)
  nada treina.
- **Shadow: a posterior não aprende (desde a 2.3).** Nada é enviado
  (`executed_action=0`) e, com `SPYNO_SHADOW_LEARN=0` (padrão), o shadow é só
  coleta de logs:
  - `/v1/decide` registra tudo como antes (contexto bruto e efetivo,
    `counter_basis`, propensões, `arm_probs`, `proposed_action`) e marca
    `learning_frozen=true` na resposta e na coluna `decisions.learning_frozen`;
  - `update_eligible` **continua** dizendo se o ponto seria treinável
    (elegível, sem guardrail bloqueando, sem contingência): é a seleção que a
    OPE precisa reproduzir sobre esses logs;
  - `/v1/feedback` grava o desfecho, mas não chama `update`:
    `trained=false`, `train_skip_reason="decisao_em_shadow_congelado"` (ou o
    motivo anterior, como `sem_treino_censored`, quando ele já impediria o treino);
  - `/v1/end-day` fecha o dia em `policy_days` com `updates_applied=0`, de forma
    idempotente, **sem** chamar `policy.end_day()` e **sem** gravar posterior
    em `policy_posteriors` (escolha: a posterior persistida só nasce de
    aprendizado; uma linha gravada em shadow seria cópia do estado inicial);
  - o digest em `/v1/policy-state` fica igual ao da subida durante todo o
    shadow, e a `policy_version` não depende do shadow.

  Por quê: o warm-start servido foi validado fora do shadow
  (`spyno-decision`, `exp/warmstart-served`). Treinar em shadow a modificaria só
  com dados do braço 0 e contadores hipotéticos (`*_proposed`), e esse estado
  pós-shadow nunca foi testado. Ao sair do shadow (reinício com
  `SPYNO_SHADOW=0`), a posterior carregada é exatamente a que entrou: o
  artefato de warm-start, ou a última posterior persistida fora do shadow.
  Desfechos de decisões tomadas em shadow que chegam depois do reinício também
  não treinam (`decisao_em_shadow_congelado`). Desfechos de decisões treináveis
  tomadas fora do shadow que chegam com o serviço já em shadow não treinam
  (`aprendizado_congelado`). Linhas antigas sem a coluna `learning_frozen`
  contam como congeladas se eram de shadow.

  **Posterior aprendida em shadow não volta.** Cada linha de
  `policy_posteriors` registra `learned_in_shadow` (gravada pelo end-day: true
  só em shadow com `SPYNO_SHADOW_LEARN=1`). Na subida, sem
  `SPYNO_SHADOW_LEARN=1`, uma posterior da mesma `policy_version` **não** é
  carregada se `learned_in_shadow=true` (`reason="aprendida_em_shadow"`) ou se a
  coluna é nula (linha anterior à 2.3) e o log tem decisões de shadow dessa
  versão que treinaram
  (`reason="origem_desconhecida_com_shadow_treinando_no_historico"`). O serviço
  sobe do estado inicial, registra um WARNING e lista a linha em
  `skipped_persisted` (`/v1/policy-state`). A marca é **hereditária**: se a
  posterior carregada (com `SPYNO_SHADOW_LEARN=1`) era marcada, todo end-day
  dela, mesmo fora do shadow, grava `learned_in_shadow=true`;
  `/v1/policy-state` expõe `learned_in_shadow` da posterior em memória. A linha não é apagada, mas o
  próximo end-day fora do shadow a sobrescreve (uma linha por versão). Linha
  nula sem shadow treinando no log é carregada: se o log foi apagado, a origem
  não é detectável. Como há uma linha por versão, uma posterior aprendida fora
  do shadow e depois continuada em shadow com `SPYNO_SHADOW_LEARN=1` não
  é recuperável sem o aprendizado em shadow: a subida volta ao estado inicial.

  **Decisão repetida em shadow.** Se uma decisão gravada fora do shadow é
  pedida de novo com o serviço em shadow (reinício na mesma hora, mesmo
  `decision_id`), a resposta força `executed_action=0`, `propensity=1`,
  `shadow=true`, `guardrail_reason="shadow"`, `learning_frozen=true` e marca
  `shadow_override=true`. A decisão original no log não é reescrita, mas a
  linha ganha `shadow_override_at` (instante da primeira repetição forçada). A
  partir daí:
  - toda repetição dessa decisão, também fora do shadow, devolve a mesma
    abstenção forçada;
  - o desfecho dela nunca treina (`train_skip_reason="override_shadow"`),
    qualquer que seja o modo quando ele chega;
  - `update_eligible` continua o decidido (a seleção treinável, como nas
    linhas de shadow), e é isso que o backend grava em `atualiza_politica`.

  **Leitura para a OPE:** linha com `shadow_override_at` não nulo tem ação
  efetiva 0 com propensão 1 (abstenção forçada), não a `executed_action`
  gravada. Exceção: se o backend executou a primeira resposta (fora do shadow)
  antes da repetição, o envio aconteceu; confira na `Notificacao` (`entregue_em`)
  antes de usar a linha. Um desfecho aplicado antes da repetição pode ter
  treinado (`trained=true` com `outcome_at` < `shadow_override_at`).

  **Transição.** O backend lê `shadow` do `/v1/contract` só na subida. Para
  entrar ou sair do shadow, reinicie o serviço **e** o backend.

  `SPYNO_SHADOW_LEARN=1` restaura o comportamento anterior à 2.3 (update com
  (x, 0, y), sendo x calculado com os `*_proposed`), que não foi validado.

  Os contadores `*_proposed` do contexto
  (`notifs_today_proposed`, `fatigue_proposed`, `minutes_since_last_proposed`)
  são os calculados pelo backend sobre a ação proposta, devolvida em
  `proposed_action`: a que teria sido executada sem o shadow, ou seja, a
  recomendada depois dos guardrails reais (recomendação bloqueada por silêncio,
  cota ou intervalo não conta). Fora do shadow ela é a executada. Em shadow os
  `*_proposed` alimentam features e guardrails. O log guarda o contexto bruto e o efetivo,
  com `counter_basis`.
- **Contingência registrada.** Falha ou estouro de tempo limite cai na regra
  conservadora, marcada com `propensity_method="fallback"`.
- **Posterior persistida.** Cada `/v1/end-day` bem-sucedido grava o snapshot
  (m, q, custo, esquema, features, digest) em `policy_posteriors`, uma linha por
  `policy_version`, **na mesma transação** que marca o dia fechado em
  `policy_days`. Se a gravação falha, o dia continua aberto e o end-day responde
  `ok=false` (`error="persistencia_*"`); a nova tentativa fecha o dia. Um
  end-day repetido na mesma data não refaz o ajuste nem regrava a posterior.

### Origem da posterior na subida (contrato 2.2)

Ordem de precedência, registrada no log e em `posterior_source`
(`/health`, `/v1/policy-state`):

1. **`persisted`**: a posterior gravada pelo último end-day da **mesma**
   `policy_version`. Confere-se o digest gravado e o esquema
   (`bandit._check_snapshot`); se não conferir, o serviço não sobe.
2. **`warmstart`**: o artefato de `SPYNO_WARMSTART_SNAPSHOT`.
3. **`prior`**: a posterior inicial da fábrica.

A `policy_version` é calculada sobre o estado **inicial** (prior ou warm-start),
antes de aplicar a persistida:

- **O warm-start muda a `policy_version`**, porque é outro modelo inicial.
  Decisões tomadas a partir do prior não treinam uma posterior aquecida, nem o
  contrário. `/v1/contract` expõe `posterior_init` e `warmstart_sha256`.
- **A posterior persistida não muda a `policy_version`.** Ela continua a mesma
  linhagem: um reinício não invalida o `decision_id` das decisões já tomadas nem
  descarta os desfechos que ainda vão chegar.
- Sem `SPYNO_WARMSTART_SNAPSHOT`, a `policy_version` é a mesma de antes desta
  mudança.

Uma posterior persistida de **outra** `policy_version` (outro esquema, outro
clipping, outro warm-start, sem warm-start) nunca é carregada. O serviço sobe
pela regra 2 ou 3, registra um WARNING e lista as versões ignoradas em
`skipped_persisted`. As linhas continuam no banco: voltar à configuração
anterior recupera aquela posterior.

Um warm-start pedido e incompatível derruba a subida (`IncompatibleSnapshotError`
ou `FileNotFoundError`), em vez de servir o prior sem avisar. São recusados:
formato diferente de `spyno-warmstart/1`, esquema diferente de `v2`, outra lista
de features, outra política (`ts_disjoint` e `ts_binary` têm posteriors de mesma
forma), outro custo, outra dimensão, ou política sem posterior (`never`,
`always`, `ppo`).

### Warm-start

`training/build_warmstart.py` gera o artefato versionado
`warmstart/ts_hybrid-v2-c0.1.json`, e o `docker-compose.yml` o liga por padrão.
O `ts_hybrid-v2-c0.05.json` fica no repositório: é o artefato do padrão anterior
(c = 0,05), conferido por sha256 nos experimentos do `spyno-decision`.
`SPYNO_WARMSTART_SNAPSHOT=` (vazio) começa do prior. A receita é a dos
experimentos do `spyno-decision` que mediram o warm-start
(`evaluate.run_evaluation`, `run_hybrid_ablation.py`, `run_fbm_ablation.py`),
com a semente fixa s = 0:

- mundo `build_world(SimConfig(12 usuários, 28 dias, llm_mode="stub"),
  seed_offset=1000+s)`;
- log da política MRT (P(enviar) = 0,25, uniforme nos três braços), desfecho
  com semente 90 000+s;
- `warm_start` original do `spyno-decision`: Laplace sobre os 4 941 pontos
  elegíveis do log, com o custo c = 0,1 da política servida. O ajuste não
  depende de c: os dois artefatos têm m e q idênticos bit a bit e só diferem no
  campo `cost` e na proveniência.

O ajuste é determinístico: duas execuções geram os mesmos m e q (só a proveniência
muda com os commits), e um teste (pulado sem o `spyno-sim`)
confere isso. O artefato registra a receita, os commits do `spyno-sim` e do
`spyno-service`, o sha256 dos arquivos vendorizados do `spyno-decision` e um
*crosscheck*: o mesmo ajuste refeito com o `spyno-decision` master (`74d1437`)
deu m e q idênticos bit a bit.

```sh
SPYNO_SIM_PATH=../spyno-sim .venv/bin/python training/build_warmstart.py \
    --crosscheck-decision ../_wt/decision-master
```

**Vetor de features.** O serviço serve o vetor v2 de 13 dimensões (12 atributos
mais o viés). As classes do `spyno-decision` vetorizam 16 posições, e o
adaptador entrega `location_home`, `motion_stationary` e `fbm_obs` sempre
zerados. O warm-start é feito **nesse mesmo vetor**: as linhas do simulador têm
as três colunas zeradas (`bandit.warm_start_rows`). As linhas do rollout já
trazem `notifs_today` ÷ 5, então não passam por `policy_obs`. Com as colunas
zeradas, os pesos dessas três posições ficam no prior. Um teste confere que a
mesma observação dá o mesmo vetor depois do adaptador (contexto do contrato,
`notifs_today` bruto) e depois de `warm_start_rows` (linha do simulador,
`notifs_today` ÷ 5). Ele cobre só essa transformação, os zeros e o ÷ 5. Não
mostra que o backend calcula as features como o simulador; ver abaixo.

**Identidade da linhagem.** Quem identifica a linhagem é a `policy_version`, não
o `warmstart_sha256`. O sha256 é do arquivo inteiro, proveniência incluída:
regerar o artefato em outro commit muda o sha256, mas mantém m, q e, portanto, a
`policy_version`, e a posterior persistida continua valendo.

**O que o warm-start não é.** O +0,36 E[treinos]/usuário/semana do RQ2 (warm −
cold, sintético, 10 sementes) foi medido com o TS *disjoint* com c = 0,03,
`fbm_obs` e sensores com os valores do simulador. A configuração servida (TS
híbrido, c = 0,05, três colunas zeradas) não foi medida nesse contraste.
(Depois foi: `exp/warmstart-served` no `spyno-decision`, com c = 0,05 e p10.)

O que apoia a escolha das colunas, com pesos diferentes:

- **`fbm_obs`.** `run_fbm_ablation.py` no `spyno-decision`, com warm-start, c =
  0,05, n = 20 sementes e Holm: zerar a coluna dá n.s. É a evidência boa, mas
  n.s. não prova equivalência.
- **Sensores.** `training/ablacao_sensores.py` é exploratória: cold start, 10
  sementes, IC normal sem Holm e nenhum `results/*.json`. Além disso, hoje ela é
  inócua: `build_policy("ts_hybrid")` já zera os sensores no adaptador, então a
  condição "completo" também roda sem eles. Vale só como indício.
- **As duas remoções juntas**, que é o que o serviço faz, nunca foram medidas.

O warm-start é **sintético**: vem do simulador e das fórmulas de feature do
simulador. O backend calcula as features com regras próprias, e há descompassos
de **semântica** anteriores a esta mudança:

- **`in_free_window`.** No simulador é a janela livre declarada pela persona. No
  backend é uma aproximação, "acordado e fora do expediente"
  (`decision_features.js`, `janelaLivre`). É a feature de maior peso no
  warm-start: m = +2,25 no bloco base e +0,54 no bloco de envio.
- **`valence_pos`.** No simulador é 0 ou 1. No backend vale 0,5 quando não há
  feedback de rotina nas últimas 48 h, um valor fora do suporte dos dados do
  warm-start.
- **`recency`.** No simulador, log1p(h)/log1p(48) passa de 1 depois de 48 h
  parado. No backend ela é saturada em 1.
- **Elegibilidade.** O simulador tem teto de 5 envios/dia. O serviço tem cota 3,
  silêncio e intervalo.
- **`fatigue`.** O backend a reconstrói do histórico com regras próprias.

Por isso o warm − cold na configuração servida e com features calculadas no
estilo do backend **não foi medido**. Em produção, o warm-start pode render menos
que no simulador, ou nada.

### Piso por braço e peso de importância (contrato 2.1)

Na v2 original a massa de envio era clipada em [0,1; 0,9] e repartida entre os
três braços com mistura uniforme η = `SPYNO_ARM_MIX` = 0,1. Isso deixava um piso
de p_min·η/3 ≈ 0,0033 por braço: para uma política-alvo que escolhe sempre um
braço específico, o peso de importância π_alvo/π_log chegava a ~300, e a OPE
sobre os logs do serviço ficaria com variância enorme.

Desde a 2.1 há um piso explícito f = `SPYNO_ARM_FLOOR` por braço de envio,
aplicado depois do clip da massa total: os braços abaixo de f sobem para f e os
demais repartem o resto na mesma proporção (water-filling). A massa de envio, e
portanto a soma 1 e P(enviar) ∈ [p_min, p_max], não mudam. Com isso, o peso
máximo para qualquer alvo determinístico é 1/min(f, 1 − p_max), exposto em
`/v1/contract` como `max_importance_weight`.

| configuração | P(enviar) mínima | π mínima por braço | peso máximo |
|---|---|---|---|
| v2 sem piso (`SPYNO_CLIP_MIN=0.1 SPYNO_ARM_FLOOR=0`) | 0,1 | 0,0033 | ~300 |
| **padrão p05** (desde 2026-10-26): p_min = 0,05, f = p_min/3 = 0,05/3 | 0,05 | 0,0167 | **60** |
| p10 (padrão até 2026-10-25): `SPYNO_CLIP_MIN=0.1 SPYNO_ARM_FLOOR=0.03333333333333333` | 0,1 | 0,0333 | 30 |
| `SPYNO_CLIP_MIN=0.3 SPYNO_ARM_FLOOR=0.1` | 0,3 | 0,1 | 10 |

f = p_min/3 é o maior piso que **não aumenta a exploração mínima**: P(enviar) ≥
p_min continua sendo o único custo fixo. Pesos ≤ 10 exigiriam f = 0,1 por braço e,
com três braços, P(enviar) ≥ 0,3 em toda hora elegível. Com 16 pontos horários fora
do silêncio (22h–6h), isso dá ≥ 4,8 envios esperados por dia antes de cota e
intervalo, contra ≥ 0,8 com p05 (≥ 1,6 com p10): a cota diária de 3 se esgotaria
quase todo dia só por exploração, um custo de fadiga que não se justifica antes de
haver dados reais. Por isso o serviço exige 3·f ≤ p_min e recusa subir; quem quiser
pesos menores sobe os dois juntos. O piso tem de ser p_min/3 em double ou menor:
`SPYNO_ARM_FLOOR=0.016667` (arredondado para cima) é **recusado** com
`SPYNO_CLIP_MIN=0.05`; use `0.016666666666666666` (o `repr` de 0.05/3).

**Por que p05 e não p10 (evidência só sintética).** No simulador, na configuração
servida (warm-start, features do backend, cota 3, então com c = 0,05), baixar o piso de 0,1 para 0,05
aumentou E[treinos]/usuário/semana em 10 de 22 contrastes (t pareado + Holm, m = 66,
e IC95 bootstrap pareado; 20 sementes held-out, mesmas sementes nas variantes),
nenhum significativamente negativo; no PMData ficou n.s. O mecanismo proposto (o
piso força envios onde o TS não enviaria e gasta a cota) é hipótese, não isolada.
O custo vai para a OPE: o peso máximo dobra (30 → 60) e, para um alvo MRT p = 0,25,
o ESS/n caiu de ≈ 0,60–0,65 para ≈ 0,48–0,53. Pisos menores (p02, peso 150) renderam
mais no simulador, mas pioram a OPE e não foram adotados. Nada disso foi medido com
usuários reais. Detalhes: `results/floor_served_conclusao.md` no spyno-decision
(`run_floor_served.py`).

O custo do piso: quando a massa de envio está no mínimo (p_min), os três braços
ficam exatamente em f e a política não expressa preferência entre eles; acima
do mínimo, os braços que ficariam abaixo de f sobem para f e os demais
perdem essa massa na mesma proporção entre si (a razão entre eles se mantém).
Se nenhum braço está abaixo do piso, nada muda. A abstenção tem piso
1 − p_max = 0,1 (peso ≤ 10). Na `ts_binary` (um braço de envio) o piso é
redundante: o braço já recebe a massa toda, ≥ p_min (peso ≤ 1/0,05 = 20 com p05). O peso máximo vale para
alvos com suporte em {0} ∪ S; na `ts_binary`, um alvo que use os braços 1 ou 2
tem peso infinito, porque eles têm π = 0.

**Migração p10 → p05 (2026-10-26).** O padrão de `SPYNO_ARM_FLOOR` (0,05/3) supõe
`SPYNO_CLIP_MIN` ≥ 0,05. Quem usa `SPYNO_CLIP_MIN` < 0,05 precisa definir
`SPYNO_ARM_FLOOR` ≤ `SPYNO_CLIP_MIN`/3, ou `0` para o comportamento da v2 sem
piso; senão o serviço recusa a configuração na subida. Para manter o p10, defina
os dois: `SPYNO_CLIP_MIN=0.1 SPYNO_ARM_FLOOR=0.03333333333333333`.

Mudar o clipping muda a `policy_version` (o clipping entra no hash), e:

- o artefato de warm-start não depende do clipping (ele é conferido por formato,
  esquema, features, política e custo) e carrega na nova versão;
- uma posterior persistida da versão p10 **não** é carregada: a subida não falha,
  a linhagem p05 recomeça do warm-start (ou do prior) e a linha antiga aparece em
  `skipped_persisted` com `reason: outra_policy_version` no `/v1/policy-state`
  (e no log). O que a linhagem p10 aprendeu fora do shadow se perde; para manter,
  continue em p10;
- decisões da versão anterior ficam no log, e desfechos delas que chegarem depois
  são registrados sem treinar a nova (`policy_version_diferente`);
- o `decision_id` inclui a `policy_version`: um tick na mesma hora da troca gera
  uma decisão nova.

O contrato continua 2.3: os campos `clip_min`, `arm_floor` e
`max_importance_weight` já existiam e só mudam de valor padrão, e o backend não
os lê (ele confere a janela `outcome_window_min` e as features, e lê `shadow` e
`daily_cap`). Desde 2026-10-26 eles reportam o clipping da política servida (o mesmo da
`policy_version`), não uma cópia da configuração.

Contexto: `notifs_today` é a contagem bruta do dia (o guardrail compara com
`SPYNO_DAILY_CAP`); a política a vê ÷ 5, como no simulador. `fbm_obs` continua
aceito, mas desde a v2 não entra no vetor das políticas aprendidas.

### Custo de envio c = 0,1 (desde 2026-10-28)

O `ts_hybrid` servido passa a usar c = 0,1 (antes 0,05), com o artefato
`warmstart/ts_hybrid-v2-c0.1.json`. A escolha vem do `spyno-decision`
(sintético, simulador; `results/cost_served_conclusao.md` e
`results/served_divergences_conclusao.md`):

- `run_cost_served.py`: na configuração servida (p05, warm-start do serviço,
  features do backend), c* = 0,1 foi escolhido nas sementes de seleção
  300–304, numa grade {0; 0,02; 0,05; 0,1; 0,2};
- held-out 100–119 (n = 20, mesmas sementes), com cota 3 e janela de 180 min
  (`run_served_divergences.py`, família de 22 células): c 0,1 − 0,05 deu 6/22
  contrastes sig. positivos em E[treinos]/usuário/semana (4 deles nas 8
  células de 24 usuários × 56 dias acrescentadas nesse experimento) e nenhum sig. negativo, e 1,1–4,2 envios
  a menos por usuário por semana (22/22 sig.).

A exceção são os mundos de resposta baixa: no intercepto −8 com 12 usuários,
c 0,1 − 0,05 deu −0,049 [−0,077; −0,022], sig. na família de 14 células do
`cost_served` (p_Holm 0,040) e n.s. por Holm na de 22 (p_Holm 0,053), com o
mesmo dado; com as features corrigidas do backend, o PMData 24u também ficou
negativo (−0,051, IC excluindo 0, n.s. por Holm). n.s. não prova
equivalência: a defesa do 0,1 é sobretudo o volume menor, sem perda detectável
de treinos na maioria dos mundos.

O c = 0,1 vale só para o `ts_hybrid`; `ts_disjoint` e `ts_binary` continuam
com o padrão 0,05. Os resultados servidos anteriores (warm-start servido, shadow
congelado, piso p05) foram medidos com c = 0,05 e não foram refeitos com 0,1.

O custo está no snapshot da posterior, então entra na `policy_version`, como
o clipping (ver acima):

- a posterior persistida da linhagem c = 0,05 **não** é carregada: a linhagem
  c = 0,1 recomeça do warm-start e a antiga aparece em `skipped_persisted`;
- o artefato `c0.05` é recusado com c = 0,1 (outro custo). Para voltar ao padrão
  anterior, não basta trocar o artefato: o custo vem do código
  (`bandit.ts_hybrid`), sem variável de ambiente.

## Políticas

| nome | origem | servível em produção |
|---|---|---|
| `ts_hybrid` | `spyno-decision`, *pooling* parcial | sim (propensão acrescentada aqui) |
| `ts_disjoint`, `ts_binary` | `spyno-decision` | sim |
| `never`, `always` | referências | sim |
| `fbm_obs_threshold` | limiar sobre o escore observável | sim |
| `fbm_threshold` | heurística original, lê o FBM completo | **não**, só no simulador |
| `ppo` | checkpoint do Stable-Baselines3 | requer checkpoint sobre o vetor observável |

As políticas do `spyno-decision` são usadas sem alteração: aquele pacote é a
fonte única das políticas validadas, e foi ele que produziu os resultados de
avaliação, ablação, sensibilidade e OPE.

## Rodando

```sh
uv sync --frozen --extra dev                           # ambiente do uv.lock (Python 3.12)
uv run --frozen pytest -q                              # testes de contrato

SPYNO_POLICY=ts_hybrid SPYNO_SHADOW=1 \
  .venv/bin/uvicorn spyno_service.app:create_app --factory --reload

docker compose up --build                              # Postgres + serviço
```

A imagem instala exatamente as versões do `uv.lock` (`uv sync --frozen`),
roda como o usuário sem privilégios `spyno` (uid 10001) e tem `HEALTHCHECK` no
`/health`. Base e uv estão fixados (`python:3.12.16-slim-trixie`, uv 0.11.7).
`--build-arg COM_PPO=1` acrescenta o extra `ppo` (torch).

Ponta a ponta com o backend real (dia simulado, agendador acelerado, export do
log de `Notificacao`): `python3 demo/run_demo.py`, ver `demo/README.md`.

`SPYNO_SHADOW` vem ligado por padrão: nada chega ao usuário por acidente.

| variável | padrão | efeito |
|---|---|---|
| `SPYNO_WARMSTART_SNAPSHOT` | vazio (prior); no compose, `/app/warmstart/ts_hybrid-v2-c0.1.json` | artefato de warm-start da posterior inicial; entra na `policy_version` |
| `SPYNO_SHADOW_LEARN` | `0` | `1`: em shadow, a posterior aprende com (x, 0, y), como antes da 2.3 (não validado). Sem efeito fora do shadow; não entra na `policy_version` |

A posterior persistida não tem variável: ela fica no mesmo `SPYNO_DB_URL` das
decisões (tabela `policy_posteriors`, criada na subida).

## Desenvolvimento

```sh
uv sync --frozen --extra dev      # + --extra ppo para o PPO (torch), --extra postgres
```

`uv sync` é exato: remove extras não pedidos. Passe todos os extras na mesma
linha (ex.: `uv sync --frozen --extra dev --extra ppo`).

```sh
make lint                         # ruff check (regras em pyproject.toml)
make test                         # pytest
make ci                           # lint + test, como na CI
make docker                       # docker build local
```

Mudou alguma dependência no `pyproject.toml`? Rode `uv lock` e versione o
`uv.lock`. A CI (`.github/workflows/ci.yml`) roda `uv sync --frozen --extra dev`,
`ruff check` e `pytest`, e um job separado faz `docker build` (sem push). Sem o
`spyno-sim` ao lado, os testes que rodam o simulador são pulados.

## Dívidas conhecidas

- O `ppo` precisa de um checkpoint treinado sobre o vetor observável v2 (13
  dimensões); o `ppo_observable.zip` atual é v1 (14) e é recusado. Os cinco checkpoints do `Spyno-Fogg` observam motivação, habilidade
  e sono exatos, sinais que a formulação atual não coleta; a classe recusa
  carregá-los em vez de fabricar as entradas que faltam.
- Os updates de um dia ainda não fechado vivem em memória. Um reinício antes do
  end-day os perde, embora o log já os marque `trained=true`; a posterior volta
  à do último dia fechado.
- Um dia fechado em shadow fica fechado: se o serviço sai do shadow no mesmo
  dia local, o end-day dessa data responde `already_closed` e os updates do
  resto do dia só entram no ajuste do dia seguinte. `policy_days` não marca
  que o fechamento foi congelado (só `updates_applied=0`).
- Uma decisão tomada em shadow e repetida na mesma hora depois de sair do
  shadow devolve a decisão de shadow já gravada (`decision_id` não depende do
  shadow): nada é enviado nessa hora. No sentido inverso (fora do shadow →
  shadow), a resposta é forçada a 0 com `shadow_override` (ver Shadow), e o
  log do serviço e o que o backend gravar dessa resposta podem divergir.
- Linhas de `decisions` anteriores à 2.3 (`learning_frozen` nulo) com
  `shadow=true` são tratadas como congeladas no treino, o que é conservador.
  Para a OPE elas devem ser lidas como **desconhecido**: antes da 2.3 o shadow
  treinava, e `trained` diz o que de fato aconteceu.
- O backend não grava `learning_frozen` nem `shadow_override` na `Notificacao`.
  A fonte certa para a OPE é a tabela `decisions` do serviço.
- Os logs de shadow, com ou sem congelamento, só sustentam OPE de alvos que
  se abstêm: a ação executada é sempre 0, com propensão 1. `proposed_action` e
  `arm_probs` servem para auditar o que a política teria feito, não para
  estimar o valor de alvos que enviam. O que eles dão é o desfecho sob
  abstenção, por contexto, e a seleção `update_eligible`.
- A persistência supõe um processo servindo cada `policy_version`, como hoje (o
  lock do end-day é local). Várias réplicas precisariam de lock no banco.
- O warm-start é sintético, e o contraste warm − cold não foi medido na
  configuração servida nem com as features do backend (ver "Warm-start").
- `training/ablacao_sensores.py` não compara de fato com e sem sensores: o
  adaptador zera os sensores nas três condições.
- `fbm_threshold` depende de `fbm_full` e permanece restrito ao simulador.
- O `spyno-decision` entra como cópia vendorizada em `spyno_decision/`
  (`spyno_decision/PROVENIENCIA.md`), sincronizada à mão. A subida do serviço não
  confere o sha256 desses arquivos; quem confere é o artefato de warm-start (que
  registra os sha256) e o `spyno-decision` (blobs aceitos). Os scripts de
  `training/` importam o `spyno-sim` por `sys.path` (`SPYNO_SIM_PATH`).
- Na imagem, `/app` pertence ao usuário `spyno` para que o SQLite padrão
  (`./spyno_service.db`) possa ser criado. O código é do root, mas o usuário
  pode renomear entradas de `/app`. Separar um diretório de dados gravável
  muda o caminho padrão do SQLite e fica para depois. As imagens e as actions
  estão fixadas por tag, não por digest.
