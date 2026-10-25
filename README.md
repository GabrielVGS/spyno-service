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

## Contrato (versão 2.2, esquema de política `v2`)

Todos os `/v1/*`, exceto `GET /v1/contract`, exigem
`Authorization: Bearer <SPYNO_DECISION_TOKEN>` (comparação em tempo constante).
Token errado ou ausente: 401. Sem `SPYNO_DECISION_TOKEN` configurado o serviço
responde 503 (fail closed), salvo `SPYNO_DECISION_AUTH=disabled`, só para
desenvolvimento. `GET /health` também é aberto.

| | |
|---|---|
| `GET /v1/contract` | `policy_version`, `outcome_window_min`, features, braços, clipping (inclusive `arm_floor` e `max_importance_weight`, desde a 2.1), guardrails e, desde a 2.2, `posterior_init` (`warmstart` \| `prior`) e `warmstart_sha256`. O backend lê na subida e recusa iniciar se divergir da configuração dele |
| `POST /v1/decide` | contexto do ponto de decisão → ação recomendada e executada, `decision_id`, versão, propensões, distribuição clipada |
| `POST /v1/feedback` | `decision_id` + desfecho, aplicado uma única vez |
| `POST /v1/end-day` | `{"date": "AAAA-MM-DD"}` (data local). O backend chama uma vez por dia; ajusta a posterior com o buffer do dia, o esvazia e grava a posterior no banco na mesma transação do fechamento. Idempotente por data |
| `POST /v1/close-windows` | fecha como censura as janelas vencidas, sem treino |
| `GET /v1/policy-state` | hash da posterior, updates pendentes, dias fechados e a origem da posterior (`posterior_source`, `posterior_init`, `persisted_day`, `warmstart`, `skipped_persisted`): confirma de fora que o end-day mudou a posterior e que um reinício a recuperou (sem expor os pesos) |
| `GET /health` | política ativa, versão, decisões, desfechos pendentes e, desde a 2.2, `posterior_source` |
| `GET /v1/policies`, `GET /v1/features` | políticas disponíveis; ordem do vetor |

Garantias:

- **Idempotência por hora.** `decision_id = sha256(user_key | decision_point
  truncado na hora, em UTC | policy_version)`. Ticks repetidos do mesmo usuário
  na mesma hora devolvem a decisão já gravada, sem nova amostragem. O
  `user_key` chega pseudonimizado pelo backend e é tratado como opaco.
- **Ação recomendada e executada são campos distintos.** Um envio bloqueado por
  guardrail não é uma abstenção escolhida pela política.
- **Probability clipping e propensão exata.** Nos pontos elegíveis, P(enviar)
  fica em `[SPYNO_CLIP_MIN, SPYNO_CLIP_MAX]` (padrão [0,1; 0,9]) e cada braço de
  envio tem probabilidade de pelo menos `SPYNO_ARM_FLOOR` (padrão 1/30 ≈ 0,0333;
  ver abaixo). A ação é sorteada dessa distribuição, e a propensão registrada é
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
  ficam no log, mas não treinam.
- **Shadow.** Nada é enviado (`executed_action=0`) e o update usa (x, 0, y),
  nunca a ação proposta. Os contadores `*_proposed` do contexto
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
`warmstart/ts_hybrid-v2-c0.05.json`, e o `docker-compose.yml` o liga por padrão.
`SPYNO_WARMSTART_SNAPSHOT=` (vazio) começa do prior. A receita é a dos
experimentos do `spyno-decision` que mediram o warm-start
(`evaluate.run_evaluation`, `run_hybrid_ablation.py`, `run_fbm_ablation.py`),
com a semente fixa s = 0:

- mundo `build_world(SimConfig(12 usuários, 28 dias, llm_mode="stub"),
  seed_offset=1000+s)`;
- log da política MRT (P(enviar) = 0,25, uniforme nos três braços), desfecho
  com semente 90 000+s;
- `warm_start` original do `spyno-decision`: Laplace sobre os 4 941 pontos
  elegíveis do log, com o custo c = 0,05 da política servida.

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
as três colunas zeradas (`bandit.warm_start_rows`), e um teste confere que a
mesma hora vinda do simulador e vinda do backend dá o mesmo vetor. As linhas do
rollout já trazem `notifs_today` ÷ 5, então não passam por `policy_obs`. Com as
colunas zeradas, os pesos dessas três posições ficam no prior.

**O que o warm-start não é.** O +0,36 E[treinos]/usuário/semana do RQ2 (warm −
cold, sintético) foi medido com o TS *disjoint* com c = 0,03, `fbm_obs` e
sensores com os valores do simulador. A configuração servida (TS híbrido, c =
0,05, três colunas zeradas) não foi medida nesse contraste. Apoiam a escolha as
ablações em que zerar `fbm_obs` (híbrido, c = 0,05) e zerar os sensores dão
diferença n.s. Essas ablações não provam equivalência. O warm-start é
**sintético**: vem do simulador e de fórmulas de feature do simulador.
`fbm_obs` não entra, mas `fatigue`, `recency` e a elegibilidade (teto de 5
envios/dia no simulador, contra cota 3, silêncio e intervalo no serviço) são
calculados pelo backend com regras próprias.

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
| v2 sem piso (`SPYNO_ARM_FLOOR=0`) | 0,1 | 0,0033 | ~300 |
| **padrão**: f = p_min/3 = 1/30 | 0,1 | 0,0333 | **30** |
| `SPYNO_CLIP_MIN=0.3 SPYNO_ARM_FLOOR=0.1` | 0,3 | 0,1 | 10 |

O padrão é f = p_min/3 porque é o maior piso que **não aumenta a exploração
mínima**: P(enviar) ≥ 0,1 continua sendo o único custo fixo, e o peso cai de ~300
para 30 (um fator 10). Pesos ≤ 10 exigiriam f = 0,1 por braço e, com três braços,
P(enviar) ≥ 0,3 em toda hora elegível. Com 16 pontos horários fora do silêncio
(22h–6h), isso dá ≥ 4,8 envios esperados por dia antes de cota e intervalo, contra
≥ 1,6 no padrão: a cota diária de 3 se esgotaria quase todo dia só por exploração,
um custo de fadiga que não se justifica antes de haver dados reais. Por isso o serviço exige
3·f ≤ p_min e recusa subir; quem quiser pesos menores sobe os dois juntos.

O custo do padrão: quando a massa de envio está no mínimo (0,1), os três braços
ficam exatamente em 1/30 e a política não expressa preferência entre eles; acima
do mínimo, os braços que ficariam abaixo de 1/30 sobem para 1/30 e os demais
perdem essa massa na mesma proporção entre si (a razão entre eles se mantém).
Se nenhum braço está abaixo do piso, nada muda. A abstenção tem piso
1 − p_max = 0,1 (peso ≤ 10). Na `ts_binary` (um braço de envio) o piso é
redundante: o braço já recebe a massa toda, ≥ p_min. O peso máximo vale para
alvos com suporte em {0} ∪ S; na `ts_binary`, um alvo que use os braços 1 ou 2
tem peso infinito, porque eles têm π = 0.

**Migração.** O padrão de `SPYNO_ARM_FLOOR` (1/30) supõe `SPYNO_CLIP_MIN` ≥ 0,1.
Quem usa `SPYNO_CLIP_MIN` < 0,1 precisa definir `SPYNO_ARM_FLOOR` ≤
`SPYNO_CLIP_MIN`/3 (por exemplo 0,05/3 para `SPYNO_CLIP_MIN=0.05`), ou `0` para
o comportamento da v2 sem piso; senão o serviço recusa a configuração na subida.

Mudar `SPYNO_ARM_FLOOR` muda a `policy_version` (o clipping entra no hash), e
decisões da versão anterior ficam no log sem treinar a nova.

Contexto: `notifs_today` é a contagem bruta do dia (o guardrail compara com
`SPYNO_DAILY_CAP`); a política a vê ÷ 5, como no simulador. `fbm_obs` continua
aceito, mas desde a v2 não entra no vetor das políticas aprendidas.

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
uv venv && uv pip install -e ".[dev]"
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q     # testes de contrato

SPYNO_POLICY=ts_hybrid SPYNO_SHADOW=1 \
  .venv/bin/uvicorn spyno_service.app:create_app --factory --reload

docker compose up --build                              # Postgres + serviço
```

Ponta a ponta com o backend real (dia simulado, agendador acelerado, export do
log de `Notificacao`): `python3 demo/run_demo.py`, ver `demo/README.md`.

`SPYNO_SHADOW` vem ligado por padrão: nada chega ao usuário por acidente.

| variável | padrão | efeito |
|---|---|---|
| `SPYNO_WARMSTART_SNAPSHOT` | vazio (prior); no compose, `/app/warmstart/ts_hybrid-v2-c0.05.json` | artefato de warm-start da posterior inicial; entra na `policy_version` |

A posterior persistida não tem variável: ela fica no mesmo `SPYNO_DB_URL` das
decisões (tabela `policy_posteriors`, criada na subida).

## Dívidas conhecidas

- O `ppo` precisa de um checkpoint treinado sobre o vetor observável v2 (13
  dimensões); o `ppo_observable.zip` atual é v1 (14) e é recusado. Os cinco checkpoints do `Spyno-Fogg` observam motivação, habilidade
  e sono exatos, sinais que a formulação atual não coleta; a classe recusa
  carregá-los em vez de fabricar as entradas que faltam.
- Os updates de um dia ainda não fechado vivem em memória. Um reinício antes do
  end-day os perde, embora o log já os marque `trained=true`; a posterior volta
  à do último dia fechado.
- A persistência supõe um processo servindo cada `policy_version`, como hoje (o
  lock do end-day é local). Várias réplicas precisariam de lock no banco.
- O warm-start é sintético e o contraste warm − cold não foi medido na
  configuração servida (ver "Warm-start").
- `fbm_threshold` depende de `fbm_full` e permanece restrito ao simulador.
- O `spyno-decision` está marcado como não-pacote no seu `pyproject`, então
  entra por `sys.path`. Ajustável com `SPYNO_DECISION_PATH`.
