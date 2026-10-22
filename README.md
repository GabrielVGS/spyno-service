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

## Contrato (versão 2, esquema de política `v2`)

Todos os `/v1/*`, exceto `GET /v1/contract`, exigem
`Authorization: Bearer <SPYNO_DECISION_TOKEN>` (comparação em tempo constante).
Token errado ou ausente: 401. Sem `SPYNO_DECISION_TOKEN` configurado o serviço
responde 503 (fail closed), salvo `SPYNO_DECISION_AUTH=disabled`, só para
desenvolvimento. `GET /health` também é aberto.

| | |
|---|---|
| `GET /v1/contract` | `policy_version`, `outcome_window_min`, features, braços, clipping, guardrails. O backend lê na subida e recusa iniciar se divergir da configuração dele |
| `POST /v1/decide` | contexto do ponto de decisão → ação recomendada e executada, `decision_id`, versão, propensões, distribuição clipada |
| `POST /v1/feedback` | `decision_id` + desfecho, aplicado uma única vez |
| `POST /v1/end-day` | `{"date": "AAAA-MM-DD"}` (data local). O backend chama uma vez por dia; ajusta a posterior com o buffer do dia e o esvazia. Idempotente por data |
| `POST /v1/close-windows` | fecha como censura as janelas vencidas, sem treino |
| `GET /health` | política ativa, versão, decisões, desfechos pendentes |
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
  envio recebe ao menos `SPYNO_ARM_MIX`/3 da massa de envio. A ação é sorteada
  dessa distribuição, e a propensão registrada é exatamente a dela, nunca 0.
  O esquema completo está em `spyno_service/policies/clipping.py`.
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
  são os calculados pelo backend sobre a ação proposta; em shadow eles
  alimentam features e guardrails. O log guarda o contexto bruto e o efetivo,
  com `counter_basis`.
- **Contingência registrada.** Falha ou estouro de tempo limite cai na regra
  conservadora, marcada com `propensity_method="fallback"`.

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

`SPYNO_SHADOW` vem ligado por padrão: nada chega ao usuário por acidente.

## Dívidas conhecidas

- O `ppo` precisa de um checkpoint treinado sobre o vetor observável v2 (13
  dimensões); o `ppo_observable.zip` atual é v1 (14) e é recusado. Os cinco checkpoints do `Spyno-Fogg` observam motivação, habilidade
  e sono exatos, sinais que a formulação atual não coleta; a classe recusa
  carregá-los em vez de fabricar as entradas que faltam.
- A posterior vive em memória: um reinício do serviço volta ao prior.
- `fbm_threshold` depende de `fbm_full` e permanece restrito ao simulador.
- O `spyno-decision` está marcado como não-pacote no seu `pyproject`, então
  entra por `sys.path`. Ajustável com `SPYNO_DECISION_PATH`.
