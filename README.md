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

## Contrato

| | |
|---|---|
| `POST /v1/decide` | contexto do ponto de decisão → ação, `decision_id`, versão da política, propensão |
| `POST /v1/feedback` | `decision_id` + desfecho observado, idempotente |
| `POST /v1/close-windows` | fecha como censura as janelas vencidas (agendador) |
| `POST /v1/end-day` | fecha o lote do dia: posterior de hoje vira prior de amanhã |
| `GET /health` | política ativa, versão, decisões, desfechos pendentes |
| `GET /v1/policies` | políticas disponíveis |

Três garantias no contrato:

- **Idempotência.** `decision_id = sha256(user_key | decision_point | policy_version)`.
  Repetir a chamada devolve a decisão já tomada, sem nova amostragem. Sem isso,
  um reenvio de rede viraria exploração adicional e corromperia a propensão.
- **Ação recomendada e executada são campos distintos.** Um envio bloqueado por
  guardrail não é uma abstenção escolhida pela política, e confundir as duas
  invalidaria a análise.
- **Contingência registrada.** Falha ou estouro de tempo limite cai na regra
  conservadora, e o ponto fica marcado com `propensity_method="fallback"`,
  sinalizando que não serve para avaliação off-policy.

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

- O `ppo` precisa de um checkpoint treinado sobre o vetor observável de 16
  dimensões. Os cinco checkpoints do `Spyno-Fogg` observam motivação, habilidade
  e sono exatos, sinais que a formulação atual não coleta; a classe recusa
  carregá-los em vez de fabricar as entradas que faltam.
- `fbm_threshold` depende de `fbm_full` e permanece restrito ao simulador.
- O `spyno-decision` está marcado como não-pacote no seu `pyproject`, então
  entra por `sys.path`. Ajustável com `SPYNO_DECISION_PATH`.
