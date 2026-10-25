# Demo F4: um dia simulado de ponta a ponta

Backend real (Node/Express/Prisma) + `spyno-service` real + Postgres, no
Compose. Um roteiro sobe tudo, cria usuários sintéticos, liga o agendador do
backend com um relógio acelerado, simula treinos ao longo de um dia inteiro e a
virada do dia, exporta o log de `Notificacao` e confere o contrato entre os dois
lados.

```sh
# na raiz do spyno-service; requer docker com compose v2 e python3 (só stdlib)
python3 demo/run_demo.py                  # shadow mode (padrão)
python3 demo/run_demo.py --sem-shadow     # envio real, entrega FCM simulada
python3 demo/run_demo.py --manter         # deixa os contêineres de pé no fim
BACKEND_DIR=/caminho/do/Spyno_Backend python3 demo/run_demo.py
python3 demo/run_demo.py --warmstart --reiniciar   # warm-start + reinício do serviço
```

`--warmstart` sobe o serviço com `SPYNO_WARMSTART_SNAPSHOT` apontando para o
artefato versionado. Sem a opção, o serviço começa do prior, e a demo confere
`posterior_init` no contrato nos dois casos. `--reiniciar` reinicia o contêiner
`decision` depois do end-day e confere quatro coisas: `posterior_source =
persisted`, o mesmo digest em `/v1/policy-state`, a mesma `policy_version` e o
mesmo `posterior_init`. A saída vai para `demo/out/<modo>[-warmstart][-reinicio]/`.

Uma execução leva uns 3 minutos (a maior parte é o build). Termina com
`TUDO OK` ou com a lista de falhas e código de saída 1.

## O que sobe

| serviço | imagem | porta no host |
|---|---|---|
| `db` | `postgres:16-alpine`, bancos `spyno` (backend) e `spyno_decision` (serviço) | `127.0.0.1:55480` |
| `decision` | `Dockerfile` deste repo, `ts_hybrid`, `SPYNO_SHADOW=1` | `127.0.0.1:58480` |
| `api` | `Dockerfile` de `$BACKEND_DIR` (padrão `../../_wt/backend-pr2`) | `127.0.0.1:53480` |
| `migrate` | a imagem da API, `prisma migrate deploy` (perfil `setup`) | — |

Projeto Compose `spyno-demo-f4`. As portas fogem das padrão e podem ser
trocadas por `DEMO_PG_PORT`, `DEMO_DECISION_PORT` e `DEMO_API_PORT`. No começo e
no fim o roteiro faz `docker compose -p spyno-demo-f4 down -v`, o que só afeta os
contêineres e volumes deste projeto. Os segredos ficam em `demo.env` e são
fictícios.

## Relógio

Acelerar só `SPYNO_TICK_MS` não simula um dia. O `decision_point` é truncado na
hora cheia e o serviço deduplica por ele, então vários ticks na mesma hora real
viram uma decisão só. Por isso o backend tem um relógio de demonstração
(`src/services/relogio.js`), que só é ligado com `SPYNO_DEMO_CLOCK=on` e é recusado
com `NODE_ENV=production`. Cada `SPYNO_TICK_MS` reais (2000 na demo) valem uma
hora simulada a partir de `SPYNO_DEMO_CLOCK_START`. O "agora" do agendador, da
conclusão de rotina e do fechamento de janelas passa a ser esse relógio. Antes da
âncora (`SPYNO_DEMO_CLOCK_ANCHOR_MS`), o relógio fica parado: a API termina de
subir antes do primeiro tick. O roteiro usa a mesma conta para saber a hora
simulada.

## Roteiro

1. Build e subida de `db` e `decision`; `migrate` aplica as migrations do backend
   num banco vazio.
2. A API sobe com o agendador desligado. Seis usuários são criados pela API
   (signup, signin, treino e rotinas). Para alguns, o histórico de dias
   anteriores é inserido direto no banco, porque a API só marca conclusão no
   "agora".
3. Contrato: `GET /v1/contract` aberto e `/v1/*` com 401 sem Bearer ou com
   Bearer errado. Uma API extra sobe com `SPYNO_OUTCOME_WINDOW_MIN=120` e
   precisa recusar o agendador ("contrato divergente").
4. A API é recriada com o agendador e o relógio ligados. O dia vai de
   2026-10-21 00:00 até 2026-10-22 03:00, com 28 ticks. Os treinos são
   concluídos por `PUT /routine/:id` na hora simulada marcada para cada persona,
   e há feedback de rotina para a valência. Em `--sem-shadow`, a entrega do push
   é simulada no banco, 2 minutos depois da decisão (o FCM do backend ainda é
   TODO).
5. A posterior do serviço é lida a cada 0,25 s em `GET /v1/policy-state`, que
   devolve o hash do snapshot, os updates pendentes e os dias fechados.
6. A API para e os dois bancos são exportados. Um `end-day` repetido é chamado
   direto no serviço.

## Saída (`demo/out/shadow/` ou `demo/out/sem-shadow/`, fora do git)

| arquivo | conteúdo |
|---|---|
| `notificacoes.csv` | uma linha por ponto de decisão: propensity (executada e recomendada), π por braço, ações recomendada, proposta e executada, motivo do bloqueio, `base_contadores`, `eligible`, contadores reais e `*_proposed`, janela, desfecho, `censurada`, feedback enviado e o que o serviço gravou (`outcome`, `trained`, motivo) |
| `notificacoes.json`, `servico_decisoes.json`, `servico_dias.json` | as tabelas completas dos dois lados |
| `treinos.json`, `usuarios.json`, `posterior.json` | treinos simulados, personas e a série do hash da posterior |
| `resumo.json`, `api.log`, `decision.log` | resumo numérico, as verificações (item, ok, detalhe) e os logs |

## O que é conferido

- Bearer: o serviço só devolveu 401 para as três chamadas negativas do
  roteiro; nenhuma chamada do backend foi recusada.
- `user_key = HMAC-SHA256(id, segredo)`, sem nenhum id cru no serviço.
- `decision_id`: o mesmo conjunto nos dois bancos, recalculado como
  `sha256(user_key|hora UTC|policy_version)`, com `decidida_em` igual ao
  `decision_point`.
- Um ponto por usuário a cada hora, sem hora pulada.
- Feedback de toda decisão fechada (braço 0 incluído, e braços de envio fora do
  shadow), com o desfecho igual nos dois lados. Só ficam abertas as janelas
  que ainda não venceram. `nao_respondida` tem `observado_em` igual ao
  fechamento da janela.
- `end-day`: chamado uma vez na virada. Repetido, devolve `already_closed` e
  não muda a posterior.
- A posterior muda no `end-day` e só nele. O número de updates aplicados é
  igual ao de desfechos que treinaram até a virada.
- Shadow: nada é executado e `counter_basis` é `proposed` em todas as decisões.
  O contexto gravado é igual ao recebido pelo serviço. Contadores e `eligible`,
  recalculados a partir do próprio log, batem com o que foi enviado.
- Ação proposta: o `acao_proposta` do backend é igual ao `proposed_action` do
  serviço. Recomendação bloqueada por guardrail tem proposta 0; sem guardrail, a
  proposta é igual à recomendada.
- Nenhuma propensity zero: executada, recomendada e por braço nos pontos
  elegíveis. Nos inelegíveis, π = [1, 0, 0, 0] e a executada tem propensity 1.
  P(enviar) fica dentro do clipping. Nenhuma decisão caiu na contingência.
