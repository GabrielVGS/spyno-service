# Proveniência

Este pacote foi copiado de `spyno-decision`, o repositório que produziu os
resultados de avaliação, ablação, sensibilidade e avaliação off-policy do
trabalho. Está aqui **sem alteração**, para que o serviço tenha uma imagem
autocontida e para que as políticas servidas sejam literalmente as mesmas que
foram avaliadas.

Duas consequências a respeitar:

- Mudanças de comportamento nas políticas não devem ser feitas aqui. O que o
  serviço precisa acrescentar — propensão do TS híbrido, reposição das chaves
  herdadas — vive em `spyno_service/policies/`, por composição.
- `service.py` não veio junto: era o serviço HTTP v0, substituído pelo
  `spyno_service`.

Origem: `spyno-decision` @ commit 03d5031 (2026-07-27).
