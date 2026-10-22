"""Vetor observável comum a todas as políticas.

A ordem desta tupla é contrato do modelo: alterá-la muda a dimensão e invalida
as posteriors salvas. Qualquer mudança aqui exige subir `POLICY_SCHEMA`.

Dois espaços de observação convivem e não podem ser confundidos:

- o **contexto do contrato** (`contracts.Context`), como o backend envia:
  `notifs_today` é a contagem bruta de envios do dia, porque o guardrail da
  cota diária compara com `SPYNO_DAILY_CAP`; `fbm_obs` continua chegando, para
  log e para a política de referência `fbm_obs_threshold`;
- o **espaço da política**, o que o simulador entregava às políticas
  (`spyno_sim.response.obs_features`): `notifs_today / 5`.

`policy_obs` converte o primeiro no segundo e `vectorize` sempre parte do
contexto do contrato. `vectorize_policy_space` existe só para quem já tem a
observação do simulador em mãos (o ambiente de treino do PPO).

Histórico do esquema
--------------------
- v1: 13 atributos, incluindo `fbm_obs`; `notifs_today` bruto (divergia do
  simulador, que dividia por 5); propensão por Monte Carlo, podendo sair 0.
- v2: `fbm_obs` sai do vetor (a ablação mostrou que ele não acrescenta e a
  fórmula do backend diverge da do simulador); `notifs_today` ÷ 5; clipping da
  probabilidade de envio e propensão exata da distribuição amostrada.

`location_home` e `motion_stationary` foram removidos antes da v1. Eles
dependiam de sensores que o backend não coleta e, medido em sementes pareadas,
zerá-los não altera treinos nem notificações de forma distinguível de ruído. As
políticas herdadas do `spyno-decision` ainda esperam as duas chaves (e
`fbm_obs`), e o adaptador em `policies/bandit.py` as repõe com zero.
"""

from __future__ import annotations

import numpy as np

#: Versão do esquema da política: features, normalização, clipping e propensão.
#: Entra na `policy_version` e nos snapshots; posterior de outro esquema é recusada.
POLICY_SCHEMA = "v2"

#: Divisor de `notifs_today`, o mesmo de `spyno_sim.response.obs_features`.
NOTIFS_TODAY_SCALE = 5.0

FEATURES: tuple[str, ...] = (
    "in_free_window",
    "valence_pos",
    "is_weekend",
    "recency",
    "already_trained",
    "streak_norm",
    "emp_adhesion",
    "fatigue",
    "notifs_today",
    "morning",
    "midday",
    "evening",
)
DIM = len(FEATURES) + 1  # + bias

#: Atributos que o contrato aceita mas a política não usa. Ficam no log.
DROPPED_FROM_POLICY: tuple[str, ...] = ("fbm_obs",)


def policy_obs(context: dict) -> dict:
    """Contexto do contrato → observação no espaço da política (v2).

    Não altera o dicionário recebido: o contexto bruto é o que vai para o log.
    `fbm_obs` é removido; quem precisa da chave (as políticas herdadas) a repõe
    com zero, o que equivale a `zero_features=("fbm_obs",)`.
    """
    out = {k: v for k, v in context.items() if k not in DROPPED_FROM_POLICY}
    out["notifs_today"] = float(context["notifs_today"]) / NOTIFS_TODAY_SCALE
    return out


def vectorize_policy_space(obs: dict) -> np.ndarray:
    """Vetor a partir de uma observação JÁ no espaço da política (simulador)."""
    x = np.empty(DIM)
    x[0] = 1.0
    for i, k in enumerate(FEATURES, start=1):
        x[i] = float(obs[k])
    return x


def vectorize(context: dict) -> np.ndarray:
    """Vetor a partir do contexto do contrato (notifs_today bruto)."""
    return vectorize_policy_space(policy_obs(context))
