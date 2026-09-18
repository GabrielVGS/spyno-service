"""Vetor observável comum a todas as políticas.

A ordem desta tupla é contrato do modelo: alterá-la muda a dimensão e invalida
as posteriors salvas. Qualquer mudança aqui exige subir a `policy_version`.

`location_home` e `motion_stationary` foram removidos. Eles dependiam de sensores
que o backend não coleta e, medido em sementes pareadas, zerá-los não altera
treinos nem notificações de forma distinguível de ruído: no simulador a função
que gera a resposta não os usa. As políticas herdadas do `spyno-decision` ainda
esperam as duas chaves, e o adaptador em `policies/bandit.py` as repõe com zero.
"""

from __future__ import annotations

import numpy as np

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
    "fbm_obs",
)
DIM = len(FEATURES) + 1  # + bias


def vectorize(obs: dict) -> np.ndarray:
    x = np.empty(DIM)
    x[0] = 1.0
    for i, k in enumerate(FEATURES, start=1):
        x[i] = float(obs[k])
    return x
