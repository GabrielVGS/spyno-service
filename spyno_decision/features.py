"""Vetor de features da política — SOMENTE sinais observáveis em produção.

A ordem é o contrato do modelo: mudar a lista muda a dimensão e invalida
posteriors salvas (versionar via policy_version).
"""

from __future__ import annotations

import numpy as np

# chaves do dict `obs` (spyno_sim.response.obs_features + fbm_obs)
FEATURES: tuple[str, ...] = (
    "in_free_window",
    "valence_pos",
    "location_home",
    "motion_stationary",
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
    x[0] = 1.0  # bias
    for i, k in enumerate(FEATURES, start=1):
        x[i] = float(obs[k])
    return x
