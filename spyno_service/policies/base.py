"""A interface que toda política implementa.

A assinatura é deliberadamente a mesma já usada pelas políticas do
`spyno-decision` e pelo laço do simulador, para que elas sejam plugadas sem
alteração. Uma política nova só precisa cumprir este protocolo.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

N_ARMS = 4


@runtime_checkable
class Policy(Protocol):
    name: str

    def start_user(self, user_id: str) -> None:
        """Chamado na primeira decisão de um usuário (cold start)."""

    def decide(self, user_id: str, obs: dict, eligible: bool, rng=None) -> tuple[int, float]:
        """Devolve (braço, propensão). Propensão 1.0 quando determinística."""

    def update(self, user_id: str, obs: dict, arm: int, reward: int) -> None:
        """Incorpora um desfecho observado."""

    def end_day(self) -> None:
        """Fecha o lote do dia (retreino em batch)."""


def propensities(policy: Policy, obs: dict, executed_arm: int, n_samples: int) -> tuple[float, str]:
    """Propensão da ação executada, com o método usado.

    Três casos, em ordem de preferência:

    1. A política sabe estimar a distribuição sobre braços (`estimate_propensities`).
       É o caso do Thompson Sampling, cuja propensão não tem forma fechada, e do
       PPO, que expõe a distribuição categórica da política.
    2. A política é determinística ou devolve a própria propensão em `decide`.
    3. Nada disso: registra-se 1.0 e o método `fallback`, o que sinaliza na
       análise posterior que aquele ponto não serve para avaliação off-policy.
    """
    estimator = getattr(policy, "estimate_propensities", None)
    if callable(estimator):
        try:
            dist = np.asarray(estimator(obs, n_samples), dtype=float)
            p = float(dist[executed_arm])
            if p > 0.0:
                return p, "monte-carlo"
            # Braço com probabilidade nula sob a política: sem sobreposição.
            # Registra-se um piso para não perder o ponto, marcado como fallback.
            return 1.0 / (n_samples + 1), "fallback"
        except Exception:
            pass
    return 1.0, "exact"
