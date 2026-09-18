"""TS híbrido com pooling parcial (F2.5b) — resposta ao achado binário>K+1.

Estrutura do preditor (análoga ao LinUCB híbrido de Li et al., WWW'10, e à
própria estrutura do mundo: efeito-base + efeito comum de notificar + efeito
específico do tipo):

    z(x, a) = w_base·x  +  1[a>0]·(w_send·x)  +  1[a=k]·(v_k·x),  k∈{1,2,3}

Um ÚNICO vetor de parâmetros [w_base | w_send | v_1 | v_2 | v_3] (dim 5D):
- toda observação (qualquer braço) atualiza w_base → eficiência do binário;
- toda notificação atualiza w_send → o "quando" é compartilhado;
- só o braço k atualiza v_k, com prior mais apertado (shrinkage λ_pool) →
  o "o quê" é aprendido na medida em que os dados sustentam.

Com poucos dados, v_k ≈ 0 e a política se comporta como o binário; com muitos,
os v_k se diferenciam e ela recupera a expressividade do disjoint — a curva de
3 regimes que transforma o resultado negativo da F2 em achado.
"""

from __future__ import annotations

import numpy as np

from .features import DIM, vectorize
from .policy import N_ARMS, _sigmoid, fit_laplace

HDIM = 5 * DIM  # [base | send | v1 | v2 | v3]


def hybrid_x(x: np.ndarray, arm: int) -> np.ndarray:
    h = np.zeros(HDIM)
    h[:DIM] = x
    if arm > 0:
        h[DIM:2 * DIM] = x
        k = 2 + (arm - 1)          # bloco do braço: 2,3,4
        h[k * DIM:(k + 1) * DIM] = x
    return h


class HybridLogisticTS:
    """Mesma interface Policy do simulador; um modelo único com blocos."""

    def __init__(self, cost: float = 0.05, seed: int = 0, name: str = "ts_hybrid",
                 prior_base: float = 1.0, prior_send: float = 1.0,
                 prior_arm: float = 4.0):
        self.name = name
        self.cost = cost
        self.rng = np.random.default_rng(seed)
        self.m = np.zeros(HDIM)
        q0 = np.empty(HDIM)
        q0[:DIM] = prior_base
        q0[DIM:2 * DIM] = prior_send
        q0[2 * DIM:] = prior_arm    # shrinkage dos offsets por braço
        self.q = q0
        self._day_buf: list[tuple[np.ndarray, int]] = []

    # ---- warm-start ------------------------------------------------------
    def warm_start(self, rows: list[dict]) -> int:
        X, y = [], []
        for r in rows:
            if int(r["eligible"]) == 1:
                X.append(hybrid_x(vectorize(r), int(r["action"])))
                y.append(int(r["action_performed"]))
        if y:
            self.m, self.q = fit_laplace(np.array(X), np.array(y, dtype=float),
                                         self.m, self.q)
        return len(y)

    # ---- interface Policy ------------------------------------------------
    def start_user(self, user_id: str) -> None:
        pass

    def decide(self, user_id: str, obs: dict, eligible: bool, rng=None) -> tuple[int, float]:
        if not eligible:
            return 0, 1.0
        x = vectorize(obs)
        w = self.m + self.rng.standard_normal(HDIM) / np.sqrt(self.q)
        utility = np.empty(N_ARMS)
        for a in range(N_ARMS):
            utility[a] = _sigmoid(float(w @ hybrid_x(x, a))) - self.cost * (a > 0)
        return int(np.argmax(utility)), 1.0

    def update(self, user_id: str, obs: dict, arm: int, reward: int) -> None:
        self._day_buf.append((hybrid_x(vectorize(obs), arm), reward))

    def end_day(self) -> None:
        if self._day_buf:
            X = np.array([x for x, _ in self._day_buf])
            y = np.array([r for _, r in self._day_buf], dtype=float)
            self.m, self.q = fit_laplace(X, y, self.m, self.q)
            self._day_buf = []
