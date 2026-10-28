"""Thompson Sampling disjoint com regressão logística bayesiana (Laplace diagonal).

Um modelo por braço (Li et al., WWW'10) sobre o mesmo vetor de contexto —
equivale a interações braço×contexto completas, condição necessária para
aprender o pareamento tipo-de-gatilho ↔ déficit (RQ3).

Atualização: aproximação de Laplace com precisão diagonal, em batch diário
(Chapelle & Li, NeurIPS'11): a posterior de hoje vira o prior de amanhã.
Decisão: amostra w_a ~ N(m_a, diag(1/q_a)); escolhe argmax de
sigmoid(w_a·x) − custo·1[a>0] entre os braços elegíveis.
"""

from __future__ import annotations

import numpy as np

from .features import DIM, vectorize

N_ARMS = 4
ARM_NAMES = ("no_notify", "spark", "facilitator", "signal")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _nll(X, y, w, m0, q0) -> float:
    z = X @ w
    # log-loss numérico-estável + prior gaussiano
    ll = np.sum(np.logaddexp(0.0, z) - y * z)
    return float(ll + 0.5 * np.sum(q0 * (w - m0) ** 2))


def fit_laplace(X: np.ndarray, y: np.ndarray, m0: np.ndarray, q0: np.ndarray,
                iters: int = 50, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """MAP por IRLS (Newton com Hessiana COMPLETA + backtracking) sob prior
    gaussiano N(m0, diag(1/q0)); posterior aproximada com precisão diagonal
    q = q0 + diag(Xᵀ S X) — aproximação de Laplace diagonal no MAP.

    Newton com Hessiana só-diagonal diverge aqui: as features são correlacionadas
    (bias, dummies de período, janela livre) e o passo sem os termos cruzados
    superestima a curvatura — foi exatamente o bug do warm-start degenerar.
    """
    w = m0.copy()
    loss = _nll(X, y, w, m0, q0)
    for _ in range(iters):
        p = _sigmoid(X @ w)
        grad = X.T @ (p - y) + q0 * (w - m0)
        s = p * (1.0 - p)
        hess = (X.T * s) @ X + np.diag(q0)
        step = np.linalg.solve(hess, grad)
        # backtracking: garante que o passo reduz a perda
        t = 1.0
        for _ in range(20):
            w_new = w - t * step
            loss_new = _nll(X, y, w_new, m0, q0)
            if loss_new <= loss:
                break
            t *= 0.5
        if float(np.max(np.abs(t * step))) < tol:
            w = w_new
            break
        w, loss = w_new, loss_new
    p = _sigmoid(X @ w)
    q = q0 + (X ** 2).T @ (p * (1.0 - p))
    return w, q


class DisjointLogisticTS:
    """Política TS de K+1 braços com retreino em batch diário."""

    def __init__(self, cost: float = 0.01, prior_precision: float = 1.0,
                 seed: int = 0, name: str = "ts",
                 zero_features: tuple[str, ...] = ()):
        from .features import FEATURES
        self.name = name
        self.cost = cost
        self.m = np.zeros((N_ARMS, DIM))
        self.q = np.full((N_ARMS, DIM), prior_precision)
        self.rng = np.random.default_rng(seed)          # rng próprio: não acopla
        self._day_buf: list[list[tuple[np.ndarray, int]]] = [[] for _ in range(N_ARMS)]
        # ablation: zera colunas de features (ex. "fatigue") mantendo a dimensão
        self._zero_idx = [1 + FEATURES.index(f) for f in zero_features]

    def _vec(self, obs: dict) -> np.ndarray:
        x = vectorize(obs)
        for i in self._zero_idx:
            x[i] = 0.0
        return x

    # ---- warm-start (RQ2) ------------------------------------------------
    def warm_start(self, rows: list[dict]) -> dict:
        """Ajusta a posterior de cada braço a partir de linhas do bandit_log
        (política de log randomizada → outcome model não-enviesado por braço)."""
        counts = {}
        for arm in range(N_ARMS):
            X, y = [], []
            for r in rows:
                if int(r["action"]) == arm and int(r["eligible"]) == 1:
                    X.append(self._vec(r))
                    y.append(int(r["action_performed"]))
            counts[ARM_NAMES[arm]] = len(y)
            if len(y) >= 10:
                self.m[arm], self.q[arm] = fit_laplace(
                    np.array(X), np.array(y, dtype=float), self.m[arm], self.q[arm])
        return counts

    # ---- interface Policy (spyno_sim.simulate) ---------------------------
    def start_user(self, user_id: str) -> None:
        pass

    def decide(self, user_id: str, obs: dict, eligible: bool, rng=None) -> tuple[int, float]:
        if not eligible:
            return 0, 1.0
        x = self._vec(obs)
        w = self.m + self.rng.standard_normal((N_ARMS, DIM)) / np.sqrt(self.q)
        p_hat = _sigmoid(w @ x)
        utility = p_hat - self.cost * np.array([0.0, 1.0, 1.0, 1.0])
        return int(np.argmax(utility)), 1.0  # propensity via estimate_propensities()

    def update(self, user_id: str, obs: dict, arm: int, reward: int) -> None:
        self._day_buf[arm].append((self._vec(obs), reward))

    def end_day(self) -> None:
        """Retreino diário: posterior de hoje vira prior de amanhã (Bayes streaming)."""
        for arm in range(N_ARMS):
            buf = self._day_buf[arm]
            if buf:
                X = np.array([x for x, _ in buf])
                y = np.array([r for _, r in buf], dtype=float)
                self.m[arm], self.q[arm] = fit_laplace(X, y, self.m[arm], self.q[arm])
                self._day_buf[arm] = []

    # ---- produção --------------------------------------------------------
    def estimate_propensities(self, obs: dict, n_samples: int = 64) -> np.ndarray:
        """P(escolher cada braço) por Monte Carlo — exigência do propensity
        logging para OPE em produção (não usada na avaliação por regret exato)."""
        x = self._vec(obs)
        wins = np.zeros(N_ARMS)
        for _ in range(n_samples):
            w = self.m + self.rng.standard_normal((N_ARMS, DIM)) / np.sqrt(self.q)
            p_hat = _sigmoid(w @ x)
            utility = p_hat - self.cost * np.array([0.0, 1.0, 1.0, 1.0])
            wins[int(np.argmax(utility))] += 1
        return wins / n_samples

    def snapshot(self) -> dict:
        return {"m": self.m.tolist(), "q": self.q.tolist(),
                "cost": self.cost, "arms": list(ARM_NAMES)}

    def load_snapshot(self, snap: dict) -> None:
        self.m = np.array(snap["m"])
        self.q = np.array(snap["q"])
        self.cost = snap["cost"]


class BinaryLogisticTS(DisjointLogisticTS):
    """Ablation (RQ3): ação BINÁRIA {não notificar, notificar} com um único tipo
    de mensagem (signal), como o sistema legado. Warm-start agrega os dados dos
    três braços de notificação num modelo único — mede quanto vale a dimensão
    "que tipo de gatilho" em separado da dimensão "quando".

    Internamente usa os slots 0 (quieto) e 3 (signal) das mesmas estruturas.
    """

    _NOTIFY = 3

    def warm_start(self, rows: list[dict]) -> dict:
        collapsed = [{**r, "action": 0 if int(r["action"]) == 0 else self._NOTIFY}
                     for r in rows]
        return super().warm_start(collapsed)

    def decide(self, user_id: str, obs: dict, eligible: bool, rng=None) -> tuple[int, float]:
        if not eligible:
            return 0, 1.0
        x = self._vec(obs)
        arms = (0, self._NOTIFY)
        w = self.m[list(arms)] + self.rng.standard_normal((2, DIM)) / np.sqrt(self.q[list(arms)])
        p_hat = _sigmoid(w @ x)
        utility = p_hat - self.cost * np.array([0.0, 1.0])
        return arms[int(np.argmax(utility))], 1.0

    def update(self, user_id: str, obs: dict, arm: int, reward: int) -> None:
        super().update(user_id, obs, 0 if arm == 0 else self._NOTIFY, reward)
