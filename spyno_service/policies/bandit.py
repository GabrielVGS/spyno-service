"""Políticas do `spyno-decision` servidas sem alteração.

O pacote `spyno_decision` é a fonte única das políticas validadas: foi ele que
produziu os resultados de avaliação, ablação, sensibilidade e avaliação
off-policy. Este módulo não reimplementa o aprendizado; constrói as instâncias,
converte o contexto do contrato para o espaço da política, acrescenta a
distribuição sobre braços (para o clipping) e, quando necessário, carrega a
posterior salva.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import CONFIG
from ..features import FEATURES, POLICY_SCHEMA, policy_obs
from .clipping import ClipConfig, ClippedPolicy


#: O `spyno_decision.features.vectorize` ainda espera estas chaves. Elas saíram
#: do vetor do serviço; o adaptador as repõe com zero, que é exatamente a
#: condição "removido" medida nas ablações (sensores e `fbm_obs`). Coluna
#: constante em zero não move a posterior do peso correspondente.
_CHAVES_HERDADAS = ("location_home", "motion_stationary", "fbm_obs")


class IncompatibleSnapshotError(ValueError):
    """Posterior salva sob outro esquema de política."""


class _CompatibilidadeHerdada:
    """Converte o contexto do contrato no `obs` que as políticas herdadas esperam.

    Aplica `features.policy_obs` (notifs_today ÷ 5, sem fbm_obs) e repõe com zero
    as chaves que o vetor do `spyno_decision` ainda exige. Toda entrada na
    política (decide, update, distribuição) passa por aqui, então o contexto
    bruto pode ir para o log e voltar no feedback sem dupla normalização.
    """

    def __init__(self, inner):
        self._inner = inner
        self.name = inner.name

    def _obs(self, obs: dict) -> dict:
        return {**policy_obs(obs), **{k: 0.0 for k in _CHAVES_HERDADAS}}

    def start_user(self, user_id):
        return self._inner.start_user(user_id)

    def decide(self, user_id, obs, eligible, rng=None):
        return self._inner.decide(user_id, self._obs(obs), eligible, rng)

    def update(self, user_id, obs, arm, reward):
        return self._inner.update(user_id, self._obs(obs), arm, reward)

    def end_day(self):
        return self._inner.end_day()

    def arm_wins(self, obs: dict, n_samples: int):
        return self._inner.arm_wins(self._obs(obs), n_samples)

    def estimate_propensities(self, obs: dict, n_samples: int = 64):
        return self._inner.estimate_propensities(self._obs(obs), n_samples)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _missing(name: str) -> None:
    raise RuntimeError(f"a política '{name}' depende do pacote spyno_decision")


def _snapshot_meta() -> dict:
    return {"schema": POLICY_SCHEMA, "features": list(FEATURES)}


def _check_snapshot(snap: dict) -> None:
    schema = snap.get("schema", "v1")
    if schema != POLICY_SCHEMA:
        raise IncompatibleSnapshotError(
            f"posterior salva no esquema '{schema}', mas o serviço serve o esquema "
            f"'{POLICY_SCHEMA}'. Entre os dois mudaram a escala de notifs_today (÷5) "
            "e a remoção de fbm_obs: os pesos não são reaproveitáveis sem conversão. "
            "Recomece do prior (ou do warm-start) sob o esquema atual."
        )


def _servable_classes():
    """Acrescenta às políticas do `spyno-decision` o que falta para servi-las.

    As classes originais implementam a interface do laço do simulador, onde o
    arrependimento é exato e a propensão não é necessária. Em produção ela é, e
    o clipping precisa da distribuição sobre braços. Falta também persistir a
    posterior com o esquema, para que uma posterior antiga seja recusada de
    forma explícita.

    A extensão fica aqui, e não no pacote original, para não alterar o código
    que produziu os resultados já publicados.
    """
    from spyno_decision.hybrid import HDIM, HybridLogisticTS, hybrid_x
    from spyno_decision.features import vectorize
    from spyno_decision.policy import (N_ARMS, BinaryLogisticTS,
                                       DisjointLogisticTS, _sigmoid)

    import numpy as np

    cost_vec = lambda c: c * np.array([0.0, 1.0, 1.0, 1.0])  # noqa: E731

    class _Snapshot:
        def snapshot(self) -> dict:
            return {"m": np.asarray(self.m).tolist(), "q": np.asarray(self.q).tolist(),
                    "cost": self.cost, **_snapshot_meta()}

        def load_snapshot(self, snap: dict) -> None:
            _check_snapshot(snap)
            m, q = np.array(snap["m"]), np.array(snap["q"])
            if m.shape != np.shape(self.m) or q.shape != np.shape(self.q):
                raise IncompatibleSnapshotError(
                    f"dimensão da posterior {m.shape} difere da política {np.shape(self.m)}")
            self.m, self.q, self.cost = m, q, snap["cost"]

        def estimate_propensities(self, obs: dict, n_samples: int = 64):
            """π do TS *sem* clipping. Diagnóstico; o serviço usa ClippedPolicy."""
            wins = self.arm_wins(obs, n_samples)
            return wins / wins.sum()

    class HybridServableTS(_Snapshot, HybridLogisticTS):
        send_arms = (1, 2, 3)

        def arm_wins(self, obs: dict, n_samples: int) -> np.ndarray:
            """Vitórias de cada braço em n amostras da posterior (o sorteio de `decide`)."""
            x = vectorize(obs)
            H = np.stack([hybrid_x(x, a) for a in range(N_ARMS)])        # (4, HDIM)
            W = self.m + self.rng.standard_normal((n_samples, HDIM)) / np.sqrt(self.q)
            util = _sigmoid(W @ H.T) - cost_vec(self.cost)                # (n, 4)
            return np.bincount(np.argmax(util, axis=1), minlength=N_ARMS).astype(float)

    class DisjointServableTS(_Snapshot, DisjointLogisticTS):
        send_arms = (1, 2, 3)

        def arm_wins(self, obs: dict, n_samples: int) -> np.ndarray:
            x = self._vec(obs)
            W = self.m + self.rng.standard_normal((n_samples, *self.m.shape)) / np.sqrt(self.q)
            util = _sigmoid(W @ x) - cost_vec(self.cost)                  # (n, 4)
            return np.bincount(np.argmax(util, axis=1), minlength=N_ARMS).astype(float)

    class BinaryServableTS(_Snapshot, BinaryLogisticTS):
        send_arms = (BinaryLogisticTS._NOTIFY,)

        def arm_wins(self, obs: dict, n_samples: int) -> np.ndarray:
            x = self._vec(obs)
            arms = [0, self._NOTIFY]
            W = (self.m[arms] + self.rng.standard_normal((n_samples, 2, self.m.shape[1]))
                 / np.sqrt(self.q[arms]))
            util = _sigmoid(W @ x) - self.cost * np.array([0.0, 1.0])
            win = np.argmax(util, axis=1)
            out = np.zeros(N_ARMS)
            out[0] = float(np.sum(win == 0))
            out[self._NOTIFY] = float(np.sum(win == 1))
            return out

    return HybridServableTS, DisjointServableTS, BinaryServableTS


def _serve(cls, name, cost, seed, snapshot, clip, n_samples):
    policy = cls(cost=cost, seed=seed)
    _restore(policy, snapshot)
    clip = clip or ClipConfig.from_config(CONFIG)
    n = n_samples or CONFIG.propensity_samples
    served = ClippedPolicy(_CompatibilidadeHerdada(policy), clip, n, seed=seed + 1)
    # O `name` das classes herdadas é "ts" tanto na binária quanto na disjoint;
    # o nome da fábrica é o que entra no prefixo da policy_version.
    served.name = name
    return served


def ts_hybrid(cost: float = 0.05, seed: int = 0, snapshot: str | Path | None = None,
              clip: ClipConfig | None = None, n_samples: int | None = None):
    """Thompson Sampling com pooling parcial: o modelo recomendado pela avaliação."""
    try:
        cls = _servable_classes()[0]
    except ImportError:
        _missing("ts_hybrid")
    return _serve(cls, "ts_hybrid", cost, seed, snapshot, clip, n_samples)


def ts_disjoint(cost: float = 0.05, seed: int = 0, snapshot: str | Path | None = None,
                clip: ClipConfig | None = None, n_samples: int | None = None):
    """Thompson Sampling com posterior independente por braço."""
    try:
        cls = _servable_classes()[1]
    except ImportError:
        _missing("ts_disjoint")
    return _serve(cls, "ts_disjoint", cost, seed, snapshot, clip, n_samples)


def ts_binary(cost: float = 0.05, seed: int = 0, snapshot: str | Path | None = None,
              clip: ClipConfig | None = None, n_samples: int | None = None):
    """Thompson Sampling sobre a decisão binária de notificar."""
    try:
        cls = _servable_classes()[2]
    except ImportError:
        _missing("ts_binary")
    return _serve(cls, "ts_binary", cost, seed, snapshot, clip, n_samples)


def never(**_):
    """Referência: nunca notificar. Sem clipping: é a referência de abstenção."""
    try:
        from spyno_decision.baselines import NeverNotify
    except ImportError:
        _missing("never")
    return _CompatibilidadeHerdada(NeverNotify())


def always(**_):
    """Referência: notificar sempre que elegível, braço uniforme. Sem clipping."""
    try:
        from spyno_decision.baselines import AlwaysNotifyRandomArm
    except ImportError:
        _missing("always")
    return _CompatibilidadeHerdada(AlwaysNotifyRandomArm())


def fbm_threshold(threshold: float = 40.0, **_):
    """Referência: a heurística de limiar do trabalho anterior.

    Atenção: a implementação original lê `obs["fbm_full"]`, o escore FBM
    completo, que incluía sono e arousal. Esses sinais não são coletados na
    formulação atual, então esta política só é executável dentro do simulador.
    Em produção ela cai na contingência, o que fica registrado. A versão
    servível abaixo usa o escore observável e é, por isso, uma heurística
    diferente da avaliada nos experimentos.
    """
    try:
        from spyno_decision.baselines import FBMHeuristic
    except ImportError:
        _missing("fbm_threshold")
    return _CompatibilidadeHerdada(FBMHeuristic(threshold=threshold))


def fbm_threshold_observable(threshold: float = 0.4, **_):
    """Limiar sobre o escore FBM *observável*, executável em produção.

    Lê `fbm_obs` direto do contexto do contrato. Por isso não passa pelo
    adaptador: no esquema v2 `fbm_obs` saiu do vetor das políticas aprendidas,
    mas continua chegando no contrato justamente para esta referência e o log.
    """

    class ObservableFBM:
        name = "fbm_obs_threshold"

        def start_user(self, user_id): pass
        def update(self, user_id, obs, arm, reward): pass
        def end_day(self): pass

        def decide(self, user_id, obs, eligible, rng=None):
            if eligible and float(obs["fbm_obs"]) > threshold:
                return 3, 1.0
            return 0, 1.0

    return ObservableFBM()


def _restore(policy, snapshot: str | Path | None) -> None:
    if snapshot is None:
        return
    path = Path(snapshot)
    if not path.exists():
        raise FileNotFoundError(f"posterior não encontrada em {path}")
    policy.load_snapshot(json.loads(path.read_text()))
