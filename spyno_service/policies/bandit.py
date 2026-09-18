"""Políticas do `spyno-decision` servidas sem alteração.

O pacote `spyno_decision` é a fonte única das políticas validadas: foi ele que
produziu os resultados de avaliação, ablação, sensibilidade e avaliação
off-policy. Este módulo não reimplementa nada; apenas constrói as instâncias e,
quando necessário, carrega a posterior salva.
"""

from __future__ import annotations

import json
from pathlib import Path


#: O `spyno_decision.features.vectorize` ainda espera estas chaves. Elas saíram
#: do contrato do serviço; o adaptador as repõe com zero, que é exatamente a
#: condição "removido" medida na ablação de sensores.
_CHAVES_HERDADAS = ("location_home", "motion_stationary")


class _CompatibilidadeHerdada:
    """Repõe as chaves que as políticas do spyno-decision ainda exigem."""

    def __init__(self, inner):
        self._inner = inner
        self.name = inner.name

    def _obs(self, obs: dict) -> dict:
        return {**{k: 0.0 for k in _CHAVES_HERDADAS}, **obs}

    def start_user(self, user_id):
        return self._inner.start_user(user_id)

    def decide(self, user_id, obs, eligible, rng=None):
        return self._inner.decide(user_id, self._obs(obs), eligible, rng)

    def update(self, user_id, obs, arm, reward):
        return self._inner.update(user_id, self._obs(obs), arm, reward)

    def end_day(self):
        return self._inner.end_day()

    def estimate_propensities(self, obs: dict, n_samples: int = 64):
        return self._inner.estimate_propensities(self._obs(obs), n_samples)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _missing(name: str) -> None:
    raise RuntimeError(f"a política '{name}' depende do pacote spyno_decision")


def _hybrid_servable_class():
    """Acrescenta ao TS híbrido o que falta para ele ser servido.

    O `HybridLogisticTS` do `spyno-decision` implementa a interface exigida pelo
    laço do simulador, onde o arrependimento é exato e a propensão não é
    necessária. Em produção ela é: sem registrar a probabilidade com que a ação
    foi escolhida, os registros não servem para avaliação off-policy, que é
    metade do protocolo do trabalho. Falta também persistir a posterior, sem o
    que cada reinício do serviço joga o aprendizado fora.

    A extensão fica aqui, e não no pacote original, para não alterar o código
    que produziu os resultados já publicados.
    """
    from spyno_decision.features import vectorize
    from spyno_decision.hybrid import HDIM, HybridLogisticTS, hybrid_x
    from spyno_decision.policy import N_ARMS, _sigmoid

    import numpy as np

    class HybridServableTS(HybridLogisticTS):
        def estimate_propensities(self, obs: dict, n_samples: int = 64) -> np.ndarray:
            """P(escolher cada braço) por Monte Carlo sobre a posterior.

            Reproduz exatamente o sorteio de `decide`, contando o braço vencedor.
            """
            x = vectorize(obs)
            wins = np.zeros(N_ARMS)
            for _ in range(n_samples):
                w = self.m + self.rng.standard_normal(HDIM) / np.sqrt(self.q)
                utility = np.empty(N_ARMS)
                for a in range(N_ARMS):
                    utility[a] = _sigmoid(float(w @ hybrid_x(x, a))) - self.cost * (a > 0)
                wins[int(np.argmax(utility))] += 1
            return wins / n_samples

        def snapshot(self) -> dict:
            return {"m": self.m.tolist(), "q": self.q.tolist(), "cost": self.cost}

        def load_snapshot(self, snap: dict) -> None:
            self.m = np.array(snap["m"])
            self.q = np.array(snap["q"])
            self.cost = snap["cost"]

    return HybridServableTS


def ts_hybrid(cost: float = 0.05, seed: int = 0, snapshot: str | Path | None = None):
    """Thompson Sampling com pooling parcial: o modelo recomendado pela avaliação."""
    try:
        cls = _hybrid_servable_class()
    except ImportError:
        _missing("ts_hybrid")
    policy = cls(cost=cost, seed=seed)
    _restore(policy, snapshot)
    return _CompatibilidadeHerdada(policy)


def ts_disjoint(cost: float = 0.05, seed: int = 0, snapshot: str | Path | None = None):
    """Thompson Sampling com posterior independente por braço."""
    try:
        from spyno_decision.policy import DisjointLogisticTS
    except ImportError:
        _missing("ts_disjoint")
    policy = DisjointLogisticTS(cost=cost, seed=seed)
    _restore(policy, snapshot)
    return _CompatibilidadeHerdada(policy)


def ts_binary(cost: float = 0.05, seed: int = 0, snapshot: str | Path | None = None):
    """Thompson Sampling sobre a decisão binária de notificar."""
    try:
        from spyno_decision.policy import BinaryLogisticTS
    except ImportError:
        _missing("ts_binary")
    policy = BinaryLogisticTS(cost=cost, seed=seed)
    _restore(policy, snapshot)
    return _CompatibilidadeHerdada(policy)


def never(**_):
    """Referência: nunca notificar."""
    try:
        from spyno_decision.baselines import NeverNotify
    except ImportError:
        _missing("never")
    return _CompatibilidadeHerdada(NeverNotify())


def always(**_):
    """Referência: notificar sempre que elegível, braço uniforme."""
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
    """Limiar sobre o escore FBM *observável*, executável em produção."""

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
