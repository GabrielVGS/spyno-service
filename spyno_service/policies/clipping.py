"""Probability clipping e propensão exata (esquema v2).

Por que
-------
O Thompson Sampling puro pode chegar a probabilidade de envio 0 ou 1 num
contexto. Isso destrói a sobreposição de que a avaliação off-policy depende e
deixa a política sem exploração. Os sistemas implantados limitam essa
probabilidade: [0,1; 0,8] em Liao et al. (2020), [0,2; 0,8] no Oralytics. Aqui o
padrão é [0,05; 0,9] (p05, desde 2026-10-26; antes [0,1; 0,9], p10), configurável
(`SPYNO_CLIP_MIN`, `SPYNO_CLIP_MAX`).

Antes (v1) a ação vinha de UMA amostra da posterior e a propensão era estimada
à parte, com outras 64 amostras: as duas não correspondiam ao mesmo sorteio e a
estimativa podia sair 0 para a ação efetivamente escolhida.

Esquema (v2), num ponto elegível
--------------------------------
1. **Distribuição do TS.** Com M amostras da posterior (`SPYNO_PROPENSITY_SAMPLES`),
   conta-se quantas vezes cada braço vence o argmax de sigmoid(w·x) − custo·1[a>0]:
   π_TS(a) = vitórias_a / M. É a política de Thompson Sampling, estimada.
2. **Clipping da massa de envio.** p_envio = Σ_{a∈S} π_TS(a), onde S são os
   braços de envio que a política usa ({1,2,3}; só {3} no `ts_binary`).
   p̃_envio = clip(p_envio, p_min, p_max).
3. **Repartição entre os braços de envio.** A massa p̃_envio é dividida na
   proporção das vitórias de cada braço de envio, misturada com uniforme:
       share_k = (1 − η)·vitórias_k / Σ_{j∈S} vitórias_j + η / |S|
   com η = `SPYNO_ARM_MIX` (padrão 0,1). Se nenhum braço de envio venceu, share
   é uniforme em S. Sozinho, este passo só garante p̃_envio·share_k ≥ p_min·η/|S|
   (≈ 0,0017 no padrão), daí o passo 4.
4. **Piso por braço de envio.** Cada braço de S recebe pelo menos f =
   `SPYNO_ARM_FLOOR` (padrão 0,05/3 ≈ 0,0167). Os braços que ficariam abaixo de f
   sobem para f; os demais repartem o resto da massa de envio na proporção de
   share_k (projeção por *water-filling*: os braços já acima do piso não mudam
   de ordem e, se nenhum estiver abaixo, nada muda). A massa total de envio
   continua p̃_envio. Exige |S|·f ≤ p_min, validado com |S| = 3.
5. **Abstenção.** π(0) = 1 − p̃_envio ≥ 1 − p_max > 0.
6. **Sorteio.** A ação é sorteada de π e a propensão registrada é π(ação).

Piso e peso de importância (v2.1)
---------------------------------
Sem o passo 4 (v2 original), o menor π(k) de um braço de envio era
p_min·η/|S| = 0,1·0,1/3 ≈ 0,0033: para uma política-alvo determinística que
escolhe um braço específico, o peso de importância π_alvo/π chegava a ~300 e a
OPE sobre o log ficava com variância enorme. Com o piso, o peso máximo para
qualquer alvo determinístico é 1/min(f, 1 − p_max) (ver
`max_importance_weight`). O padrão p05 (p_min = 0,05, f = p_min/3 = 0,05/3)
limita o peso a 60 sem aumentar a exploração mínima: P(enviar) ≥ 0,05 continua
sendo o único custo fixo. O padrão anterior p10 (`SPYNO_CLIP_MIN=0.1
SPYNO_ARM_FLOOR=0.03333333333333333`) limita o peso a 30; no simulador, ele
custou E[treinos] em parte dos mundos (ver o README, "Piso de envio"). Pesos ≤ 10
exigiriam f = 0,1 e, portanto, P(enviar) ≥ 0,3 em toda hora elegível
(`SPYNO_CLIP_MIN=0.3 SPYNO_ARM_FLOOR=0.1`). `SPYNO_ARM_FLOOR=0` reproduz a v2
sem piso. Um f arredondado para cima (0.016667 com p_min 0,05) é recusado:
exige-se 3·f ≤ p_min.

Por que a propensão é exata
---------------------------
A ação é literalmente sorteada de π, então π(ação) é a probabilidade com que
ela foi escolhida, sem segunda estimativa. O erro de Monte Carlo do passo 1 só
afeta *qual* π a política usa, não a validade do registro: as amostras da
posterior são independentes do desfecho, e o IPS condicional a elas continua
não enviesado. Nenhum braço de S nem a abstenção têm probabilidade nula.

Fora de ponto elegível nada é sorteado: ação 0 com probabilidade 1.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .base import N_ARMS

#: Maior |S| possível (todos os braços menos a abstenção). O piso é validado
#: contra ele, para valer para qualquer política.
MAX_SEND_ARMS = N_ARMS - 1
_EPS = 1e-12


@dataclass(frozen=True)
class ClipConfig:
    # Padrões = os do serviço (config.py): p05. Os experimentos do
    # spyno-decision usam p10 explícito.
    p_min: float = 0.05
    p_max: float = 0.9
    arm_mix: float = 0.1
    #: Piso absoluto de π(k) para cada braço de envio k ∈ S. 0 desliga.
    arm_floor: float = 0.05 / 3.0

    def __post_init__(self):
        if not (0.0 < self.p_min <= self.p_max < 1.0):
            raise ValueError(
                f"clipping inválido: exige 0 < p_min <= p_max < 1, "
                f"recebido [{self.p_min}, {self.p_max}]")
        if not (0.0 < self.arm_mix <= 1.0):
            raise ValueError(f"arm_mix precisa estar em (0, 1], recebido {self.arm_mix}")
        if not (0.0 <= self.arm_floor and
                MAX_SEND_ARMS * self.arm_floor <= self.p_min + _EPS):
            raise ValueError(
                f"arm_floor inválido: exige 0 <= arm_floor <= p_min/{MAX_SEND_ARMS} "
                f"= {self.p_min / MAX_SEND_ARMS:g}, recebido {self.arm_floor}. Para um "
                f"piso maior, suba também SPYNO_CLIP_MIN")

    @classmethod
    def from_config(cls, cfg) -> "ClipConfig":
        return cls(p_min=cfg.clip_min, p_max=cfg.clip_max, arm_mix=cfg.arm_mix,
                   arm_floor=cfg.arm_floor)

    def tag(self) -> str:
        return (f"clip[{self.p_min:g},{self.p_max:g}]mix{self.arm_mix:g}"
                f"floor{self.arm_floor:.6g}")

    def min_send_arm_prob(self, n_send: int) -> float:
        """Menor π(k) garantido a um braço de envio quando |S| = n_send."""
        if n_send == 1:
            return self.p_min
        return max(self.arm_floor, self.p_min * self.arm_mix / n_send)

    def max_importance_weight(self, n_send: int = MAX_SEND_ARMS) -> float:
        """Maior peso π_alvo(a)/π(a) para alvos com suporte em {0} ∪ S.

        O pior caso é um alvo determinístico que escolhe sempre o braço de menor
        probabilidade garantida: um braço de envio (piso) ou a abstenção
        (1 − p_max). Fora de {0} ∪ S o peso é infinito: na `ts_binary`
        (S = {3}) os braços 1 e 2 têm π = 0.
        """
        return 1.0 / min(self.min_send_arm_prob(n_send), 1.0 - self.p_max)


def _apply_floor(mass: float, share: np.ndarray, floor: float) -> np.ndarray:
    """Reparte `mass` em proporção a `share` com piso `floor` por braço.

    Water-filling: fixa no piso os braços que ficariam abaixo dele e redistribui
    o resto entre os demais, na proporção de `share`, até ninguém ficar abaixo.
    Termina em no máximo len(share) passos. Exige len(share)·floor <= mass.
    """
    out = mass * share
    if floor <= 0.0:
        return out
    fixed = np.zeros(len(share), dtype=bool)
    for _ in range(len(share)):
        low = (~fixed) & (out < floor)
        if not low.any():
            break
        fixed |= low
        free = ~fixed
        rest = mass - floor * fixed.sum()
        out = np.where(fixed, floor, 0.0)
        if free.any():
            w = share[free]
            out[free] = rest * (w / w.sum() if w.sum() > 0 else np.full(free.sum(), 1.0 / free.sum()))
    return out


def clip_distribution(wins: np.ndarray, send_arms: tuple[int, ...],
                      clip: ClipConfig) -> np.ndarray:
    """Aplica os passos 2 a 5 do esquema a uma contagem (ou distribuição) do TS."""
    wins = np.asarray(wins, dtype=float)
    total = wins.sum()
    if total <= 0:
        raise ValueError("distribuição do TS vazia")
    raw = wins / total
    send = list(send_arms)
    p_send = float(np.clip(raw[send].sum(), clip.p_min, clip.p_max))

    send_wins = wins[send]
    if send_wins.sum() > 0:
        prop = send_wins / send_wins.sum()
    else:
        prop = np.full(len(send), 1.0 / len(send))
    share = (1.0 - clip.arm_mix) * prop + clip.arm_mix / len(send)

    out = np.zeros(N_ARMS)
    out[0] = 1.0 - p_send
    out[send] = _apply_floor(p_send, share, clip.arm_floor)
    return out / out.sum()


class ClippedPolicy:
    """Envolve uma política que expõe `arm_wins(obs, n)` e decide por π clipada.

    `arm_wins` devolve a contagem de vitórias de cada braço em n amostras da
    posterior (ou, para o PPO, a distribuição categórica da própria política).
    """

    def __init__(self, inner, clip: ClipConfig, n_samples: int, seed: int = 0,
                 send_arms: tuple[int, ...] | None = None):
        self._inner = inner
        self.name = inner.name
        self.clip = clip
        self.n_samples = int(n_samples)
        self.send_arms = tuple(send_arms or getattr(inner, "send_arms", (1, 2, 3)))
        self._rng = np.random.default_rng(seed)
        # O serviço decide num pool de threads; o Generator do numpy não é seguro
        # para uso concorrente.
        self._lock = threading.Lock()

    def distribution(self, obs: dict, eligible: bool) -> np.ndarray:
        if not eligible:
            out = np.zeros(N_ARMS)
            out[0] = 1.0
            return out
        with self._lock:
            wins = self._inner.arm_wins(obs, self.n_samples)
        return clip_distribution(wins, self.send_arms, self.clip)

    def decide_with_distribution(self, user_id, obs, eligible, rng=None):
        """Devolve (ação, π). A ação é sorteada de π; π[ação] é a propensão exata."""
        pi = self.distribution(obs, eligible)
        with self._lock:
            arm = int(self._rng.choice(N_ARMS, p=pi))
        return arm, pi

    def decide(self, user_id, obs, eligible, rng=None):
        arm, pi = self.decide_with_distribution(user_id, obs, eligible, rng)
        return arm, float(pi[arm])

    def estimate_propensities(self, obs: dict, n_samples: int | None = None) -> np.ndarray:
        """π clipada num ponto elegível. Nova estimativa: só para diagnóstico.

        O registro usa a π de `decide_with_distribution`, a mesma do sorteio.
        Definido aqui para que o `__getattr__` não exponha a distribuição não
        clipada da política interna.
        """
        return self.distribution(obs, True)

    def start_user(self, user_id):
        return self._inner.start_user(user_id)

    def update(self, user_id, obs, arm, reward):
        with self._lock:
            return self._inner.update(user_id, obs, arm, reward)

    def end_day(self):
        with self._lock:
            return self._inner.end_day()

    def __getattr__(self, item):
        return getattr(self._inner, item)
