"""Baselines para a avaliação (RQ1). Todas seguem a interface Policy do simulador."""

from __future__ import annotations


class _Stateless:
    def start_user(self, user_id): pass
    def update(self, user_id, obs, arm, reward): pass
    def end_day(self): pass


class NeverNotify(_Stateless):
    name = "never"

    def decide(self, user_id, obs, eligible, rng):
        return 0, 1.0


class AlwaysNotifyRandomArm(_Stateless):
    """Notifica sempre que elegível (sob caps do ambiente), braço uniforme."""

    name = "always"

    def decide(self, user_id, obs, eligible, rng):
        if not eligible:
            return 0, 1.0
        return int(rng.integers(1, 4)), 1.0 / 3.0


class FBMHeuristic(_Stateless):
    """Heurística do Spyno-Fogg: notificar quando FBM > 40 (trainer.py:187-215).

    Recebe generosamente o fbm COMPLETO (com sono e arousal, que o sistema
    original assumia coletáveis). Braço 'signal' — o sistema original tinha um
    único tipo de notificação (lembrete).
    """

    name = "fbm_gt40"

    def __init__(self, threshold: float = 40.0):
        self.threshold = threshold

    def decide(self, user_id, obs, eligible, rng):
        if eligible and float(obs["fbm_full"]) > self.threshold:
            return 3, 1.0
        return 0, 1.0
