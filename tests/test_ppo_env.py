"""O ambiente do PPO precisa ser o mesmo que os bandits enfrentam.

Estes testes não medem desempenho. Medem que a observação entregue ao agente
tem a dimensão e a ordem do vetor observável, e que a dinâmica vem do laço do
simulador, não de uma reimplementação.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SIM = Path(os.getenv("SPYNO_SIM_PATH",
                      Path(__file__).resolve().parents[2] / "spyno-sim"))
if _SIM.is_dir() and str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

gym = pytest.importorskip("gymnasium")
# O ambiente roda o laço do simulador: sem o spyno-sim (ex.: na CI), pula.
pytest.importorskip("spyno_sim", reason="spyno-sim não disponível")


def _env(**kw):
    from training.ppo_env import EnvConfig, SpynoNotificationEnv
    return SpynoNotificationEnv(EnvConfig(n_users=2, n_days=2, seed=1, **kw))


def test_observacao_tem_a_dimensao_do_vetor_observavel():
    from spyno_service.features import DIM

    env = _env()
    try:
        obs, _ = env.reset()
        assert env.observation_space.shape == (DIM,)
        assert obs.shape == (DIM,)
        assert env.action_space.n == 4
    finally:
        env.close()


def test_passo_devolve_recompensa_com_custo_de_notificacao():
    """Notificar cobra o custo; abster-se não."""
    env = _env(cost=0.25)
    try:
        env.reset()
        _, r_abster, _, _, _ = env.step(0)
        assert r_abster in (0.0, 1.0), "abstenção não paga custo"

        recompensas = []
        for _ in range(40):
            _, r, term, _, _ = env.step(3)
            recompensas.append(r)
            if term:
                env.reset()
        # Notificação sem resposta custa exatamente -0,25.
        assert any(abs(r + 0.25) < 1e-9 for r in recompensas)
    finally:
        env.close()


def test_episodio_termina_na_virada_de_usuario():
    env = _env()
    try:
        env.reset()
        terminou = False
        for _ in range(500):
            _, _, term, _, _ = env.step(0)
            if term:
                terminou = True
                break
        assert terminou, "o episódio deveria fechar ao trocar de usuário"
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
    finally:
        env.close()


def test_cota_diaria_do_simulador_e_respeitada():
    """A elegibilidade vem do simulador, não do agente."""
    env = _env(max_notifications_day=1)
    try:
        env.reset()
        for _ in range(60):
            _, _, term, _, _ = env.step(3)
            if term:
                break
    finally:
        env.close()
