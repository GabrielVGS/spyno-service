"""Ambiente Gymnasium sobre o simulador do Spyno, para treinar PPO.

O ponto delicado deste módulo é a comparabilidade. Se o ambiente do PPO fosse
reescrito à parte, ele divergiria do laço que avalia os bandits e a comparação
entre as duas famílias perderia sentido: qualquer diferença de desempenho
poderia vir da dinâmica, não da política.

Por isso aqui não há reimplementação. O laço `spyno_sim.simulate.rollout` roda
numa thread e a política que ele consulta é um adaptador que bloqueia à espera
da ação do agente. O PPO enfrenta literalmente o mesmo código de ambiente,
com a mesma função de resposta, a mesma fadiga e a mesma contagem de cota.

A observação é `vectorize(obs)`, o mesmo vetor observável de 16 dimensões que
as demais políticas recebem. Motivação e habilidade latentes não entram, o que
é a diferença central em relação aos checkpoints do trabalho anterior.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from spyno_service.features import DIM, vectorize

_SENTINEL_TIMEOUT = 60.0


@dataclass
class EnvConfig:
    n_users: int = 12
    n_days: int = 28
    seed: int = 42
    max_notifications_day: int = 5
    #: Custo por notificação. É o mesmo λ da utilidade dos bandits: sem isso o
    #: PPO otimizaria um objetivo diferente e a comparação seria inválida.
    cost: float = 0.05


class _AgentPolicy:
    """Política que delega cada decisão ao agente, através de duas filas."""

    name = "ppo-trainer"

    def __init__(self, to_env: queue.Queue, to_policy: queue.Queue, cost: float):
        self.to_env = to_env
        self.to_policy = to_policy
        self.cost = cost
        self._last_arm = 0

    def start_user(self, user_id: str) -> None:
        self.to_env.put(("start", user_id))

    def decide(self, user_id, obs, eligible, rng=None):
        self.to_env.put(("obs", vectorize(obs).astype(np.float32), eligible))
        arm = self.to_policy.get()
        if arm is None:                     # pedido de encerramento
            raise _Stop()
        self._last_arm = int(arm)
        return self._last_arm, 1.0

    def update(self, user_id, obs, arm, reward) -> None:
        # Mesma utilidade dos bandits: resposta obtida menos o custo de interromper.
        self.to_env.put(("reward", float(reward) - self.cost * (arm > 0)))

    def end_day(self) -> None:
        pass


class _Stop(Exception):
    """Encerramento cooperativo da thread do simulador."""


class SpynoNotificationEnv(gym.Env):
    """Um episódio é a trajetória completa de um usuário simulado."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self._to_env: queue.Queue = queue.Queue()
        self._to_policy: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._pending_obs: np.ndarray | None = None
        self._world_epoch = 0

    # ---- ciclo do simulador ---------------------------------------------

    def _start_world(self) -> None:
        from spyno_sim.config import Demographics
        from spyno_sim.response import ResponseParams
        from spyno_sim.simulate import SimConfig, build_world, rollout

        sim_cfg = SimConfig(
            n_users=self.cfg.n_users,
            n_days=self.cfg.n_days,
            seed=self.cfg.seed + self._world_epoch,
            llm_mode="stub",                 # determinístico: sem chamada externa
            max_notifications_day=self.cfg.max_notifications_day,
        )
        world, _ = build_world(sim_cfg, Demographics())
        policy = _AgentPolicy(self._to_env, self._to_policy, self.cfg.cost)

        def _run():
            try:
                rollout(world, policy, ResponseParams(), sim_cfg,
                        outcome_seed=sim_cfg.seed + 10_000, collect_rows=False)
                self._to_env.put(("done",))
            except _Stop:
                self._to_env.put(("done",))
            except Exception as exc:                     # pragma: no cover
                self._to_env.put(("error", exc))

        self._world_epoch += 1
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _recv(self):
        msg = self._to_env.get(timeout=_SENTINEL_TIMEOUT)
        if msg[0] == "error":                            # pragma: no cover
            raise RuntimeError("simulador falhou") from msg[1]
        return msg

    # ---- interface Gymnasium --------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._pending_obs is not None:
            obs, self._pending_obs = self._pending_obs, None
            return obs, {}

        if self._thread is None or not self._thread.is_alive():
            self._start_world()

        while True:
            msg = self._recv()
            if msg[0] == "obs":
                return msg[1], {}
            if msg[0] == "done":
                self._start_world()

    def step(self, action):
        self._to_policy.put(int(action))

        reward = 0.0
        while True:
            msg = self._recv()
            if msg[0] == "reward":
                reward = msg[1]
                continue
            if msg[0] == "obs":
                return msg[1], reward, False, False, {}
            if msg[0] == "start":
                # Virada de usuário: fim do episódio. A próxima observação já
                # pertence ao usuário seguinte e fica guardada para o reset.
                nxt = self._recv()
                if nxt[0] == "obs":
                    self._pending_obs = nxt[1]
                return np.zeros(DIM, dtype=np.float32), reward, True, False, {}
            if msg[0] == "done":
                return np.zeros(DIM, dtype=np.float32), reward, True, False, {}

    def close(self):
        if self._thread is not None and self._thread.is_alive():
            self._to_policy.put(None)
            self._thread.join(timeout=5.0)
        self._thread = None
