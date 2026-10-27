"""Política PPO servida a partir de um checkpoint do Stable-Baselines3.

Sobre os checkpoints existentes
-------------------------------
O `Spyno-Fogg` treinou PPO sobre uma observação de oito dimensões:

    [hour, day_period, is_weekend, motivation, ability, trigger, sleeping,
     min(notifs_today, 10)]

Três dessas entradas não existem na formulação atual. Motivação, habilidade e
gatilho entravam ali como valores exatos do simulador; aqui só existe o escore
estimado `fbm_obs`. E `sleeping` dependia de sensor de sono, descartado na
definição de sensores viáveis. Servir aqueles checkpoints exigiria fabricar
essas três entradas, o que quebraria a comparabilidade com as demais políticas
e invalidaria qualquer avaliação off-policy sobre os registros resultantes.

Por isso esta classe carrega um checkpoint treinado sobre o MESMO vetor
observável das demais políticas. `vectorize` parte do contexto do contrato
(notifs_today bruto) e o converte para o espaço da política.

Esquema v2: o vetor perdeu `fbm_obs` e passou a ter `DIM` = 13 dimensões. O
checkpoint `ppo_observable.zip` foi treinado no esquema v1 (14 dimensões) e é
recusado de forma explícita pela checagem de dimensão abaixo até ser retreinado
com `training/train_ppo.py`. Servido, o PPO passa pelo mesmo clipping dos bandits. Os cinco checkpoints antigos ficam como
artefato histórico do trabalho anterior, não como modelo servido.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..features import DIM, vectorize
from .base import N_ARMS


class PPOPolicy:
    """Serve um PPO treinado sobre o vetor observável comum.

    A propensão vem da distribuição categórica da própria política, o que
    resolve o problema que o PPO determinístico tinha: avaliado pela ação de
    maior probabilidade, ele registra propensão 1,0 e destrói a sobreposição de
    que a avaliação off-policy depende.
    """

    def __init__(self, checkpoint: str | Path, name: str = "ppo",
                 deterministic: bool = False, device: str = "cpu"):
        self.name = name
        self.checkpoint = Path(checkpoint)
        self.deterministic = deterministic
        self._model = None
        self._device = device

    # ---- ciclo de vida ---------------------------------------------------

    def _lazy_load(self):
        if self._model is None:
            from stable_baselines3 import PPO  # importado aqui: torch é opcional

            if not self.checkpoint.exists():
                raise FileNotFoundError(
                    f"checkpoint do PPO não encontrado em {self.checkpoint}. "
                    "Treine sobre o vetor observável antes de servir esta política."
                )
            self._model = PPO.load(str(self.checkpoint), device=self._device)
            obs_dim = int(np.prod(self._model.observation_space.shape))
            if obs_dim != DIM:
                raise ValueError(
                    f"o checkpoint espera observação de {obs_dim} dimensões, mas o "
                    f"vetor observável tem {DIM}. Este é o checkpoint antigo do "
                    "Spyno-Fogg, treinado com motivação, habilidade e sono exatos; "
                    "ele não pode ser servido sob a formulação atual."
                )
        return self._model

    def warmup(self) -> None:
        """Carrega o modelo fora do caminho da requisição.

        Sem isso a primeira decisão paga o custo de importar o torch e ler o
        checkpoint, estoura o orçamento de latência e cai na contingência.
        """
        self._lazy_load()

    def start_user(self, user_id: str) -> None:
        pass

    def end_day(self) -> None:
        pass

    def update(self, user_id: str, obs: dict, arm: int, reward: int) -> None:
        """PPO é on-policy: não aprende de desfecho isolado fora do laço de treino.

        O desfecho fica no log de decisões e alimenta o retreino em lote, feito
        fora do caminho de serving.
        """

    # ---- decisão ---------------------------------------------------------

    def decide(self, user_id: str, obs: dict, eligible: bool, rng=None) -> tuple[int, float]:
        if not eligible:
            return 0, 1.0
        model = self._lazy_load()
        x = vectorize(obs).astype(np.float32)
        action, _ = model.predict(x, deterministic=self.deterministic)
        return int(action), 1.0

    def estimate_propensities(self, obs: dict, n_samples: int = 64) -> np.ndarray:
        """Distribuição sobre braços, lida da política em vez de amostrada."""
        import torch

        model = self._lazy_load()
        x = vectorize(obs).astype(np.float32).reshape(1, -1)
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(x)
            dist = model.policy.get_distribution(tensor)
            probs = dist.distribution.probs.cpu().numpy().ravel()
        out = np.zeros(N_ARMS)
        out[: len(probs)] = probs[:N_ARMS]
        total = out.sum()
        return out / total if total > 0 else np.full(N_ARMS, 1.0 / N_ARMS)

    #: Braços de envio, para o clipping.
    send_arms = (1, 2, 3)

    def arm_wins(self, obs: dict, n_samples: int = 0) -> np.ndarray:
        """Entrada do ClippedPolicy: a distribuição categórica, sem amostragem."""
        return self.estimate_propensities(obs)
