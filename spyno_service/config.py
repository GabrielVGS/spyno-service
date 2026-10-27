"""Configuração por variável de ambiente, com padrões de desenvolvimento."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    database_url: str = os.getenv("SPYNO_DB_URL", "sqlite+pysqlite:///./spyno_service.db")

    #: Política ativa. Nome registrado em policies.registry.
    policy: str = os.getenv("SPYNO_POLICY", "ts_hybrid")

    #: Em shadow mode o serviço decide e registra, mas devolve executed_action=0
    #: e o backend não envia nada. Padrão ligado: nada chega ao usuário por acidente.
    shadow: bool = os.getenv("SPYNO_SHADOW", "1") not in ("0", "false", "False")

    #: Orçamento de latência da decisão. Estourado, responde pela regra de contingência.
    decide_timeout_s: float = float(os.getenv("SPYNO_DECIDE_TIMEOUT", "0.2"))

    #: Amostras da posterior para estimar a distribuição do Thompson Sampling
    #: sobre braços (passo 1 do esquema em policies/clipping.py). A propensão
    #: registrada é exata para a distribuição sorteada, qualquer que seja este
    #: número; ele só controla o ruído de Monte Carlo da própria política.
    propensity_samples: int = int(os.getenv("SPYNO_PROPENSITY_SAMPLES", "1000"))

    #: Probability clipping (esquema v2): P(enviar) fica em [clip_min, clip_max]
    #: nos pontos elegíveis; a massa de envio é repartida com mistura uniforme
    #: arm_mix e cada braço de envio recebe ao menos arm_floor (probabilidade
    #: absoluta; exige 3·arm_floor <= clip_min). O peso de importância máximo
    #: de um alvo determinístico fica em 1/min(arm_floor, 1 - clip_max): 30 no
    #: padrão. Ver policies/clipping.py e o README.
    clip_min: float = float(os.getenv("SPYNO_CLIP_MIN", "0.1"))
    clip_max: float = float(os.getenv("SPYNO_CLIP_MAX", "0.9"))
    arm_mix: float = float(os.getenv("SPYNO_ARM_MIX", "0.1"))
    arm_floor: float = float(os.getenv("SPYNO_ARM_FLOOR", repr(1.0 / 30.0)))

    #: Janela de atribuição do desfecho, em minutos.
    outcome_window_min: int = int(os.getenv("SPYNO_OUTCOME_WINDOW_MIN", "180"))

    #: Guardrails aplicados fora da política.
    quiet_start_h: int = int(os.getenv("SPYNO_QUIET_START", "22"))
    quiet_end_h: int = int(os.getenv("SPYNO_QUIET_END", "6"))
    daily_cap: int = int(os.getenv("SPYNO_DAILY_CAP", "3"))
    cooldown_min: int = int(os.getenv("SPYNO_COOLDOWN_MIN", "90"))

    #: Autenticação dos /v1/* (exceto /v1/contract): Authorization: Bearer <token>.
    #: Sem token configurado o serviço responde 503 (fail closed), a menos que
    #: SPYNO_DECISION_AUTH=disabled esteja definido explicitamente (desenvolvimento).
    decision_token: str | None = os.getenv("SPYNO_DECISION_TOKEN") or None
    auth_disabled: bool = os.getenv("SPYNO_DECISION_AUTH", "") == "disabled"

    checkpoint_dir: str = os.getenv("SPYNO_CHECKPOINT_DIR", "./checkpoints")


CONFIG = Config()
