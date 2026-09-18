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

    #: Amostras de Monte Carlo para a propensão registrada.
    propensity_samples: int = int(os.getenv("SPYNO_PROPENSITY_SAMPLES", "64"))

    #: Janela de atribuição do desfecho, em minutos.
    outcome_window_min: int = int(os.getenv("SPYNO_OUTCOME_WINDOW_MIN", "180"))

    #: Guardrails aplicados fora da política.
    quiet_start_h: int = int(os.getenv("SPYNO_QUIET_START", "22"))
    quiet_end_h: int = int(os.getenv("SPYNO_QUIET_END", "6"))
    daily_cap: int = int(os.getenv("SPYNO_DAILY_CAP", "3"))
    cooldown_min: int = int(os.getenv("SPYNO_COOLDOWN_MIN", "90"))

    checkpoint_dir: str = os.getenv("SPYNO_CHECKPOINT_DIR", "./checkpoints")


CONFIG = Config()
