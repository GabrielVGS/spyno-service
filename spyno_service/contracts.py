"""Contrato da interface comum de decisão.

Este módulo é a resposta de engenharia à questão de pesquisa sobre viabilidade:
o consumidor fala sempre o mesmo contrato, qualquer que seja a política ativa.
Trocar PPO por Thompson Sampling ou por uma política de referência não muda
nada aqui.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

#: Braços. O zero é a abstenção; os demais são os tipos de gatilho do Fogg.
ARMS: tuple[str, ...] = ("no_notify", "spark", "facilitator", "signal")


class Context(BaseModel):
    """Sinais observáveis no ponto de decisão.

    Só entram aqui grandezas que o aplicativo consegue registrar. Motivação e
    habilidade latentes não são campos deste contrato: chegam, quando chegam,
    como o escore estimado `fbm_obs`.
    """

    in_free_window: float = Field(ge=0, le=1)
    valence_pos: float = Field(ge=0, le=1)
    is_weekend: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    already_trained: float = Field(ge=0, le=1)
    streak_norm: float = Field(ge=0, le=1)
    emp_adhesion: float = Field(ge=0, le=1)
    fatigue: float = Field(ge=0, le=1)
    notifs_today: float = Field(ge=0)
    morning: float = Field(ge=0, le=1)
    midday: float = Field(ge=0, le=1)
    evening: float = Field(ge=0, le=1)
    fbm_obs: float = Field(ge=0, le=1)

    #: Hora local (0-23) no instante da decisão. Usada pelos guardrails e pela
    #: adaptação de observação do PPO, não pelas features do bandit.
    hour: int = Field(ge=0, le=23)

    #: Minutos desde a última notificação entregue. None se nunca houve.
    minutes_since_last: int | None = Field(default=None, ge=0)


class DecideRequest(BaseModel):
    user_key: str = Field(min_length=1, description="Identificador pseudonimizado")
    decision_point: datetime = Field(description="Instante do ponto de decisão, com fuso")
    context: Context
    eligible: bool = Field(default=True, description="Elegibilidade avaliada pelo backend")


class DecideResponse(BaseModel):
    decision_id: str
    recommended_action: int = Field(ge=0, le=3, description="Ação escolhida pela política")
    recommended_arm: str
    executed_action: int = Field(ge=0, le=3, description="Ação após guardrails e shadow mode")
    propensity: float = Field(gt=0, le=1, description="P(ação executada) sob a política de coleta")
    propensity_method: Literal["monte-carlo", "exact", "fallback"]
    policy: str
    policy_version: str
    shadow: bool
    fallback: bool
    fallback_reason: str | None = None
    guardrail_reason: str | None = None
    latency_ms: float


class FeedbackRequest(BaseModel):
    decision_id: str
    observed_at: datetime
    outcome: Literal["responded", "not_responded", "delivery_failed", "censored"]
    value: int = Field(ge=0, le=1, default=0)


class FeedbackResponse(BaseModel):
    decision_id: str
    applied: bool
    reason: str | None = None


class Health(BaseModel):
    status: Literal["ok"]
    policy: str
    policy_version: str
    shadow: bool
    decisions: int
    pending_outcomes: int
