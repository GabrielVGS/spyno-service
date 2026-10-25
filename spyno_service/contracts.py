"""Contrato da interface comum de decisão.

Este módulo é a resposta de engenharia à questão de pesquisa sobre viabilidade:
o consumidor fala sempre o mesmo contrato, qualquer que seja a política ativa.
Trocar PPO por Thompson Sampling ou por uma política de referência não muda
nada aqui.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

#: Braços. O zero é a abstenção; os demais são os tipos de gatilho do Fogg.
ARMS: tuple[str, ...] = ("no_notify", "spark", "facilitator", "signal")


class Context(BaseModel):
    """Sinais observáveis no ponto de decisão.

    Só entram aqui grandezas que o aplicativo consegue registrar. Motivação e
    habilidade latentes não são campos deste contrato.

    `notifs_today` é a CONTAGEM BRUTA de envios no dia local (o guardrail da
    cota compara com `SPYNO_DAILY_CAP`); a política a vê dividida por 5.
    `fbm_obs` continua aceito, mas desde o esquema v2 não entra no vetor das
    políticas aprendidas: fica no log e alimenta a referência `fbm_obs_threshold`.

    Contadores no shadow mode: em shadow nada é enviado, então os contadores
    "reais" (`notifs_today`, `fatigue`, `minutes_since_last`) ficam presos em
    zero/None. Os campos `*_proposed` são os mesmos contadores calculados pelo
    backend sobre a ação PROPOSTA (`proposed_action` da resposta), isto é, a
    que teria sido executada sem o shadow: a `recommended_action` depois dos
    guardrails reais (silêncio, cota, intervalo). Uma recomendação bloqueada
    por guardrail não teria saído e não conta. Em shadow o serviço usa os `*_proposed` para as
    features e os guardrails, e registra as duas versões no log
    (`counter_basis` diz qual foi usada). Fora do shadow os dois coincidem e o
    serviço usa os reais.
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

    #: Contadores sobre a ação proposta (o que a política teria enviado). Usados
    #: em shadow mode; ausentes, o serviço cai nos reais e registra isso.
    notifs_today_proposed: float | None = Field(default=None, ge=0)
    fatigue_proposed: float | None = Field(default=None, ge=0, le=1)
    minutes_since_last_proposed: int | None = Field(default=None, ge=0)


class DecideRequest(BaseModel):
    user_key: str = Field(min_length=1, description="Identificador pseudonimizado (HMAC no "
                          "backend); o serviço o trata como opaco")
    decision_point: datetime = Field(description="Instante do ponto de decisão, com fuso. "
                                     "A deduplicação usa o instante truncado na hora (UTC)")
    context: Context
    eligible: bool = Field(default=True, description="Elegibilidade avaliada pelo backend")


class DecideResponse(BaseModel):
    decision_id: str
    recommended_action: int = Field(ge=0, le=3, description="Ação escolhida pela política")
    recommended_arm: str
    executed_action: int = Field(ge=0, le=3, description="Ação após guardrails e shadow mode")
    proposed_action: int = Field(ge=0, le=3, description="Ação que teria sido executada sem o "
                                 "shadow: a recomendada após os guardrails reais. Fora do "
                                 "shadow é igual à executed_action. Base dos contadores "
                                 "*_proposed que o backend envia")
    propensity: float = Field(gt=0, le=1, description="P(ação executada | contexto) sob o "
                              "processo de coleta: política clipada, guardrails e shadow. "
                              "Vale 1 quando um guardrail ou o shadow forçam a abstenção")
    recommended_propensity: float = Field(gt=0, le=1, description="P(ação recomendada) sob a "
                                          "política clipada: a propensão da ação escolhida")
    arm_probabilities: list[float] | None = Field(default=None, description="Distribuição "
                                                  "clipada sobre os 4 braços, da qual a ação "
                                                  "recomendada foi sorteada")
    propensity_method: Literal["monte-carlo", "exact", "fallback"]
    counter_basis: Literal["executed", "proposed"] = Field(description="Contadores usados nas "
                                                           "features e nos guardrails")
    update_eligible: bool = Field(description="O ponto é treinável: elegível pelo backend, sem "
                                  "guardrail que bloquearia um envio e sem contingência. "
                                  "Treina de fato só se learning_frozen=false")
    learning_frozen: bool = Field(default=False, description="Decidido com a posterior "
                                  "congelada (shadow sem SPYNO_SHADOW_LEARN): o desfecho é "
                                  "registrado, mas nunca treina (desde o contrato 2.3)")
    policy: str
    policy_version: str
    shadow: bool
    fallback: bool
    fallback_reason: str | None = None
    guardrail_reason: str | None = None
    latency_ms: float


class FeedbackRequest(BaseModel):
    """Desfecho de uma decisão: o usuário treinou dentro da janela?

    Vale para qualquer braço, inclusive o 0. `responded` treina com y=1,
    `not_responded` com y=0, sempre sobre a `executed_action`. `censored` e
    `delivery_failed` só fecham o pendente, sem treino. Uma resposta chegada
    depois da janela é gravada como `censored`. `value` é guardado no log; não
    define o y do treino.
    """

    decision_id: str
    observed_at: datetime
    outcome: Literal["responded", "not_responded", "delivery_failed", "censored"]
    value: int = Field(ge=0, le=1, default=0)


class FeedbackResponse(BaseModel):
    decision_id: str
    applied: bool
    reason: str | None = None
    #: A posterior recebeu (x, executed_action, y) deste desfecho.
    trained: bool = False
    train_skip_reason: str | None = None


class EndDayRequest(BaseModel):
    #: Data LOCAL do dia que se fecha. Uma segunda chamada com a mesma data não faz nada.
    date: date


class EndDayResponse(BaseModel):
    ok: bool
    date: date
    already_closed: bool = False
    updates_applied: int = 0
    error: str | None = None
    #: Posterior congelada (shadow): o dia foi fechado sem ajuste e sem gravar
    #: posterior (desde o contrato 2.3).
    learning_frozen: bool = False


class ContractInfo(BaseModel):
    """O que o backend confere na subida; ele se recusa a iniciar se divergir."""

    contract_version: str
    policy: str
    policy_version: str
    policy_schema: str
    outcome_window_min: int
    shadow: bool
    features: list[str]
    arms: list[str]
    clip_min: float
    clip_max: float
    arm_mix: float
    #: Piso de π(k) por braço de envio (desde o contrato 2.1).
    arm_floor: float
    #: Maior peso de importância para alvos com suporte em {0} ∪ S (S = braços
    #: de envio da política ativa), dado o clipping dela; fora desse suporte o
    #: peso é infinito (ts_binary: braços 1 e 2). None se a política não é
    #: clipada (never, always).
    max_importance_weight: float | None
    daily_cap: int
    cooldown_min: int
    quiet_start_h: int
    quiet_end_h: int
    #: Estado inicial da linhagem (desde o contrato 2.2): "warmstart" ou
    #: "prior". Faz parte da policy_version. Campos aditivos: o backend não
    #: precisa lê-los.
    posterior_init: str | None = None
    #: sha256 do artefato de warm-start carregado, ou None.
    warmstart_sha256: str | None = None
    #: Desde o contrato 2.3. True: a posterior não aprende neste processo
    #: (shadow sem SPYNO_SHADOW_LEARN); a policy_version não muda com isso.
    learning_frozen: bool = False
    #: SPYNO_SHADOW_LEARN: em shadow, treinar com (x, 0, y) (comportamento
    #: anterior à 2.3, não validado). Padrão False.
    shadow_learn: bool = False


class Health(BaseModel):
    status: Literal["ok"]
    policy: str
    policy_version: str
    shadow: bool
    decisions: int
    pending_outcomes: int
    #: De onde veio a posterior em uso na subida: "persisted", "warmstart" ou
    #: "prior" (desde o contrato 2.2). None para políticas sem posterior.
    posterior_source: str | None = None
    #: A posterior está congelada (shadow sem SPYNO_SHADOW_LEARN): decide e
    #: feedback só registram, end-day não ajusta (desde o contrato 2.3).
    learning_frozen: bool = False
