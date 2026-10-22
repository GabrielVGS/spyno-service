"""Testes de contrato: as propriedades de que a avaliação posterior depende.

Não testam qualidade de política. Testam que o registro produzido é utilizável:
idempotência, propensão válida, separação entre ação recomendada e executada,
desfecho aplicado uma única vez e modo de observação que não envia nada.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("SPYNO_DB_URL", "sqlite+pysqlite:///:memory:")

#: Token só de teste; o real vem de SPYNO_DECISION_TOKEN.
TOKEN = "token-de-teste"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class StubPolicy:
    """Política determinística: notifica com o braço 'signal' quando elegível."""

    name = "stub"

    def __init__(self):
        self.updates = []
        self.decide_calls = 0

    def start_user(self, user_id):
        pass

    def decide(self, user_id, obs, eligible, rng=None):
        self.decide_calls += 1
        return (3, 1.0) if eligible else (0, 1.0)

    def update(self, user_id, obs, arm, reward):
        self.updates.append((user_id, arm, reward))

    def end_day(self):
        pass

    def estimate_propensities(self, obs, n_samples=64):
        import numpy as np
        return np.array([0.25, 0.0, 0.0, 0.75])


def _context(hour=10, notifs_today=0, minutes_since_last=None):
    return {
        "in_free_window": 1, "valence_pos": 0.5, "is_weekend": 0, "recency": 0.3,
        "already_trained": 0, "streak_norm": 0.2, "emp_adhesion": 0.4,
        "fatigue": 0.1, "notifs_today": notifs_today, "morning": 1,
        "midday": 0, "evening": 0, "fbm_obs": 0.7,
        "hour": hour, "minutes_since_last": minutes_since_last,
    }


@pytest.fixture()
def client(tmp_path):
    """Serviço com política de teste e banco próprio, sem tocar no ambiente."""
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from spyno_service.app import create_app
    from spyno_service.config import Config
    from spyno_service.store import Store

    cfg = replace(
        Config(),
        database_url=f"sqlite+pysqlite:///{tmp_path}/t.db",
        policy="stub",
        shadow=False,
        decision_token=TOKEN,
    )
    stub = StubPolicy()
    app = create_app(policy=stub, store=Store(cfg.database_url), config=cfg)
    with TestClient(app, headers=AUTH) as c:
        c.stub = stub
        yield c


def test_decide_devolve_contrato_completo(client):
    r = client.post("/v1/decide", json={
        "user_key": "u1",
        "decision_point": "2026-09-17T10:00:00-03:00",
        "context": _context(),
        "eligible": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recommended_action"] == 3
    assert body["recommended_arm"] == "signal"
    assert body["executed_action"] == 3
    assert 0.0 < body["propensity"] <= 1.0
    assert body["propensity_method"] == "monte-carlo"
    assert body["policy_version"]
    assert body["fallback"] is False


def test_idempotencia_nao_reamostra(client):
    payload = {
        "user_key": "u2",
        "decision_point": "2026-09-17T11:00:00-03:00",
        "context": _context(hour=11),
        "eligible": True,
    }
    first = client.post("/v1/decide", json=payload).json()
    chamadas = client.stub.decide_calls
    second = client.post("/v1/decide", json=payload).json()

    assert first["decision_id"] == second["decision_id"]
    assert client.stub.decide_calls == chamadas, "a política foi consultada de novo"


def test_propensao_corresponde_a_acao_executada(client):
    body = client.post("/v1/decide", json={
        "user_key": "u3",
        "decision_point": "2026-09-17T12:00:00-03:00",
        "context": _context(hour=12),
        "eligible": True,
    }).json()
    # o stub atribui 0,75 ao braço 3 e 0,25 à abstenção
    assert body["executed_action"] == 3
    assert body["propensity"] == pytest.approx(0.75)


def test_guardrail_separa_recomendada_de_executada(client):
    body = client.post("/v1/decide", json={
        "user_key": "u4",
        "decision_point": "2026-09-17T23:00:00-03:00",
        "context": _context(hour=23),
        "eligible": True,
    }).json()
    assert body["recommended_action"] == 3, "a política decidiu notificar"
    assert body["executed_action"] == 0, "o guardrail bloqueou"
    assert body["guardrail_reason"] == "quiet_hours"


def test_cota_diaria_bloqueia(client):
    body = client.post("/v1/decide", json={
        "user_key": "u5",
        "decision_point": "2026-09-17T14:00:00-03:00",
        "context": _context(hour=14, notifs_today=3),
        "eligible": True,
    }).json()
    assert body["executed_action"] == 0
    assert body["guardrail_reason"] == "daily_cap"


def test_feedback_aplicado_uma_unica_vez(client):
    decision = client.post("/v1/decide", json={
        "user_key": "u6",
        "decision_point": "2026-09-17T15:00:00-03:00",
        "context": _context(hour=15),
        "eligible": True,
    }).json()
    fb = {
        "decision_id": decision["decision_id"],
        "observed_at": "2026-09-17T15:20:00-03:00",
        "outcome": "responded",
        "value": 1,
    }
    first = client.post("/v1/feedback", json=fb).json()
    second = client.post("/v1/feedback", json=fb).json()

    assert first["applied"] is True
    assert second["applied"] is False
    assert second["reason"] == "desfecho_ja_aplicado"
    assert len(client.stub.updates) == 1, "a política foi atualizada duas vezes"


def test_feedback_fora_da_janela_vira_censura(client):
    decision = client.post("/v1/decide", json={
        "user_key": "u7",
        "decision_point": "2026-09-17T09:00:00-03:00",
        "context": _context(hour=9),
        "eligible": True,
    }).json()
    tarde = datetime(2026, 9, 17, 9, tzinfo=timezone.utc) + timedelta(days=1)
    r = client.post("/v1/feedback", json={
        "decision_id": decision["decision_id"],
        "observed_at": tarde.isoformat(),
        "outcome": "responded",
        "value": 1,
    }).json()
    assert r["applied"] is True
    assert client.stub.updates == [], "resposta fora da janela não pode treinar"


def test_feedback_sem_decisao_e_rejeitado(client):
    r = client.post("/v1/feedback", json={
        "decision_id": "inexistente",
        "observed_at": "2026-09-17T10:00:00-03:00",
        "outcome": "responded",
        "value": 1,
    }).json()
    assert r["applied"] is False
    assert r["reason"] == "decisao_inexistente"


def _app_with(policy, tmp_path, **overrides):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from spyno_service.app import create_app
    from spyno_service.config import Config
    from spyno_service.store import Store

    defaults = {"database_url": f"sqlite+pysqlite:///{tmp_path}/x.db",
                "policy": "stub", "shadow": False, "decision_token": TOKEN}
    cfg = replace(Config(), **{**defaults, **overrides})
    return TestClient(create_app(policy=policy, store=Store(cfg.database_url), config=cfg),
                      headers=AUTH)


def test_shadow_mode_nunca_executa(tmp_path):
    """Em modo de observação a decisão é registrada, mas nada é enviado."""
    stub = StubPolicy()
    with _app_with(stub, tmp_path, shadow=True) as c:
        body = c.post("/v1/decide", json={
            "user_key": "s1",
            "decision_point": "2026-09-17T10:00:00-03:00",
            "context": _context(),
            "eligible": True,
        }).json()
    assert body["recommended_action"] == 3, "a política decidiu notificar"
    assert body["executed_action"] == 0, "nada pode ser enviado em shadow"
    assert body["guardrail_reason"] == "shadow"
    assert body["shadow"] is True


def test_politica_que_falha_cai_na_contingencia(tmp_path):
    """Falha da política não deixa o aplicativo sem resposta."""

    class Quebrada(StubPolicy):
        name = "quebrada"

        def decide(self, user_id, obs, eligible, rng=None):
            raise RuntimeError("modelo ausente")

    with _app_with(Quebrada(), tmp_path) as c:
        body = c.post("/v1/decide", json={
            "user_key": "f1",
            "decision_point": "2026-09-17T10:00:00-03:00",
            "context": _context(),
            "eligible": True,
        }).json()
    assert body["fallback"] is True
    assert body["fallback_reason"] == "RuntimeError"
    assert body["executed_action"] == 0, "a contingência é conservadora"
    assert body["propensity_method"] == "fallback", \
        "o ponto precisa ficar marcado como inutilizável para avaliação off-policy"


def test_timeout_cai_na_contingencia(tmp_path):
    """Estourar o orçamento de latência aciona a regra fixa."""
    import time as _time

    class Lenta(StubPolicy):
        name = "lenta"

        def decide(self, user_id, obs, eligible, rng=None):
            _time.sleep(0.5)
            return 3, 1.0

    with _app_with(Lenta(), tmp_path, decide_timeout_s=0.05) as c:
        body = c.post("/v1/decide", json={
            "user_key": "t1",
            "decision_point": "2026-09-17T10:00:00-03:00",
            "context": _context(),
            "eligible": True,
        }).json()
    assert body["fallback"] is True
    assert body["fallback_reason"] == "timeout"
    assert body["executed_action"] == 0


def test_janela_vencida_e_fechada_como_censura(tmp_path):
    """O job periódico fecha as janelas que passaram sem resposta."""
    stub = StubPolicy()
    with _app_with(stub, tmp_path, outcome_window_min=0) as c:
        c.post("/v1/decide", json={
            "user_key": "w1",
            "decision_point": "2026-09-17T10:00:00-03:00",
            "context": _context(),
            "eligible": True,
        })
        assert c.get("/health").json()["pending_outcomes"] == 1
        assert c.post("/v1/close-windows").json()["censored"] == 1
        assert c.get("/health").json()["pending_outcomes"] == 0


def test_inelegivel_nao_consulta_politica_para_notificar(tmp_path):
    stub = StubPolicy()
    with _app_with(stub, tmp_path) as c:
        body = c.post("/v1/decide", json={
            "user_key": "i1",
            "decision_point": "2026-09-17T10:00:00-03:00",
            "context": _context(),
            "eligible": False,
        }).json()
    assert body["recommended_action"] == 0
    assert body["executed_action"] == 0
