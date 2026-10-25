"""Shadow congela a posterior (contrato 2.3).

Em shadow (`SPYNO_SHADOW=1`, padrão) o serviço só coleta logs: decide e
feedback são registrados por inteiro, mas nenhum update entra no buffer e o
end-day fecha o dia sem tocar em m/q. A posterior que sai do shadow é
exatamente a que entrou. `SPYNO_SHADOW_LEARN=1` restaura o comportamento antigo.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import select

os.environ.setdefault("SPYNO_DB_URL", "sqlite+pysqlite:///:memory:")

from spyno_service.policies import bandit  # noqa: E402
from spyno_service.policies.clipping import ClipConfig, ClippedPolicy  # noqa: E402
from spyno_service.posterior import digest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WARM = ROOT / "warmstart" / "ts_hybrid-v2-c0.05.json"
TOKEN = "token-de-teste"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CLIP = ClipConfig(p_min=0.1, p_max=0.9, arm_mix=0.1)
DIA0 = date(2026, 10, 22)
USERS = ("a", "b", "c")


def _context(hour, **extra):
    return {"in_free_window": 1, "valence_pos": 0.5, "is_weekend": 0, "recency": 0.3,
            "already_trained": 0, "streak_norm": 0.2, "emp_adhesion": 0.4,
            "fatigue": 0.1, "notifs_today": 0, "morning": 1, "midday": 0,
            "evening": 0, "fbm_obs": 0.7, "hour": hour, "minutes_since_last": None,
            "notifs_today_proposed": 0, "fatigue_proposed": 0.0,
            "minutes_since_last_proposed": None, **extra}


def _app(db, shadow, warm=WARM, policy=None, **overrides):
    from spyno_service.app import create_app
    from spyno_service.config import Config
    from spyno_service.store import Store

    cfg = replace(Config(), database_url=f"sqlite+pysqlite:///{db}", policy="ts_hybrid",
                  shadow=shadow, decision_token=TOKEN,
                  warmstart_snapshot=None if warm is None else str(warm), **overrides)
    pol = policy if policy is not None else bandit.ts_hybrid(clip=CLIP, n_samples=50)
    return create_app(policy=pol, store=Store(cfg.database_url), config=cfg)


def _client(app):
    from fastapi.testclient import TestClient

    return TestClient(app, headers=AUTH)


def _state(c):
    return c.get("/v1/policy-state").json()


def _dia(c, dia: date, hours=range(8, 14), feedback=True):
    """Um dia de decisões horárias para USERS, com desfechos alternados."""
    ids = []
    for h in hours:
        for i, u in enumerate(USERS):
            point = f"{dia.isoformat()}T{h:02d}:00:00-03:00"
            r = c.post("/v1/decide", json={"user_key": u, "decision_point": point,
                                           "context": _context(h), "eligible": True})
            assert r.status_code == 200, r.text
            body = r.json()
            ids.append(body["decision_id"])
            if feedback:
                outcome = ("responded", "not_responded", "censored")[(h + i) % 3]
                fb = c.post("/v1/feedback", json={
                    "decision_id": body["decision_id"], "outcome": outcome,
                    "value": int(outcome == "responded"),
                    "observed_at": point.replace(":00:00", ":30:00")}).json()
                assert fb["applied"] is True
    return ids


def _artifact_digest():
    art = json.loads(WARM.read_text())
    return digest({k: art[k] for k in ("m", "q", "cost", "schema", "features")})


def _posterior_rows(app):
    from spyno_service.store import policy_posteriors

    with app.state.ctx["store"].engine.begin() as conn:
        return conn.execute(select(policy_posteriors)).mappings().all()


# ---- shadow: digest constante -------------------------------------------------

def test_shadow_varios_dias_nao_mudam_o_digest(tmp_path):
    app = _app(tmp_path / "s.db", shadow=True)
    with _client(app) as c:
        s0 = _state(c)
        assert s0["learning_frozen"] is True
        assert s0["posterior_digest"] == _artifact_digest()
        for k in range(3):
            dia = DIA0 + timedelta(days=k)
            _dia(c, dia)
            meio = _state(c)
            assert meio["posterior_digest"] == s0["posterior_digest"]
            assert meio["pending_updates"] == 0, "nada entra no buffer em shadow"
            r = c.post("/v1/end-day", json={"date": dia.isoformat()}).json()
            assert r == {"ok": True, "date": dia.isoformat(), "already_closed": False,
                         "updates_applied": 0, "error": None, "learning_frozen": True}
            again = c.post("/v1/end-day", json={"date": dia.isoformat()}).json()
            assert again["already_closed"] is True and again["updates_applied"] == 0
            assert _state(c)["posterior_digest"] == s0["posterior_digest"]
        fim = _state(c)
        health = c.get("/health").json()
        contrato = c.get("/v1/contract").json()
    assert fim["posterior_digest"] == s0["posterior_digest"]
    assert fim["policy_version"] == s0["policy_version"]
    assert [d["updates_applied"] for d in fim["days_closed"]] == [0, 0, 0]
    assert fim["persisted_day"] is None
    assert _posterior_rows(app) == [], "shadow congelado não grava posterior"
    assert health["learning_frozen"] is True and health["shadow"] is True
    assert contrato["learning_frozen"] is True and contrato["shadow_learn"] is False
    assert contrato["contract_version"] == "2.3"


def test_policy_version_nao_depende_do_shadow(tmp_path):
    with _client(_app(tmp_path / "a.db", shadow=True)) as c:
        v_shadow = _state(c)["policy_version"]
    with _client(_app(tmp_path / "b.db", shadow=False)) as c:
        v_real = _state(c)["policy_version"]
    with _client(_app(tmp_path / "c.db", shadow=True, shadow_learn=True)) as c:
        v_learn = _state(c)["policy_version"]
    assert v_shadow == v_real == v_learn


def test_shadow_nao_chama_update_nem_end_day_da_politica(tmp_path):
    from tests.test_policy_v2 import Recorder

    rec = Recorder(wins=(0, 0, 0, 10))
    app = _app(tmp_path / "r.db", shadow=True, warm=None,
               policy=ClippedPolicy(rec, CLIP, n_samples=1, seed=1))
    with _client(app) as c:
        _dia(c, DIA0)
        assert c.post("/v1/end-day", json={"date": DIA0.isoformat()}).json()["ok"] is True
    assert rec.updates == [] and rec.end_days == 0


# ---- shadow: log completo e utilizável offline ---------------------------------

def test_shadow_log_continua_completo(tmp_path):
    app = _app(tmp_path / "l.db", shadow=True)
    with _client(app) as c:
        ids = _dia(c, DIA0)
        store = app.state.ctx["store"]
        rows = [store.get(i) for i in ids]
        resp = c.post("/v1/decide", json={"user_key": "a",
                                           "decision_point": "2026-10-22T08:10:00-03:00",
                                           "context": _context(8), "eligible": True}).json()
    assert resp["learning_frozen"] is True and resp["update_eligible"] is True
    for r in rows:
        assert r["shadow"] is True and r["learning_frozen"] is True
        assert r["executed_action"] == 0 and r["propensity"] == 1.0
        assert r["update_eligible"] is True, "a seleção treinável fica no log para a OPE"
        assert r["counter_basis"] == "proposed"
        assert json.loads(r["context"])["notifs_today_proposed"] == 0
        assert json.loads(r["policy_context"])["notifs_today"] == 0
        pi = json.loads(r["arm_probs"])
        assert len(pi) == 4 and abs(sum(pi) - 1) < 1e-9
        assert r["recommended_propensity"] == pi[r["recommended_action"]]
        assert r["proposed_action"] == r["recommended_action"]
        assert r["outcome"] in ("responded", "not_responded", "censored")
        assert r["trained"] is False
        esperado = ("sem_treino_censored" if r["outcome"] == "censored"
                    else "decisao_em_shadow_congelado")
        assert r["train_skip_reason"] == esperado
    assert {r["recommended_action"] for r in rows} - {0}, "a política propôs envios"


# ---- fora do shadow: comportamento antigo ------------------------------------

def test_fora_do_shadow_end_day_muda_e_persiste(tmp_path):
    app = _app(tmp_path / "n.db", shadow=False)
    with _client(app) as c:
        s0 = _state(c)
        assert s0["learning_frozen"] is False
        ids = _dia(c, DIA0)
        meio = _state(c)
        r = c.post("/v1/end-day", json={"date": DIA0.isoformat()}).json()
        s1 = _state(c)
        row = app.state.ctx["store"].get(ids[0])
    assert meio["pending_updates"] > 0
    assert r["updates_applied"] == meio["pending_updates"] and r["learning_frozen"] is False
    assert s1["posterior_digest"] != s0["posterior_digest"]
    assert s1["persisted_day"] == DIA0.isoformat()
    assert row["learning_frozen"] is False
    assert len(_posterior_rows(app)) == 1


def test_shadow_learn_restaura_o_aprendizado_em_shadow(tmp_path):
    with _client(_app(tmp_path / "sl.db", shadow=True, shadow_learn=True)) as c:
        s0 = _state(c)
        _dia(c, DIA0)
        r = c.post("/v1/end-day", json={"date": DIA0.isoformat()}).json()
        s1 = _state(c)
        health = c.get("/health").json()
    assert s0["learning_frozen"] is False and health["learning_frozen"] is False
    assert r["updates_applied"] > 0 and r["learning_frozen"] is False
    assert s1["posterior_digest"] != s0["posterior_digest"]


# ---- transições ----------------------------------------------------------------

def test_saida_do_shadow_carrega_exatamente_o_artefato(tmp_path):
    db = tmp_path / "t.db"
    with _client(_app(db, shadow=True)) as c:
        s_shadow = _state(c)
        for k in range(2):
            _dia(c, DIA0 + timedelta(days=k))
            c.post("/v1/end-day", json={"date": (DIA0 + timedelta(days=k)).isoformat()})
        # decisão de shadow com janela ainda aberta no reinício
        aberta = c.post("/v1/decide", json={
            "user_key": "z", "decision_point": "2026-10-24T09:00:00-03:00",
            "context": _context(9), "eligible": True}).json()
    app = _app(db, shadow=False)
    with _client(app) as c:
        s = _state(c)
        assert s["posterior_source"] == "warmstart" and s["learning_frozen"] is False
        assert s["posterior_digest"] == s_shadow["posterior_digest"] == _artifact_digest()
        assert s["policy_version"] == s_shadow["policy_version"]
        # o desfecho da decisão de shadow chega depois de sair do shadow: não treina
        fb = c.post("/v1/feedback", json={"decision_id": aberta["decision_id"],
                                          "outcome": "responded", "value": 1,
                                          "observed_at": "2026-10-24T09:30:00-03:00"}).json()
        assert fb["trained"] is False
        assert fb["train_skip_reason"] == "decisao_em_shadow_congelado"
        assert _state(c)["pending_updates"] == 0
    np.testing.assert_array_equal(
        np.asarray(app.state.ctx["policy"].snapshot()["m"]),
        np.asarray(json.loads(WARM.read_text())["m"]))


def test_shadow_depois_de_aprender_preserva_a_ultima_persistida(tmp_path):
    db = tmp_path / "p.db"
    with _client(_app(db, shadow=False)) as c:
        _dia(c, DIA0)
        c.post("/v1/end-day", json={"date": DIA0.isoformat()})
        aprendida = _state(c)
        # decisão treinável com janela aberta quando o serviço vai para o shadow
        aberta = c.post("/v1/decide", json={
            "user_key": "z", "decision_point": "2026-10-23T09:00:00-03:00",
            "context": _context(9), "eligible": True}).json()
    assert aprendida["posterior_digest"] != _artifact_digest()
    with _client(_app(db, shadow=True)) as c:
        s = _state(c)
        assert s["posterior_source"] == "persisted"
        assert s["posterior_digest"] == aprendida["posterior_digest"]
        fb = c.post("/v1/feedback", json={"decision_id": aberta["decision_id"],
                                          "outcome": "responded", "value": 1,
                                          "observed_at": "2026-10-23T09:30:00-03:00"}).json()
        assert fb["trained"] is False and fb["train_skip_reason"] == "aprendizado_congelado"
        for k in (1, 2):
            dia = DIA0 + timedelta(days=k)
            _dia(c, dia, hours=range(10, 13))
            c.post("/v1/end-day", json={"date": dia.isoformat()})
            assert _state(c)["posterior_digest"] == aprendida["posterior_digest"]
        assert _state(c)["persisted_day"] == DIA0.isoformat()
    with _client(_app(db, shadow=False)) as c:
        s = _state(c)
    assert s["posterior_source"] == "persisted" and s["persisted_day"] == DIA0.isoformat()
    assert s["posterior_digest"] == aprendida["posterior_digest"]
