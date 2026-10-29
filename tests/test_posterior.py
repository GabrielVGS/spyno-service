"""Warm-start, persistência da posterior e ordem de precedência na subida.

persisted (mesma policy_version) > warmstart (SPYNO_WARMSTART_SNAPSHOT) > prior.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

os.environ.setdefault("SPYNO_DB_URL", "sqlite+pysqlite:///:memory:")

from spyno_service.features import FEATURES, POLICY_SCHEMA, vectorize  # noqa: E402
from spyno_service.policies import bandit  # noqa: E402
from spyno_service.policies.clipping import ClipConfig  # noqa: E402
from spyno_service.posterior import digest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WARM = ROOT / "warmstart" / "ts_hybrid-v2-c0.1.json"
TOKEN = "token-de-teste"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CLIP = ClipConfig(p_min=0.1, p_max=0.9, arm_mix=0.1, arm_floor=1.0 / 30.0)  # p10


def _context(hour=10, notifs_today=0, **extra):
    return {"in_free_window": 1, "valence_pos": 0.5, "is_weekend": 0, "recency": 0.3,
            "already_trained": 0, "streak_norm": 0.2, "emp_adhesion": 0.4,
            "fatigue": 0.1, "notifs_today": notifs_today, "morning": 1, "midday": 0,
            "evening": 0, "fbm_obs": 0.7, "hour": hour, "minutes_since_last": None,
            **extra}


def _app(db, policy=None, warm=None, name="ts_hybrid", **overrides):
    from spyno_service.app import create_app
    from spyno_service.config import Config
    from spyno_service.store import Store

    fields = {"database_url": f"sqlite+pysqlite:///{db}", "policy": name, "shadow": False,
              "decision_token": TOKEN,
              "warmstart_snapshot": None if warm is None else str(warm), **overrides}
    cfg = replace(Config(), **fields)
    pol = policy if policy is not None else bandit.ts_hybrid(clip=CLIP, n_samples=50)
    return create_app(policy=pol, store=Store(cfg.database_url), config=cfg)


def _client(app):
    from fastapi.testclient import TestClient

    return TestClient(app, headers=AUTH)


def _train_one(c, point="2026-10-22T10:00:00-03:00", user="u"):
    r = c.post("/v1/decide", json={"user_key": user, "decision_point": point,
                                   "context": _context(), "eligible": True})
    assert r.status_code == 200, r.text
    fb = c.post("/v1/feedback", json={"decision_id": r.json()["decision_id"],
                                      "observed_at": point.replace(":00:00", ":30:00"),
                                      "outcome": "responded", "value": 1}).json()
    assert fb["trained"] is True
    return r.json()


def _artifact(tmp_path, **changes) -> Path:
    art = json.loads(WARM.read_text())
    art.update(changes)
    p = tmp_path / "warm.json"
    p.write_text(json.dumps(art))
    return p


# ---- artefato ---------------------------------------------------------------

def test_artefato_versionado_no_esquema_servido():
    art = json.loads(WARM.read_text())
    assert art["format"] == "spyno-warmstart/1"
    assert art["policy"] == "ts_hybrid" and art["cost"] == 0.1
    assert art["schema"] == POLICY_SCHEMA and art["features"] == list(FEATURES)
    assert art["vector"]["service_dim"] == 13
    assert set(art["vector"]["zeroed_columns"]) == {"location_home", "motion_stationary",
                                                    "fbm_obs"}
    assert len(art["m"]) == len(bandit.ts_hybrid(clip=CLIP, n_samples=1).snapshot()["m"])
    assert np.any(np.array(art["m"]) != 0.0), "o warm-start move a média"
    assert bandit.ts_hybrid(clip=CLIP, n_samples=1).snapshot()["cost"] == art["cost"], \
        "o artefato do compose tem o custo padrão da fábrica"
    assert art["fit"]["eligible_rows"] > 1000
    assert art["recipe"]["seed"] == 0 and art["recipe"]["outcome_seed"] == 90_000
    cc = art["provenance"]["spyno_decision"].get("crosscheck")
    assert cc is None or (cc["identical_m"] and cc["identical_q"])
    for repo in ("spyno_sim", "spyno_service"):
        assert art["provenance"][repo]["commit"]


def test_colunas_zeradas_do_warm_start_coincidem_com_o_serviço():
    """Linha do simulador (notifs ÷5, sensores e fbm do sim) e o mesmo ponto
    vindo do backend (notifs bruto) dão o MESMO vetor para a política."""
    from spyno_decision.features import vectorize as vec_interno

    linha_sim = {**_context(notifs_today=2 / 5.0), "location_home": 1.0,
                 "motion_stationary": 1.0, "fbm_obs": 0.83, "eligible": 1, "action": 2,
                 "action_performed": 1}
    servida = bandit.ts_hybrid(clip=CLIP, n_samples=1)
    adaptador = servida._inner                           # _CompatibilidadeHerdada
    x_warm = vec_interno(bandit.warm_start_rows([linha_sim])[0])
    x_servico = vec_interno(adaptador._obs(_context(notifs_today=2)))
    np.testing.assert_array_equal(x_warm, x_servico)
    # as 12 posições do contrato (mais o viés) aparecem iguais no vetor interno (16)
    from spyno_decision.features import FEATURES as INTERNAS
    np.testing.assert_array_equal(vectorize(_context(notifs_today=2)),
                                  x_servico[[0] + [1 + INTERNAS.index(k) for k in FEATURES]])
    # e o resto do vetor interno é exatamente o que o serviço zera
    resto = [1 + INTERNAS.index(k) for k in INTERNAS if k not in FEATURES]
    assert np.all(x_servico[resto] == 0.0) and np.all(x_warm[resto] == 0.0)


def _sim_path():
    env = os.getenv("SPYNO_SIM_PATH")
    cands = [Path(env)] if env else [p / "spyno-sim" for p in ROOT.parents]
    return next((c for c in cands if (c / "spyno_sim" / "simulate.py").exists()), None)


@pytest.mark.skipif(_sim_path() is None, reason="spyno-sim não disponível")
def test_build_warmstart_e_deterministico(tmp_path):
    out = tmp_path / "ws.json"
    env = {**os.environ, "SPYNO_SIM_PATH": str(_sim_path())}
    subprocess.run([sys.executable, str(ROOT / "training" / "build_warmstart.py"),
                    "--out", str(out)], check=True, cwd=ROOT, env=env,
                   capture_output=True)
    novo, versionado = json.loads(out.read_text()), json.loads(WARM.read_text())
    assert novo["m"] == versionado["m"] and novo["q"] == versionado["q"]
    assert novo["fit"] == versionado["fit"]


# ---- warm-start na subida ---------------------------------------------------

def test_warm_start_muda_posterior_e_policy_version(tmp_path):
    with _client(_app(tmp_path / "a.db")) as c:
        prior = c.get("/v1/policy-state").json()
        contrato_prior = c.get("/v1/contract").json()
    with _client(_app(tmp_path / "b.db", warm=WARM)) as c:
        warm = c.get("/v1/policy-state").json()
        contrato = c.get("/v1/contract").json()
        health = c.get("/health").json()
    assert prior["posterior_source"] == "prior" and warm["posterior_source"] == "warmstart"
    assert warm["posterior_digest"] != prior["posterior_digest"]
    assert warm["policy_version"] != prior["policy_version"], "warm-start é outra linhagem"
    assert warm["policy_version"].startswith("ts_hybrid-v2-")
    assert contrato["contract_version"] == "2.3"
    assert contrato["posterior_init"] == "warmstart" and contrato_prior["posterior_init"] == "prior"
    assert contrato["warmstart_sha256"] and contrato_prior["warmstart_sha256"] is None
    assert health["posterior_source"] == "warmstart"
    art = json.loads(WARM.read_text())
    assert warm["posterior_digest"] == digest(
        {k: art[k] for k in ("m", "q", "cost", "schema", "features")})


def test_sem_warm_start_policy_version_do_prior_nao_muda(tmp_path):
    """Quem não define SPYNO_WARMSTART_SNAPSHOT continua na mesma versão de antes."""
    from spyno_service.policies.registry import version

    with _client(_app(tmp_path / "a.db")) as c:
        v = c.get("/v1/contract").json()["policy_version"]
    assert v == version(bandit.ts_hybrid(clip=CLIP, n_samples=50))


# ---- persistência -----------------------------------------------------------

def test_reinicio_apos_end_day_recupera_a_posterior(tmp_path):
    db = tmp_path / "p.db"
    with _client(_app(db, warm=WARM)) as c:
        _train_one(c)
        assert c.post("/v1/end-day", json={"date": "2026-10-22"}).json()["updates_applied"] == 1
        antes = c.get("/v1/policy-state").json()
    assert antes["persisted_day"] == "2026-10-22"

    with _client(_app(db, warm=WARM)) as c:        # nova política, mesmo banco
        depois = c.get("/v1/policy-state").json()
        assert c.get("/health").json()["posterior_source"] == "persisted"
        # a mesma versão: a decisão de antes do reinício continua idempotente
        r = c.post("/v1/decide", json={"user_key": "u",
                                       "decision_point": "2026-10-22T10:10:00-03:00",
                                       "context": _context(), "eligible": True}).json()
        assert c.get("/health").json()["decisions"] == 1, "mesmo decision_id"
    assert depois["posterior_source"] == "persisted"
    assert depois["posterior_digest"] == antes["posterior_digest"]
    assert depois["policy_version"] == antes["policy_version"]
    assert depois["posterior_init"] == "warmstart"
    assert r["policy_version"] == antes["policy_version"]


def test_persistida_tem_precedencia_e_continua_aprendendo(tmp_path):
    db = tmp_path / "p.db"
    with _client(_app(db, warm=WARM)) as c:
        _train_one(c)
        c.post("/v1/end-day", json={"date": "2026-10-22"})
        d1 = c.get("/v1/policy-state").json()["posterior_digest"]
    with _client(_app(db, warm=WARM)) as c:
        _train_one(c, point="2026-10-23T10:00:00-03:00")
        c.post("/v1/end-day", json={"date": "2026-10-23"})
        d2 = c.get("/v1/policy-state").json()["posterior_digest"]
    with _client(_app(db, warm=WARM)) as c:
        d3 = c.get("/v1/policy-state").json()
    assert d1 != d2 and d3["posterior_digest"] == d2 and d3["persisted_day"] == "2026-10-23"


def test_end_day_repetido_nao_duplica_update_nem_persistencia(tmp_path):
    db = tmp_path / "p.db"
    app = _app(db, warm=WARM)
    store = app.state.ctx["store"]
    with _client(app) as c:
        _train_one(c)
        r1 = c.post("/v1/end-day", json={"date": "2026-10-22"}).json()
        linha1 = store.load_posterior(app.state.ctx["policy_version"])
        s1 = c.get("/v1/policy-state").json()
        _train_one(c, point="2026-10-22T12:00:00-03:00")   # update novo no buffer
        r2 = c.post("/v1/end-day", json={"date": "2026-10-22"}).json()
        linha2 = store.load_posterior(app.state.ctx["policy_version"])
        s2 = c.get("/v1/policy-state").json()
    assert r1["updates_applied"] == 1 and r2["already_closed"] is True
    assert s2["posterior_digest"] == s1["posterior_digest"]
    assert s2["pending_updates"] == 1, "o update novo espera o próximo dia"
    assert linha2 == linha1, "a persistência não foi regravada"
    assert s2["days_closed"] == [{"day": "2026-10-22", "updates_applied": 1}]
    with store.engine.begin() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM policy_posteriors")).scalar_one() == 1


def test_falha_ao_gravar_nao_fecha_o_dia(tmp_path):
    """Fechamento e posterior vão juntos: sem a posterior, o dia fica aberto."""
    from spyno_service.store import METADATA

    app = _app(tmp_path / "p.db", warm=WARM)
    store = app.state.ctx["store"]
    with _client(app) as c:
        _train_one(c)
        with store.engine.begin() as conn:
            conn.execute(text("DROP TABLE policy_posteriors"))
        r1 = c.post("/v1/end-day", json={"date": "2026-10-22"}).json()
        assert r1["ok"] is False and r1["error"].startswith("persistencia_")
        assert not store.day_closed(__import__("datetime").date(2026, 10, 22))
        falhou = c.get("/v1/policy-state").json()
        assert falhou["pending_updates"] == 1, "não zera sem gravar"
        # feedback chegado no intervalo: entra no ajuste e no dia D da nova tentativa
        _train_one(c, point="2026-10-22T12:00:00-03:00")
        METADATA.create_all(store.engine)
        r2 = c.post("/v1/end-day", json={"date": "2026-10-22"}).json()
        s = c.get("/v1/policy-state").json()
    assert r2["ok"] is True and r2["updates_applied"] == 2, "os dois lotes contam no dia"
    assert s["pending_updates"] == 0
    assert s["posterior_digest"] != falhou["posterior_digest"], "o update do intervalo entrou"
    row = store.load_posterior(app.state.ctx["policy_version"])
    assert row["digest"] == s["posterior_digest"]


# ---- recusas ----------------------------------------------------------------

@pytest.mark.parametrize("mudanca", [
    {"schema": "v1"},
    {"features": list(FEATURES) + ["fbm_obs"]},
    {"policy": "ts_disjoint"},
    {"cost": 0.03},
    {"format": "outro"},
])
def test_warm_start_incompativel_e_recusado(tmp_path, mudanca):
    with pytest.raises(bandit.IncompatibleSnapshotError):
        _app(tmp_path / "x.db", warm=_artifact(tmp_path, **mudanca))


def test_warm_start_de_outra_politica_e_recusado(tmp_path):
    """ts_disjoint e ts_hybrid: o nome no artefato não confere."""
    with pytest.raises(bandit.IncompatibleSnapshotError):
        _app(tmp_path / "x.db", warm=WARM, name="ts_disjoint",
             policy=bandit.ts_disjoint(clip=CLIP, n_samples=10))


def test_warm_start_inexistente_derruba_a_subida(tmp_path):
    with pytest.raises(FileNotFoundError):
        _app(tmp_path / "x.db", warm=tmp_path / "nao-existe.json")


def test_warm_start_em_politica_sem_posterior_e_recusado(tmp_path):
    with pytest.raises(bandit.IncompatibleSnapshotError):
        _app(tmp_path / "x.db", warm=WARM, name="never", policy=bandit.never())


def test_persistida_de_outra_versao_nao_e_carregada(tmp_path):
    db = tmp_path / "p.db"
    with _client(_app(db, warm=WARM)) as c:
        _train_one(c)
        c.post("/v1/end-day", json={"date": "2026-10-22"})
        v_warm = c.get("/v1/policy-state").json()["policy_version"]
    # sem o warm-start: outra linhagem; a posterior gravada não é carregada
    with _client(_app(db)) as c:
        s = c.get("/v1/policy-state").json()
    assert s["posterior_source"] == "prior" and s["policy_version"] != v_warm
    assert [v["policy_version"] for v in s["skipped_persisted"]] == [v_warm]
    # outro clipping também muda a versão
    with _client(_app(db, warm=WARM, policy=bandit.ts_hybrid(
            clip=ClipConfig(0.2, 0.8, 0.1), n_samples=50))) as c:
        s = c.get("/v1/policy-state").json()
    assert s["posterior_source"] == "warmstart"
    assert s["policy_version"] != v_warm
    assert v_warm in [v["policy_version"] for v in s["skipped_persisted"]]


def _treina(c, point, user, ctx, outcome):
    r = c.post("/v1/decide", json={"user_key": user, "decision_point": point,
                                   "context": ctx, "eligible": True})
    assert r.status_code == 200, r.text
    fb = c.post("/v1/feedback", json={"decision_id": r.json()["decision_id"],
                                      "observed_at": point.replace(":00:00", ":30:00"),
                                      "outcome": outcome, "value": int(outcome == "responded")})
    assert fb.json()["trained"] is True, fb.text
    return r.json()


def _dia_aberto(tmp_path, **cfg):
    """Dia 22 fechado e quatro updates distintos no buffer do dia 23.

    Contextos, desfechos e usuários diferentes: o ajuste em lote depende da ordem
    das linhas, então só assim o teste de reconstrução detecta ordem errada.
    """
    db = tmp_path / "p.db"
    app = _app(db, warm=WARM, **cfg)
    c = _client(app)
    c.__enter__()
    _train_one(c)
    c.post("/v1/end-day", json={"date": "2026-10-22"})
    abertas = [
        _treina(c, "2026-10-23T09:00:00-03:00", "u", _context(fatigue=0.0), "responded"),
        _treina(c, "2026-10-23T10:00:00-03:00", "v", _context(in_free_window=0, fatigue=0.6),
                "not_responded"),
        _treina(c, "2026-10-23T13:00:00-03:00", "w", _context(morning=0, midday=1, recency=0.9),
                "responded"),
        _treina(c, "2026-10-23T16:00:00-03:00", "u", _context(morning=0, evening=1, notifs_today=2),
                "not_responded"),
    ]
    assert c.get("/v1/policy-state").json()["pending_updates"] == 4
    return db, app, c, abertas


def test_reinicio_reconstroi_o_buffer_do_dia_aberto(tmp_path):
    """Reiniciar antes do end-day dá a mesma posterior que não reiniciar, bit a bit."""
    import shutil

    db, app, c, abertas = _dia_aberto(tmp_path)
    copia = tmp_path / "reinicio.db"
    shutil.copy(db, copia)                       # o banco no instante do reinício
    sem = c.post("/v1/end-day", json={"date": "2026-10-23"}).json()
    sem_estado = c.get("/v1/policy-state").json()
    c.__exit__(None, None, None)

    with _client(_app(copia, warm=WARM)) as c2:
        subida = c2.get("/v1/policy-state").json()
        com = c2.post("/v1/end-day", json={"date": "2026-10-23"}).json()
        com_estado = c2.get("/v1/policy-state").json()
    assert subida["pending_updates"] == 4 and subida["persisted_day"] == "2026-10-22"
    assert com["updates_applied"] == sem["updates_applied"] == 4
    assert com_estado["posterior_digest"] == sem_estado["posterior_digest"]
    from spyno_service.store import Store
    for d in abertas:
        assert str(Store(f"sqlite+pysqlite:///{copia}").get(d["decision_id"])["applied_day"]) \
            == "2026-10-23"


def test_reinicio_repetido_nao_reaplica_duas_vezes(tmp_path):
    db, app, c, _ = _dia_aberto(tmp_path)
    c.__exit__(None, None, None)
    with _client(_app(db, warm=WARM)) as c2:
        assert c2.get("/v1/policy-state").json()["pending_updates"] == 4
        _train_one(c2, point="2026-10-23T12:00:00-03:00", user="x")
    with _client(_app(db, warm=WARM)) as c3:
        assert c3.get("/v1/policy-state").json()["pending_updates"] == 5
        assert c3.post("/v1/end-day", json={"date": "2026-10-23"}).json()["updates_applied"] == 5
    with _client(_app(db, warm=WARM)) as c4:      # depois do fechamento, nada pendente
        s = c4.get("/v1/policy-state").json()
    assert s["pending_updates"] == 0 and s["persisted_day"] == "2026-10-23"


def test_linhas_anteriores_a_coluna_nao_sao_reaplicadas(tmp_path):
    """Sem train_seq não se sabe se a linha já entrou num ajuste: fica de fora."""
    db, app, c, _ = _dia_aberto(tmp_path)
    c.__exit__(None, None, None)
    with app.state.ctx["store"].engine.begin() as conn:
        conn.execute(text("UPDATE decisions SET train_seq = NULL"))
    with _client(_app(db, warm=WARM)) as c2:
        assert c2.get("/v1/policy-state").json()["pending_updates"] == 0


def test_subida_congelada_nao_reaplica_e_a_saida_do_shadow_reaplica(tmp_path):
    db, app, c, _ = _dia_aberto(tmp_path)
    fechado = app.state.ctx["store"].load_posterior(app.state.ctx["policy_version"])["digest"]
    c.__exit__(None, None, None)
    with _client(_app(db, warm=WARM, shadow=True)) as c2:
        s = c2.get("/v1/policy-state").json()
    assert s["pending_updates"] == 0 and s["posterior_digest"] == fechado
    with _client(_app(db, warm=WARM)) as c3:
        assert c3.get("/v1/policy-state").json()["pending_updates"] == 4


def test_ordem_da_reconstrucao_importa(tmp_path, monkeypatch):
    """Controle do teste bit a bit: reaplicar em outra ordem muda a posterior."""
    import shutil

    from spyno_service.store import Store

    db, app, c, _ = _dia_aberto(tmp_path)
    copia = tmp_path / "reinicio.db"
    shutil.copy(db, copia)
    certo = c.post("/v1/end-day", json={"date": "2026-10-23"}).json()
    certo_digest = c.get("/v1/policy-state").json()["posterior_digest"]
    c.__exit__(None, None, None)
    original = Store.unapplied_updates
    monkeypatch.setattr(Store, "unapplied_updates",
                        lambda self, *a: list(reversed(original(self, *a))))
    with _client(_app(copia, warm=WARM)) as c2:
        c2.post("/v1/end-day", json={"date": "2026-10-23"})
        invertido = c2.get("/v1/policy-state").json()["posterior_digest"]
    assert certo["updates_applied"] == 4 and invertido != certo_digest


def test_reaplicar_linha_de_shadow_marca_a_posterior(tmp_path):
    """Dia fechado limpo; depois updates aprendidos em shadow (SPYNO_SHADOW_LEARN=1)
    e reinício fora do shadow antes do end-day: a marca vem das linhas reaplicadas,
    e a subida sem a flag não carrega a posterior resultante."""
    db = tmp_path / "p.db"
    with _client(_app(db, warm=WARM)) as c:
        _train_one(c)
        c.post("/v1/end-day", json={"date": "2026-10-22"})
        assert c.get("/v1/policy-state").json()["learned_in_shadow"] is False
    with _client(_app(db, warm=WARM, shadow=True, shadow_learn=True)) as c:
        _treina(c, "2026-10-23T09:00:00-03:00", "u", _context(), "responded")
    with _client(_app(db, warm=WARM, shadow_learn=True)) as c:
        s = c.get("/v1/policy-state").json()
        assert s["pending_updates"] == 1 and s["learned_in_shadow"] is True
        assert c.post("/v1/end-day", json={"date": "2026-10-23"}).json()["updates_applied"] == 1
    with _client(_app(db, warm=WARM)) as c:
        s = c.get("/v1/policy-state").json()
    assert s["posterior_source"] != "persisted"
    assert s["skipped_persisted"][0]["reason"] == "aprendida_em_shadow"


def test_falha_ao_gravar_marca_os_dois_lotes_na_nova_tentativa(tmp_path):
    from spyno_service.store import METADATA

    app = _app(tmp_path / "p.db", warm=WARM)
    store = app.state.ctx["store"]
    with _client(app) as c:
        a = _train_one(c)
        with store.engine.begin() as conn:
            conn.execute(text("DROP TABLE policy_posteriors"))
        assert c.post("/v1/end-day", json={"date": "2026-10-22"}).json()["ok"] is False
        assert store.get(a["decision_id"])["applied_day"] is None
        b = _train_one(c, point="2026-10-22T12:00:00-03:00")
        METADATA.create_all(store.engine)
        assert c.post("/v1/end-day", json={"date": "2026-10-22"}).json()["ok"] is True
    assert {str(store.get(d["decision_id"])["applied_day"]) for d in (a, b)} == {"2026-10-22"}


def test_persistida_corrompida_derruba_a_subida(tmp_path):
    db = tmp_path / "p.db"
    app = _app(db, warm=WARM)
    with _client(app) as c:
        _train_one(c)
        c.post("/v1/end-day", json={"date": "2026-10-22"})
    with app.state.ctx["store"].engine.begin() as conn:
        conn.execute(text("UPDATE policy_posteriors SET digest = 'adulterado'"))
    with pytest.raises(RuntimeError, match="digest"):
        _app(db, warm=WARM)


def test_persistida_de_esquema_antigo_derruba_a_subida(tmp_path):
    """Mesma versão gravada, esquema adulterado: load_snapshot recusa (bandit._check_snapshot)."""
    db = tmp_path / "p.db"
    app = _app(db, warm=WARM)
    with _client(app) as c:
        _train_one(c)
        c.post("/v1/end-day", json={"date": "2026-10-22"})
    store = app.state.ctx["store"]
    row = store.load_posterior(app.state.ctx["policy_version"])
    snap = {k: v for k, v in row["snapshot"].items() if k != "schema"}   # = v1
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE policy_posteriors SET snapshot = :s"),
                     {"s": json.dumps(snap)})
    with pytest.raises(RuntimeError, match="não carrega"):
        _app(db, warm=WARM)


# ---- padrões ------------------------------------------------------------------

def test_padroes_shadow_ligado_e_sem_warm_start():
    """Config lê o ambiente na importação: confere num processo limpo."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("SPYNO_SHADOW", "SPYNO_WARMSTART_SNAPSHOT", "SPYNO_POLICY",
                        "SPYNO_SHADOW_LEARN")}
    out = subprocess.run(
        [sys.executable, "-c",
         "from spyno_service.config import Config; c = Config(); "
         "print(c.shadow, c.warmstart_snapshot, c.policy, c.clip_min, c.clip_max, "
         "round(c.arm_floor, 6), c.shadow_learn, c.learning_frozen)"],
        cwd=ROOT, env={**env, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True,
        check=True).stdout.split()
    assert out == ["True", "None", "ts_hybrid", "0.05", "0.9", "0.016667", "False", "True"]


_SUBIDA_PADRAO = """
import json, sys
from fastapi.testclient import TestClient
from spyno_service.app import create_app
tok = {"Authorization": "Bearer t"}
with TestClient(create_app(), headers=tok) as c:
    k = c.get("/v1/contract").json()
    s = c.get("/v1/policy-state").json()
    d = c.post("/v1/decide", json={"user_key": "u",
        "decision_point": "2026-10-22T10:00:00-03:00", "eligible": True,
        "context": json.loads(sys.argv[1])}).json()
print(json.dumps({"contract": k, "state": s, "decide": d}))
"""


def _subida_padrao(tmp_path, db, **env_extra):
    """Sobe o app com a Config do ambiente (padrões reais), num processo limpo."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPYNO_")}
    env.update({"PYTHONPATH": str(ROOT), "SPYNO_DB_URL": f"sqlite+pysqlite:///{db}",
                "SPYNO_DECISION_TOKEN": "t", "SPYNO_SHADOW": "0",
                "SPYNO_PROPENSITY_SAMPLES": "50", **env_extra})
    out = subprocess.run([sys.executable, "-c", _SUBIDA_PADRAO, json.dumps(_context())],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_subida_com_padroes_p05_carrega_warm_start_e_ignora_persistida_p10(tmp_path):
    """Migração p10 -> p05: o artefato (sem dependência de clipping) carrega na
    nova policy_version; a posterior persistida da versão p10 fica registrada em
    skipped_persisted e não é carregada, sem derrubar a subida."""
    db = tmp_path / "mig.db"
    p10 = _subida_padrao(tmp_path, db, SPYNO_WARMSTART_SNAPSHOT=str(WARM),
                         SPYNO_CLIP_MIN="0.1", SPYNO_ARM_FLOOR=repr(1.0 / 30.0))
    assert p10["contract"]["max_importance_weight"] == pytest.approx(30.0)
    v10 = p10["state"]["policy_version"]
    # grava uma posterior persistida da linhagem p10 (aquecida)
    with _client(_app(db, warm=WARM, policy=bandit.ts_hybrid(
            clip=ClipConfig(0.1, 0.9, 0.1, 1.0 / 30.0), n_samples=50))) as c:
        _train_one(c, point="2026-10-22T11:00:00-03:00")
        assert c.post("/v1/end-day", json={"date": "2026-10-22"}).json()["ok"] is True
        v10_50 = c.get("/v1/policy-state").json()["policy_version"]

    p05 = _subida_padrao(tmp_path, db, SPYNO_WARMSTART_SNAPSHOT=str(WARM))
    k, s, d = p05["contract"], p05["state"], p05["decide"]
    assert (k["clip_min"], k["clip_max"], k["arm_mix"]) == (0.05, 0.9, 0.1)
    assert k["arm_floor"] == 0.05 / 3
    assert k["max_importance_weight"] == pytest.approx(60.0)
    assert k["posterior_init"] == "warmstart"
    assert k["contract_version"] == "2.3"
    assert s["posterior_source"] == "warmstart"
    assert s["warmstart"]["sha256"] == k["warmstart_sha256"]
    assert s["policy_version"] not in (v10, v10_50)
    assert k["policy_version"] == s["policy_version"]
    skipped = {v["policy_version"]: v["reason"] for v in s["skipped_persisted"]}
    assert skipped == {v10_50: "outra_policy_version"}
    # propensões da nova versão respeitam o piso p05
    assert d["fallback"] is False and d["policy_version"] == s["policy_version"]
    assert min(d["arm_probabilities"][1:]) >= 0.05 / 3 - 1e-12
    assert sum(d["arm_probabilities"][1:]) >= 0.05 - 1e-12


def test_warm_start_mantem_shadow(tmp_path):
    with _client(_app(tmp_path / "s.db", warm=WARM, shadow=True)) as c:
        assert c.get("/v1/contract").json()["shadow"] is True
        r = c.post("/v1/decide", json={"user_key": "u",
                                       "decision_point": "2026-10-22T10:00:00-03:00",
                                       "context": _context(), "eligible": True}).json()
    assert r["executed_action"] == 0 and r["shadow"] is True
