"""Esquema v2 da política: clipping, propensão, features, update e versão.

Cada bloco corresponde a um item da subida de `policy_version` para v2.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SPYNO_DB_URL", "sqlite+pysqlite:///:memory:")

from spyno_service.features import (DIM, FEATURES, NOTIFS_TODAY_SCALE, POLICY_SCHEMA,
                                    policy_obs, vectorize, vectorize_policy_space)
from spyno_service.policies import bandit
from spyno_service.policies.clipping import ClipConfig, ClippedPolicy, clip_distribution

TOKEN = "token-de-teste"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CLIP = ClipConfig(p_min=0.1, p_max=0.9, arm_mix=0.1)


def _context(hour=10, notifs_today=0, minutes_since_last=None, fbm_obs=0.7, **extra):
    return {
        "in_free_window": 1, "valence_pos": 0.5, "is_weekend": 0, "recency": 0.3,
        "already_trained": 0, "streak_norm": 0.2, "emp_adhesion": 0.4,
        "fatigue": 0.1, "notifs_today": notifs_today, "morning": 1,
        "midday": 0, "evening": 0, "fbm_obs": fbm_obs,
        "hour": hour, "minutes_since_last": minutes_since_last, **extra,
    }


class Recorder:
    """Política clipável de teste: vitórias fixas e registro dos updates."""

    name = "recorder"
    send_arms = (1, 2, 3)

    def __init__(self, wins=(0, 0, 0, 10)):
        self.wins = np.array(wins, dtype=float)
        self.updates = []
        self.end_days = 0

    def start_user(self, user_id): pass

    def arm_wins(self, obs, n):
        return self.wins

    def decide(self, user_id, obs, eligible, rng=None):  # não usado pelo ClippedPolicy
        raise AssertionError

    def update(self, user_id, obs, arm, reward):
        self.updates.append((arm, reward, obs))

    def end_day(self):
        self.end_days += 1

    def snapshot(self):
        return {"wins": self.wins.tolist()}


def _client(tmp_path, policy, **overrides):
    from fastapi.testclient import TestClient

    from spyno_service.app import create_app
    from spyno_service.config import Config
    from spyno_service.store import Store

    defaults = {"database_url": f"sqlite+pysqlite:///{tmp_path}/v2.db",
                "policy": "teste", "shadow": False, "decision_token": TOKEN}
    cfg = replace(Config(), **{**defaults, **overrides})
    return TestClient(create_app(policy=policy, store=Store(cfg.database_url), config=cfg),
                      headers=AUTH)


def _decide(c, user="u", point="2026-10-22T10:00:00-03:00", eligible=True, **ctx):
    r = c.post("/v1/decide", json={"user_key": user, "decision_point": point,
                                   "context": _context(**ctx), "eligible": eligible})
    assert r.status_code == 200, r.text
    return r.json()


def _feedback(c, decision_id, outcome="responded", value=1,
              at="2026-10-22T10:30:00-03:00"):
    r = c.post("/v1/feedback", json={"decision_id": decision_id, "observed_at": at,
                                     "outcome": outcome, "value": value})
    assert r.status_code == 200, r.text
    return r.json()


# ---- 1. clipping e propensão ------------------------------------------------

def test_clipping_limita_probabilidade_de_envio():
    tudo_envio = clip_distribution(np.array([0, 0, 0, 100.0]), (1, 2, 3), CLIP)
    nada_envio = clip_distribution(np.array([100.0, 0, 0, 0]), (1, 2, 3), CLIP)
    assert tudo_envio[1:].sum() == pytest.approx(0.9)
    assert nada_envio[1:].sum() == pytest.approx(0.1)
    for pi in (tudo_envio, nada_envio):
        assert pi.sum() == pytest.approx(1.0)
        assert (pi > 0).all(), "nenhum braço pode ter propensão nula"


def test_clipping_nao_mexe_no_interior():
    pi = clip_distribution(np.array([50.0, 10, 10, 30]), (1, 2, 3), CLIP)
    assert pi[0] == pytest.approx(0.5)
    # repartição: (1-η)·prop + η/3 sobre a massa de envio 0,5
    assert pi[3] == pytest.approx(0.5 * (0.9 * 0.6 + 0.1 / 3))
    assert pi[1] == pytest.approx(0.5 * (0.9 * 0.2 + 0.1 / 3))


def test_piso_por_braco_de_envio():
    pi = clip_distribution(np.array([0, 0, 0, 100.0]), (1, 2, 3), CLIP)
    piso = CLIP.p_min * CLIP.arm_mix / 3
    assert pi[1] >= piso and pi[2] >= piso
    sem_envio = clip_distribution(np.array([100.0, 0, 0, 0]), (1, 2, 3), CLIP)
    assert sem_envio[1] == pytest.approx(sem_envio[2]) == pytest.approx(sem_envio[3])


def test_binario_so_reparte_entre_os_bracos_que_usa():
    pi = clip_distribution(np.array([100.0, 0, 0, 0]), (3,), CLIP)
    assert pi[1] == pi[2] == 0.0
    assert pi[3] == pytest.approx(0.1)


@pytest.mark.parametrize("p_min,p_max,mix", [(0.0, 0.9, 0.1), (0.5, 0.4, 0.1),
                                             (0.1, 1.0, 0.1), (0.1, 0.9, 0.0)])
def test_clip_config_invalida(p_min, p_max, mix):
    with pytest.raises(ValueError):
        ClipConfig(p_min, p_max, mix)


# ---- 1b. piso explícito por braço de envio (contrato 2.1) -------------------

#: Distribuições do TS que estressam o piso: tudo num braço, nada de envio,
#: interior, empates, um braço de envio zerado.
_WINS = [(0, 0, 0, 100), (100, 0, 0, 0), (0, 100, 0, 0), (50, 10, 10, 30),
         (5, 90, 5, 0), (1, 1, 1, 1), (99, 0, 0, 1), (0, 33, 33, 34), (10, 0, 90, 0)]
_CLIPS = [ClipConfig(), ClipConfig(0.1, 0.9, 0.1, 0.0), ClipConfig(0.3, 0.9, 0.1, 0.1),
          ClipConfig(0.2, 0.8, 0.5, 0.05), ClipConfig(0.1, 0.9, 1.0, 1 / 30)]


def test_padrao_do_piso_e_p_min_sobre_tres():
    from spyno_service.config import Config

    assert ClipConfig().arm_floor == pytest.approx(0.1 / 3)
    assert Config().arm_floor == pytest.approx(ClipConfig().arm_floor)
    assert ClipConfig.from_config(Config()) == ClipConfig()


@pytest.mark.parametrize("clip", _CLIPS, ids=lambda c: c.tag())
@pytest.mark.parametrize("wins", _WINS)
def test_piso_soma_e_propensao_positiva(clip, wins):
    pi = clip_distribution(np.array(wins, dtype=float), (1, 2, 3), clip)
    assert pi.sum() == pytest.approx(1.0, abs=1e-12)
    assert (pi > 0).all(), "propensão nula quebra a OPE"
    assert pi[1:].min() >= clip.min_send_arm_prob(3) - 1e-12
    assert pi[1:].min() >= clip.arm_floor - 1e-12
    assert clip.p_min - 1e-12 <= pi[1:].sum() <= clip.p_max + 1e-12
    # o piso não mexe na massa de envio do passo 2
    raw = np.array(wins, dtype=float) / sum(wins)
    assert pi[1:].sum() == pytest.approx(np.clip(raw[1:].sum(), clip.p_min, clip.p_max))
    # peso de importância de qualquer alvo determinístico
    assert (1.0 / pi).max() <= clip.max_importance_weight(3) + 1e-9


def test_piso_eleva_so_os_bracos_abaixo_dele():
    sem = ClipConfig(0.1, 0.9, 0.1, 0.0)
    com = ClipConfig(0.1, 0.9, 0.1, 1 / 30)
    w = np.array([0, 0, 0, 100.0])
    a, b = clip_distribution(w, (1, 2, 3), sem), clip_distribution(w, (1, 2, 3), com)
    assert a[1] == pytest.approx(0.9 * 0.1 / 3)           # 0,03 < piso
    assert b[1] == pytest.approx(1 / 30) == pytest.approx(b[2])
    assert b[3] == pytest.approx(0.9 - 2 / 30)
    assert b[0] == pytest.approx(a[0])
    # interior longe do piso: nada muda
    w = np.array([50.0, 10, 10, 30])
    np.testing.assert_allclose(clip_distribution(w, (1, 2, 3), sem),
                               clip_distribution(w, (1, 2, 3), com))


def test_piso_na_massa_minima_reparte_uniforme():
    """Com f = p_min/3 e P(enviar) no mínimo, os três braços ficam exatamente em f.

    É o custo do padrão: no piso de envio a política não expressa preferência
    entre braços.
    """
    pi = clip_distribution(np.array([100.0, 0, 0, 5]), (1, 2, 3), ClipConfig())
    np.testing.assert_allclose(pi[1:], 1 / 30)


def test_peso_maximo_reportado():
    assert ClipConfig(0.1, 0.9, 0.1, 0.0).max_importance_weight(3) == pytest.approx(300.0)
    assert ClipConfig().max_importance_weight(3) == pytest.approx(30.0)
    assert ClipConfig(0.3, 0.9, 0.1, 0.1).max_importance_weight(3) == pytest.approx(10.0)
    # binário: o único braço de envio tem p_min; o pior caso é a abstenção
    assert ClipConfig().max_importance_weight(1) == pytest.approx(10.0)


@pytest.mark.parametrize("floor", [-0.01, 0.04, 0.1])
def test_piso_invalido(floor):
    with pytest.raises(ValueError, match="arm_floor"):
        ClipConfig(0.1, 0.9, 0.1, floor)


def test_piso_maior_exige_subir_p_min():
    ClipConfig(0.3, 0.9, 0.1, 0.1)          # válido


def test_sorteio_respeita_o_piso():
    pol = ClippedPolicy(Recorder(wins=(0, 0, 0, 10)), ClipConfig(), n_samples=1, seed=5)
    arms = np.array([pol.decide_with_distribution("u", _context(), True)[0]
                     for _ in range(20000)])
    for k in (1, 2):
        assert np.mean(arms == k) == pytest.approx(1 / 30, abs=0.006)


def test_piso_muda_a_policy_version():
    from spyno_service.policies.registry import version

    a = version(bandit.ts_hybrid(clip=ClipConfig(0.1, 0.9, 0.1, 0.0), n_samples=100))
    b = version(bandit.ts_hybrid(clip=ClipConfig(), n_samples=100))
    assert a != b
    assert a.startswith("ts_hybrid-v2-") and b.startswith("ts_hybrid-v2-")


def test_ts_binary_e_ts_disjoint_tem_versoes_distintas():
    from spyno_service.policies.registry import version

    assert version(bandit.ts_binary(clip=CLIP, n_samples=10)) != \
        version(bandit.ts_disjoint(clip=CLIP, n_samples=10))


def test_acao_e_sorteada_da_distribuicao_registrada():
    pol = ClippedPolicy(Recorder(wins=(0, 0, 0, 10)), CLIP, n_samples=1, seed=3)
    arms = [pol.decide_with_distribution("u", _context(), True)[0] for _ in range(4000)]
    freq_envio = np.mean(np.array(arms) > 0)
    assert freq_envio == pytest.approx(0.9, abs=0.02)
    _, pi = pol.decide_with_distribution("u", _context(), True)
    assert pi[1:].sum() == pytest.approx(0.9)


def test_ts_hybrid_com_posterior_contra_envio_fica_no_piso():
    pol = bandit.ts_hybrid(clip=CLIP, n_samples=500)
    inner = pol._inner._inner
    # posterior concentrada: enviar derruba muito a resposta
    from spyno_decision.hybrid import HDIM
    from spyno_decision.features import DIM as DDIM
    inner.m = np.zeros(HDIM)
    inner.m[DDIM] = -20.0          # bias do bloco w_send
    inner.q = np.full(HDIM, 1e6)
    _, pi = pol.decide_with_distribution("u", _context(), True)
    assert pi[1:].sum() == pytest.approx(0.1)
    assert (pi > 0).all()


def test_decide_registra_propensao_exata_da_acao_escolhida(tmp_path):
    pol = bandit.ts_hybrid(clip=CLIP, n_samples=500)
    with _client(tmp_path, pol) as c:
        for h in range(8, 20):
            body = _decide(c, user="p", point=f"2026-10-22T{h:02d}:00:00-03:00", hour=h,
                           in_free_window=1)
            pi = body["arm_probabilities"]
            assert body["propensity_method"] == "exact"
            assert body["recommended_propensity"] == pytest.approx(pi[body["recommended_action"]])
            assert body["recommended_propensity"] > 0
            assert 0.1 - 1e-9 <= sum(pi[1:]) <= 0.9 + 1e-9
            if body["guardrail_reason"] is None:
                assert body["propensity"] == pytest.approx(pi[body["executed_action"]])
        row = c.app.state.ctx["store"].get(body["decision_id"])
        assert row["recommended_propensity"] == pytest.approx(body["recommended_propensity"])
        assert json.loads(row["arm_probs"]) == pytest.approx(pi)


def test_abstencao_forcada_tem_propensao_um(tmp_path):
    pol = ClippedPolicy(Recorder(wins=(0, 0, 0, 10)), CLIP, n_samples=1, seed=0)
    with _client(tmp_path, pol) as c:
        body = _decide(c, hour=23, point="2026-10-22T23:00:00-03:00")
    assert body["executed_action"] == 0
    assert body["propensity"] == 1.0
    assert body["recommended_propensity"] < 1.0


def test_latencia_do_decide_ts_hybrid(tmp_path):
    from spyno_service.config import CONFIG

    pol = bandit.ts_hybrid(clip=CLIP, n_samples=CONFIG.propensity_samples)
    with _client(tmp_path, pol) as c:
        _decide(c, user="aquecimento")
        t0 = time.perf_counter()
        for i in range(20):
            body = _decide(c, user=f"lat{i}")
            assert body["fallback"] is False
            assert body["latency_ms"] < 500
        media = (time.perf_counter() - t0) / 20 * 1000
    assert media < 200, f"decide médio {media:.0f} ms"


# ---- 2. fbm_obs fora do vetor -----------------------------------------------

def test_fbm_obs_fora_do_vetor_da_politica():
    assert "fbm_obs" not in FEATURES
    assert DIM == len(FEATURES) + 1
    assert "fbm_obs" not in policy_obs(_context())


def test_fbm_obs_nao_altera_a_decisao_do_ts():
    a = bandit.ts_hybrid(clip=CLIP, n_samples=300, seed=5)
    b = bandit.ts_hybrid(clip=CLIP, n_samples=300, seed=5)
    for pol in (a, b):
        pol._inner._inner.m[:] = np.random.default_rng(1).normal(size=pol._inner._inner.m.size)
    wa = a._inner.arm_wins(_context(fbm_obs=0.0), 300)
    wb = b._inner.arm_wins(_context(fbm_obs=1.0), 300)
    np.testing.assert_array_equal(wa, wb)


def test_fbm_obs_continua_no_log_e_na_referencia(tmp_path):
    ref = bandit.fbm_threshold_observable(threshold=0.4)
    assert ref.decide("u", _context(fbm_obs=0.9), True)[0] == 3
    assert ref.decide("u", _context(fbm_obs=0.1), True)[0] == 0
    pol = ClippedPolicy(Recorder(), CLIP, n_samples=1)
    with _client(tmp_path, pol) as c:
        body = _decide(c, fbm_obs=0.77)
        row = c.app.state.ctx["store"].get(body["decision_id"])
    assert json.loads(row["context"])["fbm_obs"] == 0.77


# ---- 3. notifs_today ÷ 5 ----------------------------------------------------

def test_vectorize_normaliza_notifs_today():
    i = 1 + FEATURES.index("notifs_today")
    assert vectorize(_context(notifs_today=2))[i] == pytest.approx(2 / NOTIFS_TODAY_SCALE)
    assert NOTIFS_TODAY_SCALE == 5.0
    # quem já está no espaço da política (simulador) não é dividido de novo
    assert vectorize_policy_space({**_context(), "notifs_today": 0.4})[i] == pytest.approx(0.4)


def test_adaptador_entrega_notifs_today_normalizado_ao_spyno_decision():
    vistos = []

    class Espiao:
        name = "espiao"
        def update(self, user_id, obs, arm, reward): vistos.append(obs)

    from spyno_service.policies.bandit import _CompatibilidadeHerdada
    _CompatibilidadeHerdada(Espiao()).update("u", _context(notifs_today=3), 0, 1)
    obs = vistos[0]
    assert obs["notifs_today"] == pytest.approx(0.6)
    assert obs["fbm_obs"] == 0.0 and obs["location_home"] == 0.0


def test_normalizacao_igual_a_do_simulador():
    sim = Path(os.getenv("SPYNO_SIM_PATH", Path(__file__).resolve().parents[2] / "spyno-sim"))
    if not sim.is_dir():
        pytest.skip("spyno-sim não disponível")
    if str(sim) not in sys.path:
        sys.path.insert(0, str(sim))
    response = pytest.importorskip("spyno_sim.response")
    state = response.UserState()
    state.notifications_today = 2
    # o divisor do simulador, lido do código e não copiado
    import inspect
    src = inspect.getsource(response.obs_features)
    assert '"notifs_today": state.notifications_today / 5.0' in src
    assert policy_obs(_context(notifs_today=state.notifications_today))["notifs_today"] \
        == pytest.approx(state.notifications_today / 5.0)


def test_guardrail_continua_com_contagem_bruta(tmp_path):
    pol = ClippedPolicy(Recorder(), CLIP, n_samples=1)
    with _client(tmp_path, pol, daily_cap=3) as c:
        body = _decide(c, notifs_today=3)
    assert body["guardrail_reason"] == "daily_cap"


# ---- 4. update só em ponto elegível; semântica do feedback -----------------

def test_update_em_ponto_elegivel(tmp_path):
    rec = Recorder()
    with _client(tmp_path, ClippedPolicy(rec, CLIP, n_samples=1)) as c:
        body = _decide(c)
        fb = _feedback(c, body["decision_id"])
    assert body["update_eligible"] is True
    assert fb["trained"] is True
    assert rec.updates[0][:2] == (body["executed_action"], 1)


def test_sem_update_em_ponto_inelegivel(tmp_path):
    rec = Recorder()
    with _client(tmp_path, ClippedPolicy(rec, CLIP, n_samples=1)) as c:
        body = _decide(c, eligible=False)
        fb = _feedback(c, body["decision_id"])
    assert body["update_eligible"] is False
    assert body["recommended_propensity"] == 1.0
    assert fb["applied"] is True and fb["trained"] is False
    assert fb["train_skip_reason"] == "ponto_nao_elegivel"
    assert rec.updates == []


def test_sem_update_quando_guardrail_bloqueia(tmp_path):
    rec = Recorder()
    with _client(tmp_path, ClippedPolicy(rec, CLIP, n_samples=1)) as c:
        body = _decide(c, hour=23, point="2026-10-22T23:00:00-03:00")
        _feedback(c, body["decision_id"], at="2026-10-22T23:30:00-03:00")
    assert body["update_eligible"] is False
    assert rec.updates == []


def test_sem_update_em_contingencia(tmp_path):
    class Quebra(Recorder):
        def arm_wins(self, obs, n):
            raise RuntimeError("x")

    rec = Quebra()
    with _client(tmp_path, ClippedPolicy(rec, CLIP, n_samples=1)) as c:
        body = _decide(c)
        _feedback(c, body["decision_id"])
    assert body["fallback"] is True and body["update_eligible"] is False
    assert rec.updates == []


def test_not_responded_treina_com_zero_e_censura_nao_treina(tmp_path):
    rec = Recorder(wins=(10, 0, 0, 0))
    with _client(tmp_path, ClippedPolicy(rec, CLIP, n_samples=1)) as c:
        a = _decide(c, user="a")
        b = _decide(c, user="b")
        d = _decide(c, user="d")
        assert c.get("/health").json()["pending_outcomes"] == 3
        fa = _feedback(c, a["decision_id"], outcome="not_responded", value=0)
        fb = _feedback(c, b["decision_id"], outcome="censored", value=0)
        fd = _feedback(c, d["decision_id"], outcome="delivery_failed", value=0)
        assert c.get("/health").json()["pending_outcomes"] == 0
    assert fa["trained"] is True
    assert fb["trained"] is False and fb["train_skip_reason"] == "sem_treino_censored"
    assert fd["trained"] is False
    assert [(arm, y) for arm, y, _ in rec.updates] == [(a["executed_action"], 0)]


def test_braco_zero_tambem_treina(tmp_path):
    rec = Recorder(wins=(10, 0, 0, 0))
    with _client(tmp_path, ClippedPolicy(rec, CLIP, n_samples=1)) as c:
        for h in range(8, 20):
            body = _decide(c, user="z", point=f"2026-10-22T{h:02d}:00:00-03:00", hour=h)
            if body["executed_action"] == 0:
                break
        assert body["executed_action"] == 0
        _feedback(c, body["decision_id"], at=f"2026-10-22T{h:02d}:30:00-03:00")
    assert rec.updates[-1][:2] == (0, 1)


# ---- shadow: contadores propostos e update só com a ação executada ----------

def test_shadow_atualiza_so_com_a_acao_executada(tmp_path):
    rec = Recorder(wins=(0, 0, 0, 10))
    with _client(tmp_path, ClippedPolicy(rec, CLIP, n_samples=1, seed=1), shadow=True) as c:
        propostas = []
        for h in range(8, 20):
            body = _decide(c, user="s", point=f"2026-10-22T{h:02d}:00:00-03:00", hour=h,
                           notifs_today_proposed=0, fatigue_proposed=0.0)
            assert body["executed_action"] == 0
            assert body["propensity"] == 1.0
            propostas.append(body["recommended_action"])
            _feedback(c, body["decision_id"], at=f"2026-10-22T{h:02d}:30:00-03:00")
    assert any(p > 0 for p in propostas), "a política propôs envios"
    assert rec.updates, "o desfecho em shadow é dado legítimo do braço 0"
    assert all(arm == 0 for arm, _, _ in rec.updates), "a ação proposta nunca treina"


def test_shadow_usa_contadores_propostos_e_registra_os_dois(tmp_path):
    rec = Recorder()
    with _client(tmp_path, ClippedPolicy(rec, CLIP, n_samples=1), shadow=True,
                 daily_cap=3) as c:
        body = _decide(c, notifs_today=0, notifs_today_proposed=3,
                       fatigue_proposed=0.4, minutes_since_last_proposed=30)
        row = c.app.state.ctx["store"].get(body["decision_id"])
        sem = _decide(c, user="sem")
    assert body["counter_basis"] == "proposed"
    assert body["guardrail_reason"] == "daily_cap", "guardrail sobre o que teria sido enviado"
    bruto, visto = json.loads(row["context"]), json.loads(row["policy_context"])
    assert bruto["notifs_today"] == 0 and bruto["notifs_today_proposed"] == 3
    assert visto["notifs_today"] == 3 and visto["fatigue"] == 0.4
    assert visto["minutes_since_last"] == 30
    assert sem["counter_basis"] == "executed"


def test_fora_do_shadow_ignora_contadores_propostos(tmp_path):
    with _client(tmp_path, ClippedPolicy(Recorder(), CLIP, n_samples=1)) as c:
        body = _decide(c, notifs_today=0, notifs_today_proposed=3, fatigue_proposed=0.4)
    assert body["counter_basis"] == "executed"
    assert body["guardrail_reason"] is None


# ---- 5. versão da política e estado antigo ----------------------------------

def test_policy_version_carrega_o_esquema(tmp_path):
    from spyno_service.policies.registry import version

    v = version(bandit.ts_hybrid(clip=CLIP, n_samples=100))
    assert POLICY_SCHEMA == "v2"
    assert v.startswith("ts_hybrid-v2-")
    outra = version(bandit.ts_hybrid(clip=ClipConfig(0.2, 0.8, 0.1), n_samples=100))
    assert outra != v, "o clipping faz parte da identidade"
    with _client(tmp_path, ClippedPolicy(Recorder(), CLIP, n_samples=1)) as c:
        info = c.get("/v1/contract", headers={}).json()
    assert info["policy_version"].split("-")[1] == "v2"


def test_snapshot_v1_e_recusado(tmp_path):
    antigo = bandit.ts_hybrid(clip=CLIP, n_samples=10)._inner._inner
    snap = {"m": antigo.m.tolist(), "q": antigo.q.tolist(), "cost": 0.05}  # sem schema
    p = tmp_path / "v1.json"
    p.write_text(json.dumps(snap))
    with pytest.raises(bandit.IncompatibleSnapshotError):
        bandit.ts_hybrid(snapshot=p, clip=CLIP, n_samples=10)


def test_snapshot_v2_ida_e_volta(tmp_path):
    pol = bandit.ts_hybrid(clip=CLIP, n_samples=10)
    pol._inner._inner.m[:] = 0.3
    snap = pol.snapshot()
    assert snap["schema"] == "v2" and snap["features"] == list(FEATURES)
    p = tmp_path / "v2.json"
    p.write_text(json.dumps(snap))
    volta = bandit.ts_hybrid(snapshot=p, clip=CLIP, n_samples=10)
    np.testing.assert_allclose(volta._inner._inner.m, 0.3)


def test_desfecho_de_decisao_de_versao_antiga_nao_treina(tmp_path):
    rec = Recorder()
    with _client(tmp_path, ClippedPolicy(rec, CLIP, n_samples=1)) as c:
        store = c.app.state.ctx["store"]
        store.insert({
            "decision_id": "antiga", "user_key": "u",
            "decision_point": datetime(2026, 10, 1, 13, tzinfo=timezone.utc),
            "context": _context(), "recommended_action": 3, "executed_action": 3,
            "guardrail_reason": None, "propensity": 0.5, "propensity_method": "monte-carlo",
            "policy": "ts_hybrid", "policy_version": "ts_hybrid-abc123", "shadow": False,
            "fallback": False, "fallback_reason": None, "latency_ms": 1.0,
        }, 180)
        fb = _feedback(c, "antiga", at="2026-10-01T13:30:00+00:00")
    assert fb["applied"] is True and fb["trained"] is False
    assert fb["train_skip_reason"] == "policy_version_diferente"
    assert rec.updates == []


def test_banco_v1_e_migrado_explicitamente(tmp_path):
    from sqlalchemy import create_engine, inspect, text

    from spyno_service.store import V2_COLUMNS, Store

    url = f"sqlite+pysqlite:///{tmp_path}/antigo.db"
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE decisions (decision_id VARCHAR(64) PRIMARY KEY, "
            "user_key VARCHAR(128) NOT NULL, decision_point DATETIME NOT NULL, "
            "context TEXT NOT NULL, recommended_action INTEGER NOT NULL, "
            "executed_action INTEGER NOT NULL, guardrail_reason VARCHAR(32), "
            "propensity FLOAT NOT NULL, propensity_method VARCHAR(16) NOT NULL, "
            "policy VARCHAR(64) NOT NULL, policy_version VARCHAR(128) NOT NULL, "
            "shadow BOOLEAN NOT NULL, fallback BOOLEAN NOT NULL, "
            "fallback_reason VARCHAR(64), latency_ms FLOAT NOT NULL, "
            "created_at DATETIME NOT NULL, outcome VARCHAR(24), outcome_value INTEGER, "
            "outcome_at DATETIME, window_closes_at DATETIME NOT NULL)"))
    store = Store(url)
    assert sorted(store.migrated_columns) == sorted(V2_COLUMNS)
    cols = {c["name"] for c in inspect(store.engine).get_columns("decisions")}
    assert set(V2_COLUMNS) <= cols
    assert Store(url).migrated_columns == []


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "checkpoints"
                         / "ppo_observable.zip").exists(), reason="sem checkpoint")
def test_checkpoint_ppo_v1_e_recusado():
    pytest.importorskip("stable_baselines3")
    from spyno_service.policies.ppo import PPOPolicy

    ckpt = Path(__file__).resolve().parents[1] / "checkpoints" / "ppo_observable.zip"
    with pytest.raises(ValueError, match="dimensões"):
        PPOPolicy(checkpoint=ckpt).warmup()


# ---- contrato com o backend: auth, /v1/contract, dedup por hora, end-day ----

def test_auth_exige_bearer(tmp_path):
    with _client(tmp_path, ClippedPolicy(Recorder(), CLIP, n_samples=1)) as c:
        payload = {"user_key": "a", "decision_point": "2026-10-22T10:00:00-03:00",
                   "context": _context()}
        assert c.post("/v1/decide", json=payload, headers={"Authorization": ""}).status_code == 401
        assert c.post("/v1/decide", json=payload,
                      headers={"Authorization": "Bearer errado"}).status_code == 401
        assert c.post("/v1/decide", json=payload).status_code == 200
        assert c.get("/health", headers={"Authorization": ""}).status_code == 200
        assert c.get("/v1/contract", headers={"Authorization": ""}).status_code == 200


def test_auth_sem_token_configurado_falha_fechado(tmp_path):
    with _client(tmp_path, ClippedPolicy(Recorder(), CLIP, n_samples=1),
                 decision_token=None) as c:
        assert c.post("/v1/end-day", json={"date": "2026-10-22"}).status_code == 503
        assert c.get("/v1/contract").status_code == 200
    with _client(tmp_path, ClippedPolicy(Recorder(), CLIP, n_samples=1),
                 decision_token=None, auth_disabled=True) as c:
        assert c.post("/v1/end-day", json={"date": "2026-10-22"}).status_code == 200


def test_contrato_expoe_janela_e_versao(tmp_path):
    with _client(tmp_path, ClippedPolicy(Recorder(), CLIP, n_samples=1),
                 outcome_window_min=180) as c:
        info = c.get("/v1/contract").json()
        assert info["outcome_window_min"] == 180
        assert info["policy_version"] == c.app.state.ctx["policy_version"]
        assert info["features"] == list(FEATURES)
        assert info["clip_min"] == pytest.approx(0.1)
        assert info["contract_version"] == "2.1"
        assert info["arm_floor"] == pytest.approx(1 / 30)
        assert info["max_importance_weight"] == pytest.approx(30.0)


def test_decision_id_deduplica_pela_hora(tmp_path):
    with _client(tmp_path, ClippedPolicy(Recorder(), CLIP, n_samples=1)) as c:
        a = _decide(c, point="2026-10-22T10:00:00.123-03:00")
        b = _decide(c, point="2026-10-22T10:47:12.999-03:00")
        d = _decide(c, point="2026-10-22T11:00:00-03:00")
        assert c.get("/health").json()["decisions"] == 2
    assert a["decision_id"] == b["decision_id"]
    assert a == b, "o segundo tick devolve a decisão gravada"
    assert d["decision_id"] != a["decision_id"]


def test_end_day_idempotente_por_data(tmp_path):
    pol = bandit.ts_hybrid(clip=CLIP, n_samples=100)
    inner = pol._inner._inner
    with _client(tmp_path, pol) as c:
        body = _decide(c)
        _feedback(c, body["decision_id"])
        assert len(inner._day_buf) == 1
        m0 = inner.m.copy()
        r1 = c.post("/v1/end-day", json={"date": "2026-10-22"}).json()
        m1 = inner.m.copy()
        assert r1["ok"] is True and r1["updates_applied"] == 1 and not r1["already_closed"]
        assert inner._day_buf == [], "estado pendente limpo"
        assert not np.allclose(m0, m1)
        # segunda chamada no mesmo dia não faz nada, mesmo com update novo no buffer
        body2 = _decide(c, point="2026-10-22T12:00:00-03:00", hour=12)
        _feedback(c, body2["decision_id"], at="2026-10-22T12:30:00-03:00")
        r2 = c.post("/v1/end-day", json={"date": "2026-10-22"}).json()
        assert r2["already_closed"] is True
        np.testing.assert_array_equal(inner.m, m1)
        r3 = c.post("/v1/end-day", json={"date": str(date(2026, 10, 23))}).json()
        assert r3["already_closed"] is False and r3["updates_applied"] == 1


# ---- ação proposta: semântica única com o backend ---------------------------

def test_acao_proposta_e_a_recomendada_apos_os_guardrails(tmp_path):
    """proposed_action = o que teria sido executado sem o shadow.

    É a base dos contadores *_proposed no backend (Notificacao.acao_proposta).
    Recomendação bloqueada por guardrail não teria saído: proposta 0.
    """
    rec = Recorder(wins=(0, 0, 0, 10))
    with _client(tmp_path, ClippedPolicy(rec, CLIP, n_samples=1, seed=3), shadow=True,
                 daily_cap=3, cooldown_min=90) as c:
        livres, bloqueadas = [], []
        for h in range(8, 20):
            livres.append(_decide(c, user="livre", point=f"2026-10-22T{h:02d}:00:00-03:00",
                                  hour=h, notifs_today_proposed=0, fatigue_proposed=0.0))
        for user, ctx in (("silencio", {"hour": 23}),
                          ("cota", {"notifs_today_proposed": 3}),
                          ("intervalo", {"minutes_since_last_proposed": 30})):
            for h in range(8, 20):
                base = {"hour": h, "notifs_today_proposed": 0, "fatigue_proposed": 0.0}
                bloqueadas.append(_decide(c, user=user,
                                          point=f"2026-10-22T{h:02d}:00:00-03:00",
                                          **{**base, **ctx}))
        repetida = _decide(c, user="livre", point="2026-10-22T08:00:00-03:00", hour=8,
                           notifs_today_proposed=0, fatigue_proposed=0.0)

    assert any(b["recommended_action"] > 0 for b in livres)
    for b in livres:
        assert b["executed_action"] == 0 and b["guardrail_reason"] == "shadow"
        assert b["proposed_action"] == b["recommended_action"]
    assert any(b["recommended_action"] > 0 for b in bloqueadas)
    for b in bloqueadas:
        if b["recommended_action"] > 0:
            assert b["guardrail_reason"] in ("quiet_hours", "daily_cap", "cooldown")
        assert b["proposed_action"] == 0, "bloqueada por guardrail não teria saído"
    assert repetida["proposed_action"] == livres[0]["proposed_action"], "idempotência"


def test_acao_proposta_fora_do_shadow_e_a_executada(tmp_path):
    with _client(tmp_path, ClippedPolicy(Recorder(), CLIP, n_samples=1, seed=3)) as c:
        livre = _decide(c)
        silencio = _decide(c, user="noite", hour=23)
    assert livre["proposed_action"] == livre["executed_action"]
    assert silencio["proposed_action"] == silencio["executed_action"] == 0


def test_acao_proposta_reconstruida_para_linha_antiga():
    from spyno_service.app import _proposed_action

    base = {"recommended_action": 2, "executed_action": 0, "proposed_action": None}
    assert _proposed_action({**base, "shadow": True, "guardrail_reason": "shadow"}) == 2
    assert _proposed_action({**base, "shadow": True, "guardrail_reason": "cooldown"}) == 0
    assert _proposed_action({**base, "shadow": False, "guardrail_reason": "quiet_hours"}) == 0
