"""Idempotência do /v1/decide sob chamadas concorrentes.

Duas chamadas idênticas que chegam juntas (um retry do backend por timeout, por
exemplo) passavam as duas pelo `store.get` sem achar a decisão, amostravam as
duas e a segunda estourava a chave primária no INSERT: resposta 500. Aqui a
segunda tem de devolver a decisão gravada pela primeira, sem reamostrar no
mesmo processo, e uma corrida entre processos tem de terminar na decisão que
chegou primeiro ao banco.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime

import pytest

os.environ.setdefault("SPYNO_DB_URL", "sqlite+pysqlite:///:memory:")

TOKEN = "token-de-teste"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

PAYLOAD = {
    "user_key": "u-corrida",
    "decision_point": "2026-09-17T10:00:00-03:00",
    "context": {
        "in_free_window": 1, "valence_pos": 0.5, "is_weekend": 0, "recency": 0.3,
        "already_trained": 0, "streak_norm": 0.2, "emp_adhesion": 0.4,
        "fatigue": 0.1, "notifs_today": 0, "morning": 1,
        "midday": 0, "evening": 0, "fbm_obs": 0.7,
        "hour": 10, "minutes_since_last": None,
    },
    "eligible": True,
}


class SeqPolicy:
    """Recomenda um braço diferente a cada chamada: reamostrar fica visível."""

    name = "stub"

    def __init__(self, arms=(3, 1, 2), delay=0.0):
        self.arms = arms
        self.delay = delay
        self.decide_calls = 0
        self._lock = threading.Lock()

    def start_user(self, user_id):
        pass

    def decide(self, user_id, obs, eligible, rng=None):
        with self._lock:
            arm = self.arms[self.decide_calls % len(self.arms)]
            self.decide_calls += 1
        if self.delay:
            time.sleep(self.delay)
        return (arm, 1.0) if eligible else (0, 1.0)

    def update(self, user_id, obs, arm, reward):
        pass

    def end_day(self):
        pass

    def estimate_propensities(self, obs, n_samples=64):
        import numpy as np
        return np.array([0.4, 0.2, 0.2, 0.2])


def _app(db_url, policy, store=None):
    # timeout folgado: o atraso da política simula uma amostragem lenta, não um
    # estouro de prazo (que viraria contingência)
    from spyno_service.app import create_app
    from spyno_service.config import Config
    from spyno_service.store import Store

    cfg = replace(Config(), database_url=db_url, policy="stub", shadow=False,
                  decision_token=TOKEN, decide_timeout_s=10.0)
    return create_app(policy=policy, store=store or Store(db_url), config=cfg)


def test_chamadas_concorrentes_no_mesmo_processo_nao_reamostram(tmp_path):
    """Duas threads com o mesmo ponto: uma amostra, a outra devolve o gravado."""
    from fastapi.testclient import TestClient

    pol = SeqPolicy(delay=0.3)
    app = _app(f"sqlite+pysqlite:///{tmp_path}/t.db", pol)
    results: list = [None, None]
    start = threading.Barrier(2)

    with TestClient(app, headers=AUTH) as c:
        def call(i):
            start.wait()
            r = c.post("/v1/decide", json=PAYLOAD)
            results[i] = (r.status_code, r.json())

        threads = [threading.Thread(target=call, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert [s for s, _ in results] == [200, 200], results
        assert results[0][1] == results[1][1]
        assert pol.decide_calls == 1, "a segunda chamada reamostrou"
        # e a repetição sequencial devolve a mesma coisa
        assert c.post("/v1/decide", json=PAYLOAD).json() == results[0][1]
        # os locks por ponto não se acumulam
        assert len(c.app.state.ctx["decide_locks"]) == 0


def test_corrida_entre_processos_devolve_a_decisao_gravada(tmp_path):
    """Outro processo grava entre o get e o insert: a resposta é a dele.

    Injeção determinística: o primeiro `get` deste processo dispara, antes de
    responder "não existe", a mesma decisão numa segunda instância sobre o
    mesmo banco (o que um outro processo ou réplica faria). Este processo então
    amostra um braço diferente e o INSERT viola a chave primária.
    """
    from fastapi.testclient import TestClient

    from spyno_service.store import Store

    url = f"sqlite+pysqlite:///{tmp_path}/t.db"
    other_pol = SeqPolicy(arms=(1,))
    other = TestClient(_app(url, other_pol), headers=AUTH)

    class RacyStore(Store):
        fired = False
        other_body = None

        def get(self, decision_id):
            if not self.fired:
                RacyStore.fired = True
                r = other.post("/v1/decide", json=PAYLOAD)
                assert r.status_code == 200, r.text
                RacyStore.other_body = r.json()
                return None
            return super().get(decision_id)

    pol = SeqPolicy(arms=(3,))
    with other, TestClient(_app(url, pol, store=RacyStore(url)), headers=AUTH) as c:
        r = c.post("/v1/decide", json=PAYLOAD)
        assert r.status_code == 200, r.text
        body = r.json()
        assert RacyStore.fired and pol.decide_calls == 1
        # a decisão que vale é a que chegou primeiro ao banco (braço 1), não a
        # amostrada aqui (braço 3), e a resposta é idêntica à do outro processo
        assert body == RacyStore.other_body
        assert body["recommended_action"] == 1
        assert c.post("/v1/decide", json=PAYLOAD).json() == body
        rows = c.app.state.ctx["store"].get(body["decision_id"])
        assert rows["recommended_action"] == 1


def test_insert_so_engole_a_violacao_da_propria_chave(tmp_path):
    """Outra violação de integridade (coluna obrigatória nula) continua subindo."""
    from sqlalchemy.exc import IntegrityError

    from spyno_service.store import Store

    store = Store(f"sqlite+pysqlite:///{tmp_path}/t.db")
    with pytest.raises(IntegrityError):
        store.insert({"decision_id": "x", "user_key": None,
                      "decision_point": datetime(2026, 9, 17, 13),
                      "context": {}, "recommended_action": 0, "executed_action": 0,
                      "propensity": 1.0, "propensity_method": "exact", "policy": "p",
                      "policy_version": "v", "shadow": False, "fallback": False,
                      "latency_ms": 0.0})


def _db_urls():
    urls = ["sqlite"]
    if os.getenv("SPYNO_TEST_PG_URL"):
        urls.append("postgres")
    return urls


@pytest.mark.parametrize("backend", _db_urls())
def test_duas_instancias_concorrentes_sobre_o_mesmo_banco(tmp_path, backend):
    """Duas instâncias (sem lock em comum) decidem o mesmo ponto ao mesmo tempo.

    As duas amostram (braços diferentes) e uma perde o INSERT. As duas respostas
    têm de ser a decisão que ficou no banco. Com SPYNO_TEST_PG_URL roda também
    no PostgreSQL, onde o INSERT perdedor espera o commit do vencedor. O teste
    não toca nas tabelas existentes desse banco: cria um schema temporário de
    nome aleatório, aponta o search_path para ele e o apaga no fim.
    """
    if backend == "sqlite":
        _corrida_entre_instancias(f"sqlite+pysqlite:///{tmp_path}/t.db")
        return
    with _schema_temporario(os.environ["SPYNO_TEST_PG_URL"]) as url:
        _corrida_entre_instancias(url)


@contextmanager
def _schema_temporario(base_url: str):
    """Schema descartável no PostgreSQL; devolve a URL com search_path nele."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    schema = f"spyno_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(base_url, future=True)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    base = make_url(base_url)
    url = base.update_query_dict({"options": f"-csearch_path={schema}"})
    try:
        yield url.render_as_string(hide_password=False)
    finally:
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def _corrida_entre_instancias(url: str) -> None:
    from fastapi.testclient import TestClient

    pols = [SeqPolicy(arms=(1,), delay=0.2), SeqPolicy(arms=(3,), delay=0.2)]
    clients = [TestClient(_app(url, p), headers=AUTH) for p in pols]
    results: list = [None, None]
    start = threading.Barrier(2)

    def call(i):
        start.wait()
        r = clients[i].post("/v1/decide", json=PAYLOAD)
        results[i] = (r.status_code, r.json())

    with clients[0], clients[1]:
        threads = [threading.Thread(target=call, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert [s for s, _ in results] == [200, 200], results
        assert results[0][1] == results[1][1]
        assert [p.decide_calls for p in pols] == [1, 1]
        row = clients[0].app.state.ctx["store"].get(results[0][1]["decision_id"])
        assert row["recommended_action"] == results[0][1]["recommended_action"]
    for c in clients:
        c.app.state.ctx["store"].engine.dispose()
