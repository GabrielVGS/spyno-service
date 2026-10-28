#!/usr/bin/env python3
"""Demo F4 "dia simulado": backend real + spyno-service real + Postgres.

Sobe o Compose da demo (projeto `spyno-demo-f4`, portas fora das padrão),
aplica as migrations do backend, cria usuários sintéticos com rotinas, liga o
agendador do backend com o relógio de demonstração (1 hora simulada a cada
SPYNO_TICK_MS reais), simula treinos concluídos ao longo de um dia inteiro e a
virada do dia (end-day), exporta o log de Notificacao e o registro do serviço
para demo/out/ e confere o contrato de ponta a ponta.

Uso (na raiz do spyno-service):

    python3 demo/run_demo.py                 # shadow (padrão)
    python3 demo/run_demo.py --sem-shadow    # envio real, entrega FCM simulada
    python3 demo/run_demo.py --manter        # não derruba os contêineres no fim
    python3 demo/run_demo.py --warmstart --reiniciar   # warm-start + reinício

Em shadow (padrão) a posterior fica congelada (contrato 2.3): a demo confere que
o digest não muda no dia nem no end-day e, com --reiniciar, que o serviço
reiniciado em shadow e depois recriado fora do shadow carrega exatamente a
posterior inicial. Com --sem-shadow, a posterior muda no end-day e é a
persistida que volta no reinício.

Só usa a biblioteca padrão. Requer docker com compose v2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPOSE_FILE = HERE / "docker-compose.demo.yml"
PROJECT = "spyno-demo-f4"
OUT = HERE / "out"  # + /shadow ou /sem-shadow

#: America/Sao_Paulo não tem horário de verão desde 2019: -03:00 fixo.
SP = timezone(timedelta(hours=-3))
HORA = timedelta(hours=1)

#: Dia simulado (uma quarta-feira) e horizonte: do começo de D até D+1 03:30,
#: para ver a virada (end-day de D no tick de D+1 00:00) e o fechamento das
#: janelas da noite.
DIA = date(2026, 10, 21)
#: O relógio fica parado em INICIO_SIM até a âncora (alguns segundos depois de
#: recriar a API); o primeiro tick é o de D 00:00.
INICIO_SIM = datetime(DIA.year, DIA.month, DIA.day, 0, 0, tzinfo=SP) - timedelta(minutes=1)
ESPERA_ANCORA_S = 15
FIM_SIM = datetime(DIA.year, DIA.month, DIA.day, 3, 30, tzinfo=SP) + timedelta(days=1)

OUTCOME = {"respondida": "responded", "nao_respondida": "not_responded",
           "falha_entrega": "delivery_failed", "censurada": "censored"}

#: Usuários sintéticos: horas de trabalho, treinos do dia (hora local simulada),
#: feedback de rotina (dificuldade 1-5) depois de um treino e histórico semeado
#: (dias antes de D com treino concluído, às 19h).
PERSONAS = [
    {"nome": "ana", "horas": 8, "treinos": ["07:30"], "feedback": {"07:30": 2}, "historico": [1, 2]},
    {"nome": "bruno", "horas": 6, "treinos": ["12:40", "19:15"], "feedback": {}, "historico": [1]},
    {"nome": "carla", "horas": 8, "treinos": ["18:20"], "feedback": {"18:20": 5}, "historico": []},
    {"nome": "davi", "horas": 10, "treinos": [], "feedback": {}, "historico": [3]},
    {"nome": "elisa", "horas": 4, "treinos": ["21:10"], "feedback": {"21:10": 3}, "historico": [1, 2, 3]},
    {"nome": "fabio", "horas": 0, "treinos": ["10:05", "16:50"], "feedback": {}, "historico": []},
]


# ---- infraestrutura ------------------------------------------------------------

def carregar_env() -> dict:
    env = dict(os.environ)
    for linha in (HERE / "demo.env").read_text().splitlines():
        linha = linha.strip()
        if linha and not linha.startswith("#") and "=" in linha:
            k, v = linha.split("=", 1)
            env.setdefault(k, v)
    env.setdefault("BACKEND_DIR", str((HERE / "../../_wt/backend-pr2").resolve()))
    env.setdefault("DEMO_PG_PORT", "55480")
    env.setdefault("DEMO_DECISION_PORT", "58480")
    env.setdefault("DEMO_API_PORT", "53480")
    env.setdefault("DEMO_TICK_MS", "2000")
    return env


ENV: dict = {}


def compose(*args, extra: dict | None = None, check=True, capture=False, quiet=False):
    cmd = ["docker", "compose", "-p", PROJECT, "-f", str(COMPOSE_FILE), *args]
    env = {**ENV, **(extra or {})}
    if not quiet:
        print("  $ docker compose", " ".join(args), flush=True)
    r = subprocess.run(cmd, env=env, text=True,
                       stdout=subprocess.PIPE if capture else (subprocess.DEVNULL if quiet else None),
                       stderr=subprocess.PIPE if capture else None)
    if check and r.returncode != 0:
        raise SystemExit(f"falhou: {' '.join(cmd)}\n{r.stderr or ''}")
    return r


def psql(banco: str, sql: str):
    """Executa SQL no Postgres da demo e devolve a primeira coluna da 1a linha."""
    r = compose("exec", "-T", "db", "psql", "-U", "spyno", "-d", banco, "-At", "-v",
                "ON_ERROR_STOP=1", "-c", sql, capture=True, quiet=True)
    return r.stdout.strip()


def psql_json(banco: str, consulta: str) -> list:
    out = psql(banco, f"select coalesce(json_agg(t), '[]'::json) from ({consulta}) t")
    return json.loads(out)


def http(metodo: str, url: str, corpo=None, headers=None, timeout=10):
    dados = None if corpo is None else json.dumps(corpo).encode()
    h = {"Accept": "application/json", **(headers or {})}
    if corpo is not None:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=dados, headers=h, method=metodo)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            texto = r.read().decode()
            return r.status, (json.loads(texto) if texto else None)
    except urllib.error.HTTPError as e:
        texto = e.read().decode()
        try:
            return e.code, json.loads(texto)
        except ValueError:
            return e.code, texto


def esperar(cond, desc: str, timeout_s=60.0, passo=0.5):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if cond():
                return
        except Exception:
            pass
        time.sleep(passo)
    raise SystemExit(f"tempo esgotado esperando {desc}")


def dt(valor) -> datetime | None:
    """Datas do Postgres: timestamp sem fuso (Prisma, em UTC) ou timestamptz."""
    if valor is None:
        return None
    d = datetime.fromisoformat(valor)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# ---- relógio simulado (o mesmo de src/services/relogio.js do backend) ----------

class Relogio:
    def __init__(self, ancora_ms: int, inicio: datetime, tick_ms: int):
        self.ancora_ms, self.inicio, self.escala = ancora_ms, inicio, 3_600_000 / tick_ms

    def agora(self) -> datetime:
        decorrido = max(0.0, time.time() * 1000 - self.ancora_ms)
        return self.inicio + timedelta(milliseconds=decorrido * self.escala)


# ---- etapas --------------------------------------------------------------------

def semear(api: str) -> list[dict]:
    """Cria os usuários pela API (signup, signin, treino, rotinas)."""
    usuarios = []
    for p in PERSONAS:
        email = f"{p['nome']}@demo.spyno.invalid"
        s, _ = http("POST", f"{api}/user/signup", {
            "nome_completo": p["nome"].title() + " Demo", "email": email, "senha": "senha-demo-123",
            "tipo": "Geral", "horas_de_trabalho": p["horas"], "data_nascimento": "1995-05-05"})
        assert s in (200, 201), f"signup {p['nome']}: {s}"
        s, login = http("POST", f"{api}/user/signin", {"email": email, "senha": "senha-demo-123"})
        assert s == 200, f"signin {p['nome']}: {s}"
        jwt, uid = login["token"], login["usuario"]["id_usuario"]
        auth = {"Authorization": f"Bearer {jwt}"}
        n = max(1, len(p["treinos"]))
        s, treino = http("POST", f"{api}/training", {"email": email, "nome_treino": "Rotina demo",
                                                      "numero_dias": n}, auth)
        assert s == 201, f"treino {p['nome']}: {s} {treino}"
        id_treino = treino["treino"]["id_treino"]
        rotinas = []
        for i in range(n):
            s, r = http("POST", f"{api}/routine", {"id_treino": id_treino, "dia_num": i + 1,
                                                    "dia_semana": "quarta"}, auth)
            assert s == 201, f"rotina {p['nome']}: {s} {r}"
            rotinas.append(r["rotina"]["id_rotina"])
        usuarios.append({**p, "id": uid, "jwt": jwt, "id_treino": id_treino, "rotinas": rotinas})
    # Histórico de dias anteriores (rotinas já concluídas), direto no banco: a
    # API só marca conclusão no "agora". Dá streak, recência e valência reais.
    for u in usuarios:
        for k, dias_antes in enumerate(u["historico"]):
            quando = datetime(DIA.year, DIA.month, DIA.day, 19, tzinfo=SP) - timedelta(days=dias_antes)
            psql("spyno", 'insert into "Rotina" (id_treino, dia_num, rotina_finalizada, finalizada_em) '
                 f"values ({u['id_treino']}, {100 + k}, true, "
                 f"'{quando.astimezone(timezone.utc).replace(tzinfo=None).isoformat()}')")
    return usuarios


def agenda_de_treinos(usuarios):
    eventos = []
    for u in usuarios:
        for i, hhmm in enumerate(u["treinos"]):
            h, m = map(int, hhmm.split(":"))
            quando = datetime(DIA.year, DIA.month, DIA.day, h, m, tzinfo=SP)
            eventos.append({"quando": quando, "usuario": u, "rotina": u["rotinas"][i],
                            "feedback": u["feedback"].get(hhmm)})
    return sorted(eventos, key=lambda e: e["quando"])


def main() -> int:
    global ENV
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sem-shadow", action="store_true",
                    help="SPYNO_SHADOW=0: envios reais, entrega FCM simulada no banco")
    ap.add_argument("--manter", action="store_true", help="não derruba os contêineres no fim")
    ap.add_argument("--sem-build", action="store_true", help="reaproveita as imagens já construídas")
    ap.add_argument("--warmstart", action="store_true",
                    help="serviço sobe com SPYNO_WARMSTART_SNAPSHOT (warmstart/ts_hybrid-v2-c0.1.json)")
    ap.add_argument("--reiniciar", action="store_true",
                    help="depois do end-day, reinicia o contêiner decision e confere a posterior "
                         "carregada (em shadow, também o recria com SPYNO_SHADOW=0)")
    args = ap.parse_args()

    ENV = carregar_env()
    shadow = not args.sem_shadow
    ENV["DEMO_SHADOW"] = "1" if shadow else "0"
    ENV["DEMO_WARMSTART_SNAPSHOT"] = (
        "/app/warmstart/ts_hybrid-v2-c0.1.json" if args.warmstart else "")
    tick_ms = int(ENV["DEMO_TICK_MS"])
    backend = Path(ENV["BACKEND_DIR"])
    if not (backend / "Dockerfile").exists():
        raise SystemExit(f"BACKEND_DIR sem Dockerfile: {backend}")
    api = f"http://127.0.0.1:{ENV['DEMO_API_PORT']}"
    svc = f"http://127.0.0.1:{ENV['DEMO_DECISION_PORT']}"
    bearer = {"Authorization": f"Bearer {ENV['DEMO_DECISION_TOKEN']}"}
    servico_hdr = {"X-Service-Token": ENV["DEMO_SERVICE_TOKEN"]}  # noqa: F841  (lê o token: falha cedo se faltar no demo.env)
    global OUT
    OUT = OUT / (("shadow" if shadow else "sem-shadow")
                 + ("-warmstart" if args.warmstart else "")
                 + ("-reinicio" if args.reiniciar else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def confere(nome: str, ok: bool, detalhe=""):
        checks.append({"item": nome, "ok": bool(ok), "detalhe": detalhe})
        print(f"  {'ok   ' if ok else 'FALHA'} {nome}" + (f"  ({detalhe})" if detalhe else ""), flush=True)

    print(f"== Demo F4 ({'shadow' if shadow else 'sem shadow'}), backend em {backend}")
    compose("down", "-v", "--remove-orphans", quiet=True, check=False)
    try:
        if not args.sem_build:
            compose("build", "decision", "api")
        compose("up", "-d", "--wait", "db", "decision")

        print("== Migrations do backend")
        compose("run", "--rm", "migrate")
        n_migr = len([d for d in (backend / "prisma/migrations").iterdir() if d.is_dir()])
        aplicadas = int(psql("spyno", "select count(*) from _prisma_migrations where finished_at is not null"))
        confere("migrations aplicadas num banco vazio", aplicadas == n_migr, f"{aplicadas}/{n_migr}")

        print("== Usuários sintéticos (agendador desligado)")
        compose("up", "-d", "--wait", "api", extra={"DEMO_DECISION_ENABLED": "false"})
        esperar(lambda: http("GET", f"{api}/ping")[0] == 200, "a API")
        usuarios = semear(api)
        print(f"  {len(usuarios)} usuários: " + ", ".join(f"{u['nome']}={u['id']}" for u in usuarios))

        print("== Contrato do serviço")
        s, contrato = http("GET", f"{svc}/v1/contract")
        confere("GET /v1/contract sem Authorization -> 200", s == 200)
        init_esperado = "warmstart" if args.warmstart else "prior"
        confere(f"posterior inicial = {init_esperado}",
                contrato.get("posterior_init") == init_esperado
                and (contrato.get("warmstart_sha256") is not None) == args.warmstart,
                f"posterior_init={contrato.get('posterior_init')}, "
                f"contract_version={contrato.get('contract_version')}")
        corpo = {"user_key": "x", "decision_point": "2026-10-21T10:00:00-03:00", "eligible": True,
                 "context": {}}
        s1, _ = http("POST", f"{svc}/v1/decide", corpo)
        s2, _ = http("POST", f"{svc}/v1/decide", corpo, {"Authorization": "Bearer errado"})
        s3, _ = http("GET", f"{svc}/v1/policy-state", headers={"Authorization": "Bearer errado"})
        confere("/v1/* sem Bearer ou com Bearer errado -> 401", (s1, s2, s3) == (401, 401, 401), f"{s1},{s2},{s3}")

        print("== Subida com janela divergente (SPYNO_OUTCOME_WINDOW_MIN=120)")
        nome_div = f"{PROJECT}-api-janela-divergente"
        subprocess.run(["docker", "rm", "-f", nome_div], capture_output=True)
        compose("run", "-d", "--no-deps", "--name", nome_div, "api",
                extra={"DEMO_DECISION_ENABLED": "true", "DEMO_BACKEND_WINDOW_MIN": "120"}, quiet=True)
        log_div = ""

        def recusou():
            nonlocal log_div
            # O aviso de contrato divergente sai em stderr (console.error).
            log_div = subprocess.run(["docker", "logs", nome_div], text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT).stdout
            return "contrato divergente" in log_div
        try:
            esperar(recusou, "a recusa do contrato", timeout_s=30)
        except SystemExit:
            pass
        subprocess.run(["docker", "rm", "-f", nome_div], capture_output=True)
        linha = next((l for l in log_div.splitlines() if "contrato" in l), log_div[-200:])
        confere("janela divergente -> agendador NAO iniciado", "agendador NAO iniciado: contrato divergente" in log_div,
                linha.strip())

        _, estado0 = http("GET", f"{svc}/v1/policy-state", headers=bearer)

        print(f"== Dia simulado: {INICIO_SIM:%Y-%m-%d %H:%M} -> {FIM_SIM:%Y-%m-%d %H:%M} "
              f"(1 h simulada = {tick_ms} ms)")
        ancora = int(time.time() * 1000) + ESPERA_ANCORA_S * 1000
        relogio = Relogio(ancora, INICIO_SIM, tick_ms)
        compose("up", "-d", "--wait", "--force-recreate", "--no-deps", "api", extra={
            "DEMO_DECISION_ENABLED": "true", "DEMO_CLOCK": "on",
            "DEMO_CLOCK_START": INICIO_SIM.isoformat(), "DEMO_CLOCK_ANCHOR_MS": str(ancora)})
        esperar(lambda: http("GET", f"{api}/ping")[0] == 200, "a API com o relógio simulado")
        if time.time() * 1000 > ancora:
            print("  ATENCAO: a API subiu depois da âncora; as primeiras horas podem faltar")

        eventos = agenda_de_treinos(usuarios)
        historico_posterior = []
        ultimo_poll = ultimo_fcm = 0.0
        treinos_feitos = []
        while relogio.agora() < FIM_SIM:
            agora_sim = relogio.agora()
            while eventos and eventos[0]["quando"] <= agora_sim:
                e = eventos.pop(0)
                auth = {"Authorization": f"Bearer {e['usuario']['jwt']}"}
                s, r = http("PUT", f"{api}/routine/{e['rotina']}", {"rotina_finalizada": True}, auth)
                fin = (r or {}).get("rotina", {}).get("finalizada_em") if isinstance(r, dict) else None
                treinos_feitos.append({"usuario": e["usuario"]["nome"], "id_usuario": e["usuario"]["id"],
                                       "previsto": e["quando"].isoformat(), "finalizada_em": fin,
                                       "status": s})
                print(f"  [{agora_sim:%d %H:%M}] {e['usuario']['nome']} concluiu a rotina {e['rotina']} ({s})")
                if e["feedback"] is not None:
                    http("POST", f"{api}/feedbackroutine", {
                        "id_rotina": e["rotina"], "id_treino": e["usuario"]["id_treino"],
                        "nivel_dificuldade": e["feedback"], "nivel_familiaridade": 3}, auth)
            agora_real = time.time()
            if not shadow and agora_real - ultimo_fcm > 0.4:
                # TODO(FCM) do backend: a entrega é simulada aqui, 2 min depois
                # da decisão, para todo push decidido.
                psql("spyno", 'update "Notificacao" set entregue_em = decidida_em + interval \'2 minutes\' '
                     "where acao_executada > 0 and entregue_em is null and desfecho is null")
                ultimo_fcm = agora_real
            if agora_real - ultimo_poll > 0.25:
                s, st = http("GET", f"{svc}/v1/policy-state", headers=bearer)
                if s == 200:
                    historico_posterior.append({"sim": relogio.agora().isoformat(), **st})
                ultimo_poll = agora_real
            time.sleep(0.05)

        # Congela: sem agendador, nada muda enquanto exporta.
        compose("stop", "-t", "2", "api", quiet=True)
        print(f"  agendador parado em {relogio.agora():%Y-%m-%d %H:%M} (simulado)")
        log_api = compose("logs", "--no-color", "api", capture=True, quiet=True).stdout
        log_svc = compose("logs", "--no-color", "decision", capture=True, quiet=True).stdout
        (OUT / "api.log").write_text(log_api)
        (OUT / "decision.log").write_text(log_svc)

        print("== Exportação")
        notifs = psql_json("spyno", 'select n.*, u.email from "Notificacao" n join "Usuario" u '
                           "using (id_usuario) order by n.decidida_em, n.id_usuario")
        decisoes = psql_json("spyno_decision", "select * from decisions order by decision_point, user_key")
        dias = psql_json("spyno_decision", "select * from policy_days order by day")
        posteriores = psql_json("spyno_decision", "select policy_version, day, digest "
                                "from policy_posteriors order by saved_at")
        _, health = http("GET", f"{svc}/health")

        # end-day repetido direto no serviço: idempotente, posterior intacta.
        _, antes_rep = http("GET", f"{svc}/v1/policy-state", headers=bearer)
        s, rep = http("POST", f"{svc}/v1/end-day", {"date": DIA.isoformat()}, bearer)
        _, depois_rep = http("GET", f"{svc}/v1/policy-state", headers=bearer)

        reinicio = None
        if args.reiniciar:
            print("== Reinício do serviço depois do end-day")
            compose("restart", "-t", "5", "decision")
            esperar(lambda: http("GET", f"{svc}/health")[0] == 200, "o serviço reiniciado")
            _, apos = http("GET", f"{svc}/v1/policy-state", headers=bearer)
            reinicio = {"antes": depois_rep, "depois": apos}
            # Em shadow nada é persistido: o reinício volta ao estado inicial.
            fonte = init_esperado if shadow else "persisted"
            confere(f"reinício: posterior_source = {fonte}", apos.get("posterior_source") == fonte,
                    f"source={apos.get('posterior_source')}, dia={apos.get('persisted_day')}")
            confere("reinício: mesmo digest da posterior",
                    apos.get("posterior_digest") == depois_rep.get("posterior_digest"),
                    f"{depois_rep.get('posterior_digest')} -> {apos.get('posterior_digest')}")
            confere("reinício: mesma policy_version",
                    apos.get("policy_version") == depois_rep.get("policy_version"),
                    apos.get("policy_version", ""))
            confere(f"reinício: posterior_init = {init_esperado}",
                    apos.get("posterior_init") == init_esperado)
            if shadow:
                print("== Saída do shadow: decision recriado com SPYNO_SHADOW=0")
                compose("up", "-d", "--wait", "--force-recreate", "--no-deps", "decision",
                        extra={"DEMO_SHADOW": "0"})
                esperar(lambda: http("GET", f"{svc}/health")[0] == 200, "o serviço fora do shadow")
                _, saida = http("GET", f"{svc}/v1/policy-state", headers=bearer)
                _, h_saida = http("GET", f"{svc}/health")
                reinicio["fora_do_shadow"] = {"policy_state": saida, "health": h_saida}
                confere("saída do shadow: aprendizado ativo (shadow=false, learning_frozen=false)",
                        h_saida.get("shadow") is False and h_saida.get("learning_frozen") is False
                        and saida.get("learning_frozen") is False)
                confere("saída do shadow: posterior = a inicial (digest da subida, mesma origem)",
                        saida.get("posterior_digest") == estado0.get("posterior_digest")
                        and saida.get("posterior_source") == init_esperado,
                        f"{estado0.get('posterior_digest')} -> {saida.get('posterior_digest')}, "
                        f"source={saida.get('posterior_source')}")
                confere("saída do shadow: mesma policy_version",
                        saida.get("policy_version") == estado0.get("policy_version"))
            (OUT / "reinicio.json").write_text(json.dumps(reinicio, indent=1, default=str))

        exportar(notifs, decisoes, dias, treinos_feitos, historico_posterior, usuarios)

        print("== Verificação do contrato")
        verificar(confere, contrato, notifs, decisoes, dias, usuarios, historico_posterior,
                  estado0, (s, rep, antes_rep, depois_rep), log_api, log_svc, shadow,
                  posteriores, health)

        resumo = resumir(notifs, decisoes, dias, treinos_feitos, contrato, shadow, tick_ms)
        resumo["verificacao"] = checks
        (OUT / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=str))
        imprimir_resumo(resumo)
    finally:
        if args.manter:
            print(f"== Contêineres mantidos (docker compose -p {PROJECT} -f {COMPOSE_FILE} down -v)")
        else:
            print("== Derrubando só os contêineres da demo")
            compose("down", "-v", "--remove-orphans", quiet=True, check=False)

    falhas = [c for c in checks if not c["ok"]]
    print(f"\n{'TUDO OK' if not falhas else f'{len(falhas)} FALHA(S)'}; saída em {OUT}")
    return 1 if falhas else 0


# ---- exportação ----------------------------------------------------------------

COLUNAS = ["id_notificacao", "id_usuario", "email", "decision_id", "decidida_em", "decidida_em_local",
           "hora_local", "shadow", "politica", "versao_politica", "acao_recomendada", "acao_executada",
           "acao_proposta", "motivo_bloqueio", "propensity", "propensity_recomendada",
           "probabilidades_bracos", "atualiza_politica", "base_contadores", "eligible",
           "notifs_today", "notifs_today_proposed", "fatigue", "fatigue_proposed",
           "minutes_since_last", "minutes_since_last_proposed", "entregue_em", "janela_fecha_em",
           "desfecho", "censurada", "observado_em", "feedback_enviado_em", "servico_outcome",
           "servico_trained", "servico_train_skip_reason", "contexto"]


def linhas_csv(notifs, decisoes):
    por_id = {d["decision_id"]: d for d in decisoes}
    for n in notifs:
        ctx = n.get("contexto") or {}
        d = por_id.get(n["decision_id"], {})
        local = dt(n["decidida_em"]).astimezone(SP)
        yield {
            **{k: n.get(k) for k in COLUNAS if k in n},
            "decidida_em_local": local.isoformat(), "hora_local": local.hour,
            "eligible": ctx.get("eligible"),
            **{k: ctx.get(k) for k in ("notifs_today", "notifs_today_proposed", "fatigue",
                                       "fatigue_proposed", "minutes_since_last",
                                       "minutes_since_last_proposed")},
            "censurada": n.get("desfecho") == "censurada",
            "probabilidades_bracos": json.dumps(n.get("probabilidades_bracos")),
            "servico_outcome": d.get("outcome"), "servico_trained": d.get("trained"),
            "servico_train_skip_reason": d.get("train_skip_reason"),
            "contexto": json.dumps(ctx, sort_keys=True),
        }


def exportar(notifs, decisoes, dias, treinos, historico_posterior, usuarios):
    with open(OUT / "notificacoes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        for linha in linhas_csv(notifs, decisoes):
            w.writerow(linha)
    (OUT / "notificacoes.json").write_text(json.dumps(notifs, indent=1, ensure_ascii=False))
    (OUT / "servico_decisoes.json").write_text(json.dumps(decisoes, indent=1, ensure_ascii=False))
    (OUT / "servico_dias.json").write_text(json.dumps(dias, indent=1))
    (OUT / "treinos.json").write_text(json.dumps(treinos, indent=1, ensure_ascii=False))
    (OUT / "posterior.json").write_text(json.dumps(historico_posterior, indent=1))
    (OUT / "usuarios.json").write_text(json.dumps(
        [{k: u[k] for k in ("nome", "id", "horas", "treinos", "historico", "id_treino", "rotinas")}
         for u in usuarios], indent=1, ensure_ascii=False))
    print(f"  {len(notifs)} notificações -> {OUT / 'notificacoes.csv'} e .json")


# ---- verificação ---------------------------------------------------------------

def verificar(confere, contrato, notifs, decisoes, dias, usuarios, posterior, estado0,
              endday_rep, log_api, log_svc, shadow, posteriores, health):
    segredo = ENV["DEMO_USER_KEY_SECRET"].encode()
    chave = {u["id"]: hmac.new(segredo, str(u["id"]).encode(), hashlib.sha256).hexdigest()
             for u in usuarios}
    por_id = {d["decision_id"]: d for d in decisoes}
    versao = contrato["policy_version"]

    # Subida com contrato ok.
    linha_ok = next((l for l in log_api.splitlines() if "contrato ok" in l), "")
    confere("subida: contrato lido e agendador ativo", bool(linha_ok), linha_ok.split("| ")[-1].strip())

    # Bearer: o serviço só recusou as 3 chamadas de teste deste roteiro.
    recusas = sum(1 for l in log_svc.splitlines() if " 401" in l)
    confere("Bearer: nenhuma chamada do backend recusada (401 só nos testes negativos)",
            recusas == 3 and len(decisoes) > 0, f"{recusas} respostas 401 no log do serviço")

    # HMAC do user_key.
    chaves = {d["user_key"] for d in decisoes}
    crus = {str(u["id"]) for u in usuarios} & chaves
    confere("user_key = HMAC-SHA256(id, segredo); nenhum id cru no serviço",
            chaves == set(chave.values()) and not crus, f"{len(chaves)} chaves")
    uid_errado = [n for n in notifs if por_id.get(n["decision_id"], {}).get("user_key") != chave[n["id_usuario"]]]
    confere("cada Notificacao aponta para a decisão do user_key do seu usuário", not uid_errado,
            f"{len(uid_errado)} divergentes")

    # decision_id devolvido e usado.
    ids_back, ids_svc = {n["decision_id"] for n in notifs}, set(por_id)
    def recalcula(d):
        h = dt(d["decision_point"]).astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        return hashlib.sha256(f"{d['user_key']}|{h.isoformat()}|{d['policy_version']}".encode()).hexdigest()[:32]
    confere("decision_id: mesmo conjunto no backend e no serviço", ids_back == ids_svc,
            f"backend {len(ids_back)}, serviço {len(ids_svc)}")
    confere("decision_id = sha256(user_key|hora UTC|policy_version)",
            all(recalcula(d) == d["decision_id"] for d in decisoes))
    confere("decidida_em do backend = decision_point do serviço (hora cheia)",
            all(dt(n["decidida_em"]) == dt(por_id[n["decision_id"]]["decision_point"]) and
                dt(n["decidida_em"]).minute == 0 for n in notifs if n["decision_id"] in por_id))
    horas = sorted({dt(n["decidida_em"]) for n in notifs})
    seguidas = all(b - a == HORA for a, b in zip(horas, horas[1:]))
    completas = all(sum(1 for n in notifs if dt(n["decidida_em"]) == h) == len(usuarios) for h in horas)
    confere("um ponto de decisão por usuário a cada hora simulada, sem hora pulada",
            seguidas and completas and horas[0].astimezone(SP).hour == 0,
            f"{len(horas)} horas: {horas[0].astimezone(SP):%d %H:%M} -> {horas[-1].astimezone(SP):%d %H:%M}")
    confere("policy_version única e igual à do contrato",
            {n["versao_politica"] for n in notifs} == {versao}, versao)

    # Feedback de todas as decisões fechadas, braço 0 incluído.
    fechadas = [n for n in notifs if n["desfecho"]]
    abertas = [n for n in notifs if not n["desfecho"]]
    sem_fb = [n for n in fechadas if not n["feedback_enviado_em"]]
    diverg = [n for n in fechadas
              if por_id[n["decision_id"]]["outcome"] != OUTCOME[n["desfecho"]]]
    confere("feedback enviado para toda decisão fechada", not sem_fb and fechadas,
            f"{len(fechadas)} fechadas, {len(sem_fb)} sem feedback")
    confere("desfecho no serviço = desfecho do backend (mapa do contrato)", not diverg,
            f"{len(diverg)} divergentes")
    braco0 = [n for n in fechadas if n["acao_executada"] == 0]
    confere("feedback do braço 0 (executada = 0) enviado e aplicado",
            braco0 and all(por_id[n["decision_id"]]["outcome"] for n in braco0), f"{len(braco0)} decisões")
    if not shadow:
        envio = [n for n in fechadas if n["acao_executada"] > 0]
        confere("feedback dos braços de envio enviado e aplicado",
                envio and all(por_id[n["decision_id"]]["outcome"] for n in envio), f"{len(envio)} decisões")
    fim = max(dt(n["decidida_em"]) for n in notifs)
    abertas_ok = all(dt(n["janela_fecha_em"]) >= fim for n in abertas)
    confere("só ficam abertas as janelas que ainda não venceram", abertas_ok, f"{len(abertas)} abertas")
    confere("serviço sem pendentes além das janelas abertas",
            sum(1 for d in decisoes if d["outcome"] is None) == len(abertas))
    obs_ok = all(dt(n["observado_em"]) == dt(n["janela_fecha_em"])
                 for n in fechadas if n["desfecho"] == "nao_respondida")
    confere("nao_respondida com observado_em = fechamento da janela", obs_ok)
    respondidas = [n for n in fechadas if n["desfecho"] == "respondida"]
    trained_r = [n for n in respondidas if por_id[n["decision_id"]]["trained"]]
    confere("respondida dentro da janela não é censurada pelo serviço",
            all(por_id[n["decision_id"]]["outcome"] == "responded" for n in respondidas),
            f"{len(respondidas)} respondidas, {len(trained_r)} treinaram")

    # end-day.
    dia_d = DIA.isoformat()
    dias_fechados = {d["day"]: d for d in dias}
    confere(f"end-day de {dia_d} chamado pelo backend na virada", dia_d in dias_fechados,
            json.dumps(dias_fechados.get(dia_d)))
    chamadas_endday = [l for l in log_api.splitlines() if f"dia {dia_d} encerrado" in l]
    confere("backend chamou end-day uma vez para o dia", len(chamadas_endday) == 1,
            f"{len(chamadas_endday)} chamadas")
    s, rep, antes, depois = endday_rep
    confere("end-day repetido: already_closed, sem novo ajuste",
            s == 200 and rep.get("already_closed") and rep.get("updates_applied") == 0, json.dumps(rep))
    confere("end-day repetido não muda a posterior",
            antes["posterior_digest"] == depois["posterior_digest"])

    antes_d = [p for p in posterior if dia_d not in {x["day"] for x in p["days_closed"]}]
    depois_d = [p for p in posterior if dia_d in {x["day"] for x in p["days_closed"]}]
    n_upd = dias_fechados.get(dia_d, {}).get("updates_applied", 0)
    if shadow:
        # Contrato 2.3: em shadow a posterior está congelada.
        digests = {p["posterior_digest"] for p in posterior} | {depois["posterior_digest"]}
        confere("shadow: learning_frozen no contrato, no /health e no /v1/policy-state",
                contrato.get("learning_frozen") is True and health.get("learning_frozen") is True
                and all(p.get("learning_frozen") is True for p in posterior),
                f"contract_version={contrato.get('contract_version')}")
        confere("shadow: digest da posterior constante no dia, no end-day e no repetido",
                bool(antes_d and depois_d) and digests == {estado0["posterior_digest"]},
                f"{len(digests)} digest(s): {sorted(digests)}")
        confere("shadow: end-day fecha o dia com 0 updates e nenhum pendente",
                dia_d in dias_fechados and n_upd == 0
                and all(p["pending_updates"] == 0 for p in posterior), f"{n_upd} updates")
        confere("shadow: nenhuma posterior gravada em policy_posteriors", posteriores == [],
                json.dumps(posteriores))
        congeladas = [d for d in decisoes if d["outcome"] in ("responded", "not_responded")
                      and d["update_eligible"]]
        confere("shadow: nenhum desfecho treinou; os treináveis com motivo decisao_em_shadow_congelado",
                not any(d["trained"] for d in decisoes) and congeladas
                and all(d["train_skip_reason"] == "decisao_em_shadow_congelado" for d in congeladas),
                f"{len(congeladas)} treináveis congelados")
        confere("shadow: learning_frozen em todas as decisões do log",
                all(d["learning_frozen"] is True for d in decisoes), f"{len(decisoes)} decisões")
        confere("shadow: update_eligible preservado no log (seleção para a OPE)",
                any(n["atualiza_politica"] for n in notifs)
                and all(bool(n["atualiza_politica"]) == bool(por_id[n["decision_id"]]["update_eligible"])
                        for n in notifs),
                f"{sum(1 for n in notifs if n['atualiza_politica'])} pontos treináveis")
    elif antes_d and depois_d:
        a, b = antes_d[-1], depois_d[0]
        confere("posterior alterada pelo end-day (hash antes != depois)",
                n_upd > 0 and a["posterior_digest"] != b["posterior_digest"],
                f"{a['posterior_digest']} -> {b['posterior_digest']}, {n_upd} updates, "
                f"{a['pending_updates']} pendentes antes")
        mudou_no_dia = {p["posterior_digest"] for p in antes_d if p["sim"] >= INICIO_SIM.isoformat()}
        confere("posterior só muda no end-day (feedback só acumula)", len(mudou_no_dia) <= 1,
                f"{len(mudou_no_dia)} hashes durante o dia")
        confere("sem shadow: learning_frozen = false (contrato, /health e decisões)",
                contrato.get("learning_frozen") is False and health.get("learning_frozen") is False
                and all(d["learning_frozen"] is False for d in decisoes))
        confere("sem shadow: posterior do dia gravada em policy_posteriors",
                [p["day"] for p in posteriores] == [dia_d], json.dumps(posteriores))
    else:
        confere("posterior alterada pelo end-day", False, "sem observações antes/depois")
    trained_total = sum(1 for d in decisoes if d["trained"])
    confere("updates aplicados no end-day = desfechos que treinaram até a virada",
            n_upd == sum(1 for d in decisoes if d["trained"] and
                         dt(d["outcome_at"]) <= datetime(DIA.year, DIA.month, DIA.day, tzinfo=SP) + timedelta(days=1)),
            f"{n_upd} no dia, {trained_total} no total")

    # Shadow e contadores.
    if shadow:
        confere("shadow: nada executado", all(n["acao_executada"] == 0 and n["shadow"] for n in notifs))
        confere("shadow: counter_basis = proposed em todas",
                {n["base_contadores"] for n in notifs} == {"proposed"})
    else:
        confere("sem shadow: counter_basis = executed", {n["base_contadores"] for n in notifs} == {"executed"})
    confere("contexto enviado com *_proposed",
            all({"notifs_today_proposed", "fatigue_proposed", "minutes_since_last_proposed"}
                <= set(n["contexto"]) for n in notifs))
    def mesmo_contexto(a: dict, b: dict) -> bool:
        # jsonb do Postgres pode encurtar o último dígito de um float.
        a = {k: v for k, v in a.items() if k != "eligible"}
        return a.keys() == b.keys() and all(
            a[k] == b[k] or (isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))
                             and abs(a[k] - b[k]) < 1e-12) for k in a)
    confere("contexto gravado = contexto recebido pelo serviço",
            all(mesmo_contexto(n["contexto"], json.loads(por_id[n["decision_id"]]["context"]))
                for n in notifs))

    # Semântica única da ação proposta.
    prop_svc = [n for n in notifs if n["acao_proposta"] != por_id[n["decision_id"]]["proposed_action"]]
    confere("acao_proposta do backend = proposed_action do serviço", not prop_svc, f"{len(prop_svc)} divergentes")
    bloqueadas = [n for n in notifs if n["acao_recomendada"] > 0
                  and n["motivo_bloqueio"] in ("quiet_hours", "daily_cap", "cooldown")]
    confere("recomendação bloqueada por guardrail -> proposta 0",
            all(n["acao_proposta"] == 0 for n in bloqueadas), f"{len(bloqueadas)} bloqueadas")
    livres = [n for n in notifs if n["motivo_bloqueio"] in (None, "shadow")]
    confere("sem guardrail -> proposta = recomendada",
            all(n["acao_proposta"] == n["acao_recomendada"] for n in livres), f"{len(livres)} livres")

    # Contadores recalculados a partir do próprio log.
    erros = []
    por_usuario: dict[int, list] = {}
    for n in sorted(notifs, key=lambda n: n["decidida_em"]):
        por_usuario.setdefault(n["id_usuario"], []).append(n)
    cap = contrato["daily_cap"]
    for lista in por_usuario.values():
        for i, n in enumerate(lista):
            t = dt(n["decidida_em"])
            dia_local = t.astimezone(SP).date()
            if shadow:
                teria = [m for m in lista[:i] if m["acao_proposta"] > 0]
                instante = lambda m: dt(m["decidida_em"])  # noqa: E731
            else:
                teria = [m for m in lista[:i] if m["acao_executada"] > 0 and m["entregue_em"]
                         and dt(m["entregue_em"]) < t]
                instante = lambda m: dt(m["entregue_em"])  # noqa: E731
            hoje = sum(1 for m in teria if dt(m["decidida_em"]).astimezone(SP).date() == dia_local)
            ultimo = max((instante(m) for m in teria), default=None)
            minutos = None if ultimo is None else int((t - ultimo).total_seconds() // 60)
            c = n["contexto"]
            esperado_n = hoje
            campo_n = "notifs_today_proposed" if shadow else "notifs_today"
            campo_m = "minutes_since_last_proposed" if shadow else "minutes_since_last"
            if c[campo_n] != esperado_n or c[campo_m] != minutos or c["eligible"] != (c[campo_n] < cap):
                erros.append((n["decision_id"][:8], c[campo_n], esperado_n, c[campo_m], minutos))
            if shadow and (c["notifs_today"] != 0 or c["minutes_since_last"] is not None):
                erros.append((n["decision_id"][:8], "reais != 0 em shadow"))
    confere("contadores e eligible recalculados do log batem com o contexto enviado",
            not erros, f"{len(erros)} divergentes {erros[:3]}")

    # Propensidades.
    props = [n["propensity"] for n in notifs]
    recs = [n["propensity_recomendada"] for n in notifs if n["propensity_recomendada"] is not None]
    # Pontos inelegíveis (cota do dia): a política nem é consultada para enviar,
    # pi = [1, 0, 0, 0] e a executada (0) tem propensão 1. Nos elegíveis, o
    # clipping garante todo braço > 0.
    bracos = [p for n in notifs if n["probabilidades_bracos"] and n["contexto"]["eligible"]
              for p in n["probabilidades_bracos"]]
    inelegiveis = [n for n in notifs if not n["contexto"]["eligible"]]
    confere("nenhuma propensity zero (executada, recomendada e por braço nos elegíveis)",
            min(props) > 0 and min(recs) > 0 and (not bracos or min(bracos) > 0),
            f"min executada {min(props):.3f}, min recomendada {min(recs):.3f}, "
            f"min braço elegível {min(bracos) if bracos else float('nan'):.3f}")
    confere("inelegível: pi = [1,0,0,0], executada 0 com propensity 1, sem treino",
            all(n["probabilidades_bracos"] in (None, [1, 0, 0, 0]) and n["acao_executada"] == 0
                and n["propensity"] == 1 and not n["atualiza_politica"] for n in inelegiveis),
            f"{len(inelegiveis)} pontos inelegíveis")
    fallback = [d for d in decisoes if d["fallback"]]
    confere("nenhuma decisão caiu na contingência", not fallback,
            ", ".join(sorted({d['fallback_reason'] for d in fallback})))
    pi_ok = all(abs(sum(n["probabilidades_bracos"]) - 1) < 1e-6 and
                abs(n["probabilidades_bracos"][n["acao_recomendada"]] - n["propensity_recomendada"]) < 1e-9
                for n in notifs if n["probabilidades_bracos"])
    confere("propensity_recomendada = pi[recomendada], pi soma 1", pi_ok)
    p_envio = [sum(n["probabilidades_bracos"][1:]) for n in notifs
               if n["probabilidades_bracos"] and n["contexto"]["eligible"]]
    confere("P(enviar) clipada em [clip_min, clip_max] nos pontos elegíveis",
            all(contrato["clip_min"] - 1e-9 <= p <= contrato["clip_max"] + 1e-9 for p in p_envio),
            f"{min(p_envio):.3f}..{max(p_envio):.3f}" if p_envio else "sem pontos")
    # Piso por braço (padrão p05: 0,05/3) e peso de importância máximo do contrato
    # (60 no padrão), nos braços de envio e na abstenção dos pontos elegíveis.
    pis = [n["probabilidades_bracos"] for n in notifs
           if n["probabilidades_bracos"] and n["contexto"]["eligible"]]
    piso, w_max = contrato["arm_floor"], contrato["max_importance_weight"]
    # ts_binary (pi[1] = pi[2] = 0) ou política sem clipping (w_max None): reprova sem quebrar.
    if piso is None or w_max is None or any(min(pi) <= 0 for pi in pis):
        confere("pi por braço de envio >= arm_floor e 1/pi <= max_importance_weight nos elegíveis",
                False, f"arm_floor {piso}, max_importance_weight {w_max}, braço com pi = 0")
        return
    confere("pi por braço de envio >= arm_floor e 1/pi <= max_importance_weight nos elegíveis",
            all(min(pi[1:]) >= piso - 1e-12 and 1.0 / min(pi) <= w_max + 1e-9 for pi in pis),
            f"arm_floor {piso:.6g}, max_importance_weight {w_max:g}, "
            f"maior 1/pi observado {max(1.0 / min(pi) for pi in pis):.1f}" if pis
            else "sem pontos")


def resumir(notifs, decisoes, dias, treinos, contrato, shadow, tick_ms):
    from collections import Counter
    props = [n["propensity"] for n in notifs]
    recs = [n["propensity_recomendada"] for n in notifs if n["propensity_recomendada"] is not None]
    eleg = [n for n in notifs if n["probabilidades_bracos"] and n["contexto"]["eligible"]]
    p_env = [sum(n["probabilidades_bracos"][1:]) for n in eleg]
    return {
        "modo": "shadow" if shadow else "sem shadow (entrega FCM simulada)",
        "dia_simulado": DIA.isoformat(), "inicio_sim": INICIO_SIM.isoformat(), "fim_sim": FIM_SIM.isoformat(),
        "tick_ms_por_hora_simulada": tick_ms, "policy_version": contrato["policy_version"],
        "usuarios": len({n["id_usuario"] for n in notifs}),
        "decisoes": len(notifs),
        "acao_recomendada": dict(sorted(Counter(n["acao_recomendada"] for n in notifs).items())),
        "acao_proposta": dict(sorted(Counter(n["acao_proposta"] for n in notifs).items())),
        "acao_executada": dict(sorted(Counter(n["acao_executada"] for n in notifs).items())),
        "motivo_bloqueio": dict(Counter(n["motivo_bloqueio"] or "-" for n in notifs)),
        "desfechos": dict(Counter(n["desfecho"] or "aberta" for n in notifs)),
        "janelas_censuradas": sum(1 for n in notifs if n["desfecho"] == "censurada"),
        "base_contadores": dict(Counter(n["base_contadores"] for n in notifs)),
        "atualiza_politica": sum(1 for n in notifs if n["atualiza_politica"]),
        "treinaram_no_servico": sum(1 for d in decisoes if d["trained"]),
        "motivos_sem_treino": dict(Counter(d["train_skip_reason"] for d in decisoes
                                           if d["outcome"] and not d["trained"])),
        "propensity_executada": {"min": min(props), "max": max(props)},
        "propensity_recomendada": {"min": min(recs), "max": max(recs)},
        "p_enviar_elegiveis": {"min": min(p_env), "max": max(p_env)} if p_env else None,
        "min_prob_braco_elegiveis": min(min(n["probabilidades_bracos"]) for n in eleg) if eleg else None,
        "pontos_inelegiveis": sum(1 for n in notifs if not n["contexto"]["eligible"]),
        "treinos_concluidos": len(treinos),
        "dias_fechados": dias,
    }


def imprimir_resumo(r):
    print("\n== Resumo")
    for k in ("modo", "policy_version", "usuarios", "decisoes", "acao_recomendada", "acao_proposta",
              "acao_executada", "motivo_bloqueio", "desfechos", "janelas_censuradas", "base_contadores",
              "atualiza_politica", "treinaram_no_servico", "motivos_sem_treino",
              "propensity_executada", "propensity_recomendada", "p_enviar_elegiveis",
              "min_prob_braco_elegiveis", "pontos_inelegiveis", "treinos_concluidos", "dias_fechados"):
        print(f"  {k:24s} {json.dumps(r[k], ensure_ascii=False, default=str)}")


if __name__ == "__main__":
    sys.exit(main())
