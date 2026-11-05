"""Equivalência da cópia vendorizada do `spyno_decision` com o `spyno-decision` master.

A cópia em `spyno_decision/` veio do commit 03d5031 (julho). O master atual tem a
correção I3 (`EligibilityGate`), o `hybrid_x` de dimensão genérica e opções de
ablação (`zero_features`, `obs_transform`, `EXTRA_DIM`). Este script verifica, na
configuração servida, se essas diferenças mudam alguma decisão:

- política `ts_hybrid`, c = 0,1, clipping p05 (`ClipConfig()`), M = 1000 amostras;
- esquema v2 (12 atributos + viés, três colunas internas zeradas pelo adaptador);
- posterior inicial do artefato `warmstart/ts_hybrid-v2-c0.1.json`.

Cada implementação roda num subprocesso próprio (o pacote tem o mesmo nome): um
com a cópia vendorizada, outro com o master à frente do `sys.path`. Os dois
executam a mesma bateria com as mesmas sementes e gravam tudo em JSON (floats
com `repr`, ida e volta exata). A comparação é bit a bit.

Cenários
--------
A. **Pilha servida, em processo.** `bandit.ts_hybrid` + `apply_warmstart` +
   adaptador + `ClippedPolicy`. Para cada contexto da bateria: (a) os 4 vetores
   híbridos; (b) a distribuição clipada π e a ação sorteada; depois updates
   (inclusive o caso da I3: decisão inelegível seguida de update do mesmo
   usuário) e um fechamento de dia; (c) m e q depois de cada dia. Três dias.
B. **Pilha servida via HTTP.** `create_app` com o mesmo warm-start, SQLite em
   arquivo temporário, /v1/decide → /v1/feedback → /v1/end-day por três dias.
   Compara ações, π, propensões, motivos de não-treino e o digest da posterior.
C. **Classe pura, fora do caminho servido.** `HybridLogisticTS.decide` seguido
   de `update` para o mesmo usuário com ponto inelegível: é aqui que a I3 age
   (no master o update é descartado). Esperado: diverge. Serve de controle de
   que a bateria enxerga a diferença quando ela é alcançável.

Uso::

    uv run python scripts/check_vendor_equivalence.py \\
        --decision-path ../_wt/decision-master [--out results.json]

Sai com código 1 se A ou B divergirem e 2 se o controle C não divergir
(a bateria não estaria comparando implementações distintas).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WARM = ROOT / "warmstart" / "ts_hybrid-v2-c0.1.json"
FILES = ("features.py", "policy.py", "hybrid.py", "baselines.py", "__init__.py")
SEED_CTX = 2026
N_CTX = 400
N_DAYS = 3


# ---- bateria (roda no subprocesso) ------------------------------------------

def _contexts(rng) -> list[tuple[dict, bool]]:
    """Contextos do contrato. Inclui inelegíveis e extremos."""
    out = []
    for _ in range(N_CTX):
        tod = rng.integers(0, 4)  # 0 manhã, 1 meio-dia, 2 noite, 3 nenhum
        ctx = {
            "in_free_window": float(rng.integers(0, 2)),
            "valence_pos": float(rng.random()),
            "is_weekend": float(rng.integers(0, 2)),
            "recency": float(rng.random()),
            "already_trained": float(rng.integers(0, 2)),
            "streak_norm": float(rng.random()),
            "emp_adhesion": float(rng.random()),
            "fatigue": float(rng.random()),
            "notifs_today": float(rng.integers(0, 7)),
            "morning": float(tod == 0), "midday": float(tod == 1),
            "evening": float(tod == 2),
            "fbm_obs": float(rng.random()),
            "hour": int(rng.integers(0, 24)),
            "minutes_since_last": None,
        }
        out.append((ctx, bool(rng.random() < 0.7)))
    base = out[0][0]
    zeros = {**{k: 0.0 for k in base if k not in ("hour", "minutes_since_last")},
             "hour": 12, "minutes_since_last": None}
    ones = {**{k: 1.0 for k in base if k not in ("hour", "minutes_since_last")},
            "notifs_today": 10.0, "hour": 23, "minutes_since_last": None}
    out += [(zeros, True), (zeros, False), (ones, True), (ones, False)]
    return out


def _f(a) -> list:
    import numpy as np
    return np.asarray(a, dtype=float).ravel().tolist()


def _served_policy():
    from spyno_service.policies import bandit
    from spyno_service.policies.clipping import ClipConfig
    from spyno_service.posterior import apply_warmstart

    pol = bandit.ts_hybrid(cost=0.1, seed=0, clip=ClipConfig(), n_samples=1000)
    warm = apply_warmstart(pol, "ts_hybrid", WARM)
    return pol, warm


def scenario_served() -> dict:
    import numpy as np

    from spyno_decision.features import vectorize as vec_interno
    from spyno_decision.hybrid import hybrid_x

    pol, warm = _served_policy()
    adapter = pol._inner                      # _CompatibilidadeHerdada
    core = adapter._inner                     # HybridServableTS
    rng = np.random.default_rng(SEED_CTX)
    ctxs = _contexts(rng)
    y_rng = np.random.default_rng(SEED_CTX + 1)
    res = {"warm_posterior_digest": warm["posterior_digest"],
           "m0": _f(core.m), "q0": _f(core.q), "cost": core.cost,
           "clip": pol.clip.tag(), "n_samples": pol.n_samples, "days": []}
    for _ in range(N_DAYS):
        dres = {"hybrid_x": [], "pi": [], "arm": [], "updates": 0}
        for i, (ctx, elig) in enumerate(ctxs):
            uid = f"u{i % 7}"
            x = vec_interno(adapter._obs(ctx))
            dres["hybrid_x"].append([_f(hybrid_x(x, a)) for a in range(4)])
            arm, pi = pol.decide_with_distribution(uid, ctx, elig)
            dres["arm"].append(arm)
            dres["pi"].append(_f(pi))
            y = int(y_rng.random() < 0.3)
            # Caminho do /v1/feedback: só pontos elegíveis treinam. O caso da
            # I3 (decisão inelegível seguida de update do mesmo usuário) é
            # exercitado de propósito em 1 de cada 4 inelegíveis: o serviço não
            # o produz, mas, se produzisse, o gate do master teria de ignorá-lo
            # para divergir.
            if elig or i % 4 == 0:
                pol.update(uid, ctx, arm, y)
                dres["updates"] += 1
        pol.end_day()
        dres["m"], dres["q"] = _f(core.m), _f(core.q)
        dres["skipped_by_gate"] = getattr(core, "n_updates_skipped", None)
        res["days"].append(dres)
    return res


def scenario_http() -> dict:
    from dataclasses import replace
    from datetime import datetime, timedelta, timezone

    import numpy as np
    from fastapi.testclient import TestClient

    from spyno_service.app import create_app
    from spyno_service.config import Config
    from spyno_service.policies import bandit
    from spyno_service.policies.clipping import ClipConfig
    from spyno_service.store import Store

    tok = "token-de-verificacao"
    with tempfile.TemporaryDirectory() as tmp:
        cfg = replace(Config(), database_url=f"sqlite+pysqlite:///{tmp}/v.db",
                      policy="ts_hybrid", shadow=False, decision_token=tok,
                      warmstart_snapshot=str(WARM), decide_timeout_s=60.0,
                      propensity_samples=1000, clip_min=0.05, clip_max=0.9,
                      arm_mix=0.1, arm_floor=0.05 / 3.0)
        pol = bandit.ts_hybrid(cost=0.1, seed=0, clip=ClipConfig.from_config(cfg),
                               n_samples=cfg.propensity_samples)
        app = create_app(policy=pol, store=Store(cfg.database_url), config=cfg)
        rng = np.random.default_rng(SEED_CTX + 2)
        ctxs = _contexts(np.random.default_rng(SEED_CTX))
        res = {"days": []}
        with TestClient(app, headers={"Authorization": f"Bearer {tok}"}) as c:
            st = c.get("/v1/policy-state").json()
            res["policy_version"] = st["policy_version"]
            res["digest0"] = st["posterior_digest"]
            t0 = datetime(2026, 11, 2, 0, 0, tzinfo=timezone(timedelta(hours=-3)))
            for day in range(N_DAYS):
                dres = {"decide": [], "feedback": []}
                for h in range(24):
                    for u in range(5):
                        i = (day * 24 * 5 + h * 5 + u) % len(ctxs)
                        ctx, elig = ctxs[i]
                        ctx = {**ctx, "hour": h}
                        point = t0 + timedelta(days=day, hours=h)
                        r = c.post("/v1/decide", json={
                            "user_key": f"u{u}", "decision_point": point.isoformat(),
                            "context": ctx, "eligible": elig})
                        r.raise_for_status()
                        d = r.json()
                        dres["decide"].append({k: d.get(k) for k in (
                            "recommended_action", "executed_action", "proposed_action",
                            "propensity", "recommended_propensity", "arm_probabilities",
                            "fallback", "guardrail_reason")})
                        outcome = ["responded", "not_responded", "censored"][
                            int(rng.integers(0, 3))]
                        fb = c.post("/v1/feedback", json={
                            "decision_id": d["decision_id"],
                            "observed_at": (point + timedelta(minutes=30)).isoformat(),
                            "outcome": outcome, "value": 1}).json()
                        dres["feedback"].append([fb.get("trained"),
                                                 fb.get("train_skip_reason")])
                e = c.post("/v1/end-day", json={
                    "date": (t0 + timedelta(days=day)).date().isoformat()}).json()
                dres["end_day"] = e
                dres["digest"] = c.get("/v1/policy-state").json()["posterior_digest"]
                res["days"].append(dres)
        core = pol._inner._inner
        res["m"], res["q"] = _f(core.m), _f(core.q)
        res["skipped_by_gate"] = getattr(core, "n_updates_skipped", None)
    return res


def scenario_class_i3() -> dict:
    """HybridLogisticTS pura: decide(inelegível) → update do mesmo usuário."""
    import numpy as np

    from spyno_decision.hybrid import HybridLogisticTS
    from spyno_service.features import policy_obs
    from spyno_service.policies.bandit import _CHAVES_HERDADAS

    pol = HybridLogisticTS(cost=0.1, seed=0)
    art = json.loads(WARM.read_text())
    pol.m, pol.q = np.array(art["m"]), np.array(art["q"])
    ctxs = _contexts(np.random.default_rng(SEED_CTX))
    y_rng = np.random.default_rng(SEED_CTX + 3)
    arms = []
    for i, (ctx, elig) in enumerate(ctxs):
        obs = {**policy_obs(ctx), **{k: 0.0 for k in _CHAVES_HERDADAS}}
        arm, _ = pol.decide(f"u{i % 7}", obs, elig)
        arms.append(arm)
        pol.update(f"u{i % 7}", obs, arm, int(y_rng.random() < 0.3))
    pol.end_day()
    return {"arms": arms, "m": _f(pol.m), "q": _f(pol.q),
            "skipped_by_gate": getattr(pol, "n_updates_skipped", None)}


def run_battery(out: str, decision_path: str | None) -> None:
    os.environ["SPYNO_LEGACY_INELIGIBLE_UPDATES"] = "0"   # padrão do master
    os.environ.setdefault("SPYNO_DB_URL", "sqlite+pysqlite:///:memory:")
    sys.path.insert(0, str(ROOT))
    if decision_path:
        sys.path.insert(0, str(Path(decision_path).resolve()))
    import spyno_decision
    origin = Path(spyno_decision.__file__).resolve().parent
    expected = (Path(decision_path).resolve() if decision_path else ROOT) / "spyno_decision"
    if origin != expected:
        raise SystemExit(f"importou {origin}, esperado {expected}")
    res = {"module": str(origin), "served": scenario_served(), "http": scenario_http(),
           "class_i3": scenario_class_i3()}
    Path(out).write_text(json.dumps(res))


# ---- orquestração e comparação ---------------------------------------------

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git(path: Path) -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(path), *a], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def _diff(a, b, path="") -> list[str]:
    """Lista os caminhos onde a e b diferem (igualdade exata de floats)."""
    if type(a) is not type(b):
        return [f"{path}: tipo {type(a).__name__} × {type(b).__name__}"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: ausente de um lado")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: tamanho {len(a)} × {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: {a!r} × {b!r}"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decision-path", default=None,
                    help="checkout do spyno-decision a comparar (padrão: o primeiro "
                         "_wt/decision-master acima do repositório)")
    ap.add_argument("--out", default=None, help="grava o relatório JSON aqui")
    ap.add_argument("--run-battery", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-decision-path", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.run_battery:
        run_battery(args.run_battery, args.child_decision_path)
        return

    if args.decision_path:
        master = Path(args.decision_path).resolve()
    else:
        cands = [p / "_wt" / "decision-master" for p in ROOT.parents]
        found = next((c for c in cands if (c / "spyno_decision" / "hybrid.py").exists()), None)
        if found is None:
            raise SystemExit("spyno-decision master não encontrado: use --decision-path")
        master = found.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        outs = {}
        for tag, dpath in (("vendored", None), ("master", str(master))):
            outs[tag] = Path(tmp) / f"{tag}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--run-battery", str(outs[tag])]
            if dpath:
                cmd += ["--child-decision-path", dpath]
            subprocess.run(cmd, check=True, cwd=ROOT,
                           env={**os.environ, "PYTHONPATH": ""})
        v, m = (json.loads(outs[t].read_text()) for t in ("vendored", "master"))

    report = {
        "warmstart": {"path": str(WARM.relative_to(ROOT)), "sha256": _sha(WARM)},
        "vendored": {"module": v["module"], "service": _git(ROOT),
                     "sha256": {f: _sha(ROOT / "spyno_decision" / f) for f in FILES}},
        "master": {"module": m["module"], **_git(master),
                   "sha256": {f: _sha(master / "spyno_decision" / f) for f in FILES}},
        "config": {"policy": "ts_hybrid", "cost": v["served"]["cost"],
                   "clip": v["served"]["clip"], "n_samples": v["served"]["n_samples"],
                   "n_contexts": N_CTX + 4, "days": N_DAYS,
                   "policy_version": v["http"]["policy_version"]},
        "counts": {
            "served_decisions": N_DAYS * (N_CTX + 4),
            "served_updates": sum(d["updates"] for d in v["served"]["days"]),
            "http_decisions": sum(len(d["decide"]) for d in v["http"]["days"]),
            "http_trained": sum(sum(1 for t, _ in d["feedback"] if t)
                                for d in v["http"]["days"]),
            "master_gate_skipped_served": m["served"]["days"][-1]["skipped_by_gate"],
            "master_gate_skipped_http": m["http"]["skipped_by_gate"],
            "master_gate_skipped_class": m["class_i3"]["skipped_by_gate"],
        },
    }
    # O contador do gate só existe no master (None no vendorizado): é metadado,
    # já reportado acima, e sai da comparação.
    for res in (v, m):
        for d in res["served"]["days"]:
            d.pop("skipped_by_gate")
        for k in ("http", "class_i3"):
            res[k].pop("skipped_by_gate")
    report["diffs"] = {k: _diff(v[k], m[k], k) for k in ("served", "http", "class_i3")}
    ok = not report["diffs"]["served"] and not report["diffs"]["http"]
    report["equivalent_in_served_config"] = ok
    # Controle: a bateria precisa enxergar a I3 onde ela é alcançável. Sem isso,
    # "idêntico" pode ser só sinal de que os dois lados importaram o mesmo código.
    control_ok = bool(report["diffs"]["class_i3"]) and \
        bool(report["counts"]["master_gate_skipped_class"])
    report["control_detects_i3"] = control_ok

    print(f"vendorizado: {v['module']}")
    print(f"master:      {m['module']} @ {report['master']['commit']}"
          f"{' (sujo)' if report['master']['dirty'] else ''}")
    print(f"warm-start:  {report['warmstart']['path']} sha256 {report['warmstart']['sha256'][:16]}")
    print(f"config:      {report['config']}")
    print(f"contagens:   {report['counts']}")
    for k, d in report["diffs"].items():
        print(f"[{k}] {'IDÊNTICO' if not d else f'{len(d)} diferenças'}")
        for line in d[:5]:
            print(f"    {line}")
    print("equivalente na configuração servida:", "SIM" if ok else "NÃO")
    if not control_ok:
        print("CONTROLE FALHOU: a classe pura não divergiu (ou o gate do master não "
              "descartou nada). A bateria não está comparando implementações "
              "distintas; o resultado acima não vale.", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    sys.exit(0 if ok and control_ok else (2 if not control_ok else 1))


if __name__ == "__main__":
    main()
