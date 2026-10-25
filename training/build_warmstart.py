#!/usr/bin/env python3
"""Gera o artefato de warm-start da posterior servida (`SPYNO_WARMSTART_SNAPSHOT`).

Receita: a mesma dos experimentos do `spyno-decision` que mediram o warm-start
(`spyno_decision.evaluate.run_evaluation`, RQ2; `run_hybrid_ablation.py`;
`run_fbm_ablation.py`), com uma semente fixa:

  mundo     build_world(SimConfig(n_users=12, n_days=28, llm_mode="stub"),
                        Demographics(), seed_offset=1000 + s)
  log       rollout(mundo, MRTLoggingPolicy(p_notify=0,25), ResponseParams(),
                    cfg, outcome_seed=90_000 + s)
  ajuste    policy.warm_start(rows): Laplace sobre os pontos elegíveis do log
            randomizado (o warm_start original do spyno-decision, sem alteração)

com s = 0 por padrão. Nada aqui é aleatório além dessas sementes, e o ajuste
(IRLS) é determinístico: a mesma entrada gera o mesmo m, q.

Vetor. O warm-start é feito no vetor que o serviço usa (esquema v2, 13
dimensões = 12 atributos + viés, `spyno_service.features.FEATURES`). As
classes do `spyno-decision` vetorizam 16 posições (15 atributos + viés); o
serviço entrega `location_home`, `motion_stationary` e `fbm_obs` sempre zerados
(`policies/bandit._CompatibilidadeHerdada`), e as linhas do simulador recebem o
mesmo tratamento (`bandit.warm_start_rows`). As linhas do rollout já estão no
espaço da política (`notifs_today` ÷ 5), então não passam por `policy_obs`.
Diferença em relação ao RQ2: lá o vetor tinha `fbm_obs` e os sensores com os
valores do simulador, e a política era o TS disjoint com c = 0,03. Ver README.

`--crosscheck-decision PATH` refaz o ajuste com as classes de outro checkout do
`spyno-decision` (por exemplo o master) num subprocesso, e grava no artefato
se m e q saíram idênticos aos da cópia vendorizada.

Uso:
  SPYNO_SIM_PATH=../spyno-sim .venv/bin/python training/build_warmstart.py \\
      --crosscheck-decision ../_wt/decision-master
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORMAT = "spyno-warmstart/1"

#: Classe original do spyno-decision por fábrica servida (para o crosscheck).
_ORIGINAL = {"ts_hybrid": ("spyno_decision.hybrid", "HybridLogisticTS"),
             "ts_disjoint": ("spyno_decision.policy", "DisjointLogisticTS"),
             "ts_binary": ("spyno_decision.policy", "BinaryLogisticTS")}


def _sim_path() -> Path:
    env = os.getenv("SPYNO_SIM_PATH")
    candidates = [Path(env)] if env else [p / "spyno-sim" for p in ROOT.parents]
    for c in candidates:
        if (c / "spyno_sim" / "simulate.py").exists():
            return c.resolve()
    raise SystemExit("spyno-sim não encontrado: defina SPYNO_SIM_PATH")


def _git(path: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_only(rows_file: str, out_file: str, policy: str, cost: float,
             decision_path: str) -> None:
    """Modo do subprocesso: ajusta com o spyno_decision de `decision_path`."""
    sys.path.insert(0, str(Path(decision_path).resolve()))
    mod_name, cls_name = _ORIGINAL[policy]
    mod = importlib.import_module(mod_name)
    origin = Path(mod.__file__).resolve()
    if Path(decision_path).resolve() not in origin.parents:
        raise SystemExit(f"importou {origin}, não o checkout {decision_path}")
    pol = getattr(mod, cls_name)(cost=cost, seed=0)
    pol.warm_start(json.loads(Path(rows_file).read_text()))
    import numpy as np
    Path(out_file).write_text(json.dumps({"m": np.asarray(pol.m).tolist(),
                                          "q": np.asarray(pol.q).tolist(),
                                          "module": str(origin)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="ts_hybrid", choices=sorted(_ORIGINAL))
    ap.add_argument("--cost", type=float, default=0.05,
                    help="custo c da política servida (padrão decidido: 0,05)")
    ap.add_argument("--seed", type=int, default=0, help="s: mundo 1000+s, desfecho 90000+s")
    ap.add_argument("--users", type=int, default=12)
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--out", default=None,
                    help="padrão: warmstart/<policy>-v2-c<cost>.json")
    ap.add_argument("--crosscheck-decision", default=None, metavar="PATH")
    ap.add_argument("--fit-only", nargs=2, metavar=("ROWS", "OUT"), help=argparse.SUPPRESS)
    ap.add_argument("--decision-path", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.fit_only:
        fit_only(*args.fit_only, args.policy, args.cost, args.decision_path)
        return

    sim = _sim_path()
    for p in (str(sim), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import numpy as np
    from spyno_sim.config import Demographics
    from spyno_sim.response import ResponseParams
    from spyno_sim.simulate import MRTLoggingPolicy, SimConfig, build_world, rollout

    import spyno_decision
    from spyno_decision.features import DIM as DIM_INTERNAL
    from spyno_service.features import DIM, FEATURES, POLICY_SCHEMA
    from spyno_service.policies import bandit
    from spyno_service.policies.registry import build

    s = args.seed
    cfg = SimConfig(n_users=args.users, n_days=args.days, llm_mode="stub")
    world, llm_used = build_world(cfg, Demographics(), seed_offset=1000 + s)
    if llm_used != "stub":
        raise SystemExit(f"o mundo usou '{llm_used}', esperado stub")
    res = rollout(world, MRTLoggingPolicy(cfg.p_notify), ResponseParams(), cfg,
                  outcome_seed=90_000 + s)
    rows = res.rows
    eligible = sum(int(r["eligible"]) == 1 for r in rows)

    policy = build(args.policy, cost=args.cost)
    counts = bandit.warm_start(policy, rows)
    snap = policy.snapshot()
    if snap["schema"] != POLICY_SCHEMA or snap["features"] != list(FEATURES):
        raise SystemExit("snapshot fora do esquema servido")

    vendored = Path(spyno_decision.__file__).resolve().parent
    prov = {
        "spyno_sim": {"path": str(sim), **_git(sim)},
        "spyno_service": {**_git(ROOT)},
        "spyno_decision": {
            "origin": "cópia vendorizada em spyno-service/spyno_decision "
                      "(PROVENIENCIA.md: spyno-decision @ 03d5031)",
            "sha256": {f: _sha(vendored / f) for f in ("features.py", "policy.py", "hybrid.py")},
        },
    }

    if args.crosscheck_decision:
        with tempfile.TemporaryDirectory() as tmp:
            rows_file, out_file = Path(tmp) / "rows.json", Path(tmp) / "fit.json"
            rows_file.write_text(json.dumps(bandit.warm_start_rows(rows)))
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--policy", args.policy, "--cost", repr(args.cost),
                            "--fit-only", str(rows_file), str(out_file),
                            "--decision-path", args.crosscheck_decision],
                           check=True, cwd=tmp)
            other = json.loads(out_file.read_text())
        dp = Path(args.crosscheck_decision).resolve()
        same_m = np.array_equal(np.array(other["m"]), np.array(snap["m"]))
        same_q = np.array_equal(np.array(other["q"]), np.array(snap["q"]))
        prov["spyno_decision"]["crosscheck"] = {
            "path": str(dp), **_git(dp), "module": other["module"],
            "identical_m": bool(same_m), "identical_q": bool(same_q),
            "max_abs_diff_m": float(np.max(np.abs(np.array(other["m"]) - np.array(snap["m"])))),
        }
        print(f"crosscheck com {dp}: m idêntico={same_m}, q idêntico={same_q}")

    summary = (f"spyno_sim stub {args.users}u x {args.days}d, mundo 1000+{s}, "
               f"MRT p=0.25, desfecho 90000+{s}; {args.policy} c={args.cost:g}")
    artifact = {
        "format": FORMAT,
        "policy": args.policy,
        **snap,                              # m, q, cost, schema, features
        "recipe": {
            "summary": summary,
            "seed": s,
            "world": {"builder": "spyno_sim.simulate.build_world",
                      "sim_config": {"n_users": cfg.n_users, "n_days": cfg.n_days,
                                     "llm_mode": "stub", "seed": cfg.seed,
                                     "p_notify": cfg.p_notify,
                                     "max_notifications_day": cfg.max_notifications_day},
                      "seed_offset": 1000 + s},
            "logging_policy": f"MRTLoggingPolicy(p_notify={cfg.p_notify})",
            "response_params": "ResponseParams() (padrão)",
            "outcome_seed": 90_000 + s,
            "same_as": ["spyno_decision.evaluate.run_evaluation (RQ2)",
                        "run_hybrid_ablation.py (volume 12u x 28d)",
                        "run_fbm_ablation.py"],
        },
        "vector": {
            "service_dim": DIM,
            "policy_internal_dim": DIM_INTERNAL,
            "posterior_len": len(snap["m"]) if np.ndim(snap["m"]) == 1 else list(np.shape(snap["m"])),
            "zeroed_columns": list(bandit._CHAVES_HERDADAS),
            "notifs_today": "÷5 (já no espaço da política nas linhas do simulador)",
        },
        "fit": {"rows": len(rows), "eligible_rows": eligible,
                "warm_start_return": counts,
                "notifications_in_log": res.notifications},
        "provenance": prov,
    }
    out = Path(args.out) if args.out else ROOT / "warmstart" / f"{args.policy}-v2-c{args.cost:g}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(artifact, indent=1, ensure_ascii=False) + "\n"
    out.write_text(text)
    print(f"{out}: sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    print(f"  {len(rows)} linhas, {eligible} elegíveis; posterior {artifact['vector']['posterior_len']}")


if __name__ == "__main__":
    main()
