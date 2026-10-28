"""Avaliação multi-seed (F1b / Fase B do protocolo experimental).

Para cada seed:
  1. Mundo de WARM-START (personas próprias) com política MRT → warm-start dos TS.
     Personas de warm-start ≠ personas de avaliação (§7.1 do plano).
  2. Mundo de AVALIAÇÃO exógeno, compartilhado por todas as políticas.
  3. Métrica primária: E[treinos] = Σ p_true(braço escolhido) — exata e sensível
     à externalidade da fadiga. Regret condicional é diagnóstico.

O custo c da notificação é o preço-de-Pigou da externalidade de fadiga: o TS é
míope (bandit, não MDP), então c internaliza o dano futuro de uma notificação
ignorada. Varremos c — é hiperparâmetro de política, não do ambiente.

RQ3 é medida pelo argmax da POSTERIOR MÉDIA (o que a política aprendeu),
separado do ruído de exploração do TS: `type_match_acc` = fração das horas
elegíveis em que o braço de notificação preferido pela posterior coincide com o
verdadeiro melhor braço (chance ≈ 1/3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spyno_sim.config import Demographics
from spyno_sim.response import ResponseParams
from spyno_sim.simulate import MRTLoggingPolicy, SimConfig, build_world, rollout

from .baselines import AlwaysNotifyRandomArm, FBMHeuristic, NeverNotify
from .features import vectorize
from .policy import ARM_NAMES, DisjointLogisticTS, _sigmoid

NUMERIC_METRICS = ("expected_trainings_per_user_week", "notifs_per_expected_training",
                   "cum_regret", "regret_week1", "mean_fatigue",
                   "trainings_per_user_week", "notifications_total")
COST_GRID = (0.01, 0.03, 0.05)
PAIRING_POLICY = "ts_warm_c03"      # variante usada na análise RQ3


def _greedy_notify_arm(policy: DisjointLogisticTS, x: np.ndarray) -> int:
    """Braço de notificação preferido pela posterior média (sem sampling)."""
    p_hat = _sigmoid(policy.m[1:] @ x)   # c é igual entre braços de notificação
    return 1 + int(np.argmax(p_hat))


def _pairing_analysis(policy: DisjointLogisticTS, rows: list[dict]) -> dict:
    """RQ3: distribuição do braço preferido por regime + acurácia de pareamento."""
    eligible = [r for r in rows if int(r["eligible"]) == 1]
    match, dist = 0, {}
    regimes = {
        "m_dominant": lambda r: float(r["m_deficit"]) - float(r["a_deficit"]) > 0.15,
        "a_dominant": lambda r: float(r["a_deficit"]) - float(r["m_deficit"]) > 0.15,
        "both_low": lambda r: float(r["m_deficit"]) < 0.35 and float(r["a_deficit"]) < 0.35,
    }
    counts = {k: {a: 0 for a in ARM_NAMES[1:]} for k in regimes}
    n_reg = {k: 0 for k in regimes}
    for r in eligible:
        x = vectorize(r)
        greedy = _greedy_notify_arm(policy, x)
        true_best = 1 + int(np.argmax([float(r[f"p_true_{a}"]) for a in (1, 2, 3)]))
        if greedy == true_best:
            match += 1
        for k, pred in regimes.items():
            if pred(r):
                counts[k][ARM_NAMES[greedy]] += 1
                n_reg[k] += 1
    for k in regimes:
        total = max(n_reg[k], 1)
        dist[k] = {a: round(c / total, 3) for a, c in counts[k].items()}
        dist[k]["n"] = n_reg[k]
    return {"type_match_acc": match / max(len(eligible), 1), "greedy_dist": dist}


def run_evaluation(n_seeds: int = 10, n_users: int = 12, n_days: int = 28,
                   llm_mode: str = "stub", out_dir: str = "results") -> dict:
    demo = Demographics()
    params = ResponseParams()
    cfg = SimConfig(n_users=n_users, n_days=n_days, llm_mode=llm_mode)

    per_seed: dict[str, list[dict]] = {}
    pairing_runs = []
    warm_counts_last = None

    for s in range(n_seeds):
        # 1) warm-start em personas separadas
        warm_world, _ = build_world(cfg, demo, seed_offset=1000 + s)
        warm_res = rollout(warm_world, MRTLoggingPolicy(cfg.p_notify), params, cfg,
                           outcome_seed=90_000 + s)

        ts_variants = []
        for c in COST_GRID:
            ts = DisjointLogisticTS(cost=c, seed=500 + s,
                                    name=f"ts_warm_c{int(round(c * 100)):02d}")
            warm_counts_last = ts.warm_start(warm_res.rows)
            ts_variants.append(ts)
        ts_cold = DisjointLogisticTS(cost=0.03, seed=600 + s, name="ts_cold_c03")

        # 2) mundo de avaliação compartilhado
        eval_world, _ = build_world(cfg, demo, seed_offset=s)
        policies = [NeverNotify(), AlwaysNotifyRandomArm(),
                    MRTLoggingPolicy(cfg.p_notify), FBMHeuristic(),
                    *ts_variants, ts_cold]

        for i, pol in enumerate(policies):
            collect = pol.name == PAIRING_POLICY
            res = rollout(eval_world, pol, params, cfg,
                          outcome_seed=50_000 + s * 100 + i, collect_rows=collect)
            per_seed.setdefault(pol.name, []).append(res.metrics(n_users, n_days))
            if collect:
                pairing_runs.append(_pairing_analysis(pol, res.rows))
        print(f"  seed {s + 1}/{n_seeds} ok")

    # agregação média ± IC95
    summary = {}
    for name, runs in per_seed.items():
        agg = {}
        for metric in NUMERIC_METRICS:
            vals = np.array([r[metric] for r in runs], dtype=float)
            ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            agg[metric] = {"mean": round(float(vals.mean()), 3), "ci95": round(float(ci), 3)}
        arm_totals = {a: int(np.sum([r["arm_counts"][a] for r in runs])) for a in ARM_NAMES}
        agg["arm_share"] = {a: round(c / max(sum(arm_totals.values()), 1), 3)
                            for a, c in arm_totals.items()}
        summary[name] = agg

    accs = np.array([p["type_match_acc"] for p in pairing_runs])
    pairing = {
        "policy": PAIRING_POLICY,
        "type_match_acc_mean": round(float(accs.mean()), 3),
        "type_match_acc_ci95": round(float(1.96 * accs.std(ddof=1) / np.sqrt(len(accs))), 3)
        if len(accs) > 1 else 0.0,
        "chance_level": "~0.33",
        "greedy_dist_last_seed": pairing_runs[-1]["greedy_dist"] if pairing_runs else {},
    }

    # comparações PAREADAS (mesmo mundo por seed → diferença por seed elimina a
    # variância entre mundos; IC muito mais justo que comparar médias marginais)
    def paired(a: str, b: str, metric: str = "expected_trainings_per_user_week") -> dict:
        da = np.array([r[metric] for r in per_seed[a]], dtype=float)
        db = np.array([r[metric] for r in per_seed[b]], dtype=float)
        diff = da - db
        ci = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
        return {"diff_mean": round(float(diff.mean()), 3), "ci95": round(float(ci), 3),
                "significant": bool(abs(diff.mean()) > ci)}

    paired_diffs = {
        "ts_warm_c05_vs_mrt_random": paired("ts_warm_c05", "mrt_random"),
        "ts_warm_c05_vs_fbm_gt40": paired("ts_warm_c05", "fbm_gt40"),
        "ts_warm_c05_vs_always": paired("ts_warm_c05", "always"),
        "ts_warm_c05_vs_never": paired("ts_warm_c05", "never"),
        "rq2_ts_warm_c03_vs_ts_cold_c03": paired("ts_warm_c03", "ts_cold_c03"),
        "rq2_regret_week1_warm_vs_cold": paired("ts_warm_c03", "ts_cold_c03", "regret_week1"),
    }

    report = {
        "config": {"n_seeds": n_seeds, "n_users": n_users, "n_days": n_days,
                   "cost_grid": list(COST_GRID),
                   "warm_start_samples_per_arm_last_seed": warm_counts_last},
        "policies": summary,
        "paired_diffs": paired_diffs,
        "rq3_pairing": pairing,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out / "per_seed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_seed, f, ensure_ascii=False, indent=2)
    return report
