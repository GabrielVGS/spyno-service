"""Experimentos F2 — o que fecha o capítulo de resultados.

  ablations()            — (b) K+1 vs binário; (d) com/sem feature de fadiga.
                           [(c) LLM vs Stub requer ANTHROPIC_API_KEY — pendente]
  sensitivity()          — RQ4: perturba o mundo (one-at-a-time) e verifica se o
                           RANKING das políticas se mantém.
  ope_analysis()         — Fase C: Replay/IPS/SNIPS/DR sobre logs MRT vs. valor
                           verdadeiro (p_true) — valida os estimadores e
                           quantifica o viés de estado da OPE de bandit.
  reward_model_quality() — Fase A: AUC e calibração do modelo de recompensa em
                           dados held-out (personas novas).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from spyno_sim.config import Demographics
from spyno_sim.response import ResponseParams
from spyno_sim.simulate import MRTLoggingPolicy, SimConfig, build_world, rollout

from .baselines import AlwaysNotifyRandomArm, FBMHeuristic, NeverNotify
from .policy import BinaryLogisticTS, DisjointLogisticTS, _sigmoid

BEST_COST = 0.05  # vencedor do sweep da Fase B (ver evaluate.py / README)


def _ci95(vals: np.ndarray) -> float:
    return float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0


def _paired(per_seed: dict, a: str, b: str,
            metric: str = "expected_trainings_per_user_week") -> dict:
    da = np.array([r[metric] for r in per_seed[a]], dtype=float)
    db = np.array([r[metric] for r in per_seed[b]], dtype=float)
    diff = da - db
    ci = _ci95(diff)
    return {"diff_mean": round(float(diff.mean()), 3), "ci95": round(ci, 3),
            "significant": bool(abs(diff.mean()) > ci)}


# ----------------------------------------------------------------------------
def ablations(n_seeds: int = 10, n_users: int = 12, n_days: int = 28) -> dict:
    demo, params = Demographics(), ResponseParams()
    cfg = SimConfig(n_users=n_users, n_days=n_days, llm_mode="stub")
    per_seed: dict[str, list[dict]] = {}

    for s in range(n_seeds):
        warm_world, _ = build_world(cfg, demo, seed_offset=1000 + s)
        warm_res = rollout(warm_world, MRTLoggingPolicy(cfg.p_notify), params, cfg,
                           outcome_seed=90_000 + s)

        variants = [
            DisjointLogisticTS(cost=BEST_COST, seed=500 + s, name="ts_full"),
            DisjointLogisticTS(cost=BEST_COST, seed=500 + s, name="ts_no_fatigue",
                               zero_features=("fatigue",)),
            BinaryLogisticTS(cost=BEST_COST, seed=500 + s, name="ts_binary"),
        ]
        for v in variants:
            v.warm_start(warm_res.rows)

        eval_world, _ = build_world(cfg, demo, seed_offset=s)
        for i, pol in enumerate(variants):
            res = rollout(eval_world, pol, params, cfg,
                          outcome_seed=60_000 + s * 100 + i, collect_rows=False)
            per_seed.setdefault(pol.name, []).append(res.metrics(n_users, n_days))
        print(f"  [ablations] seed {s + 1}/{n_seeds}")

    summary = {
        name: {"expected_trainings_per_user_week": {
            "mean": round(float(np.mean([r["expected_trainings_per_user_week"] for r in runs])), 3),
            "ci95": round(_ci95(np.array([r["expected_trainings_per_user_week"] for r in runs])), 3)}}
        for name, runs in per_seed.items()
    }
    return {
        "summary": summary,
        "paired": {
            "full_vs_no_fatigue": _paired(per_seed, "ts_full", "ts_no_fatigue"),
            "full_vs_binary": _paired(per_seed, "ts_full", "ts_binary"),
        },
        "llm_vs_stub": "PENDENTE: requer ANTHROPIC_API_KEY (rodar com --llm anthropic)",
    }


# ----------------------------------------------------------------------------
PERTURBATIONS = {
    "base": {},
    "intercept_-6": {"intercept": -6.0},
    "intercept_-8": {"intercept": -8.0},
    "u_base_0.3": {"u_base": 0.3},
    "u_base_0.9": {"u_base": 0.9},
    "fatigue_step_0.03": {"fatigue_step": 0.03},
    "fatigue_step_0.08": {"fatigue_step": 0.08},
    "fatigue_decay_0.5": {"fatigue_decay": 0.5},
    "fatigue_decay_0.85": {"fatigue_decay": 0.85},
}


def sensitivity(n_seeds: int = 4, n_users: int = 12, n_days: int = 28) -> dict:
    """Para cada mundo perturbado, o TS (re-warm-started NO MESMO mundo) segue
    no topo do ranking? RQ4 exige robustez do RANKING, não dos valores."""
    demo = Demographics()
    cfg = SimConfig(n_users=n_users, n_days=n_days, llm_mode="stub")
    out = {}

    for pname, changes in PERTURBATIONS.items():
        params = replace(ResponseParams(), **changes)
        per_seed: dict[str, list[float]] = {}
        for s in range(n_seeds):
            warm_world, _ = build_world(cfg, demo, seed_offset=1000 + s)
            warm_res = rollout(warm_world, MRTLoggingPolicy(cfg.p_notify), params, cfg,
                               outcome_seed=90_000 + s)
            ts = DisjointLogisticTS(cost=BEST_COST, seed=500 + s, name="ts_warm")
            ts.warm_start(warm_res.rows)

            eval_world, _ = build_world(cfg, demo, seed_offset=s)
            policies = [ts, MRTLoggingPolicy(cfg.p_notify), FBMHeuristic(),
                        AlwaysNotifyRandomArm(), NeverNotify()]
            for i, pol in enumerate(policies):
                res = rollout(eval_world, pol, params, cfg,
                              outcome_seed=70_000 + s * 100 + i, collect_rows=False)
                per_seed.setdefault(pol.name, []).append(
                    res.metrics(n_users, n_days)["expected_trainings_per_user_week"])

        means = {name: round(float(np.mean(v)), 3) for name, v in per_seed.items()}
        ranking = sorted(means, key=means.get, reverse=True)
        diff = np.array(per_seed["ts_warm"]) - np.array(
            [max(vals) for vals in zip(*[per_seed[n] for n in per_seed if n != "ts_warm"])])
        out[pname] = {
            "params_changed": changes,
            "means": means,
            "ranking": ranking,
            "ts_is_top1": ranking[0] == "ts_warm",
            "ts_minus_best_baseline_paired": {
                "diff_mean": round(float(diff.mean()), 3), "ci95": round(_ci95(diff), 3)},
        }
        print(f"  [sensitivity] {pname}: top1={ranking[0]}")
    out["ranking_robust"] = all(v["ts_is_top1"] for k, v in out.items() if isinstance(v, dict))
    return out


# ----------------------------------------------------------------------------
def _greedy_action(policy: DisjointLogisticTS, r: dict) -> int:
    x = policy._vec(r)
    p_hat = _sigmoid(policy.m @ x)
    utility = p_hat - policy.cost * np.array([0.0, 1.0, 1.0, 1.0])
    return int(np.argmax(utility))


def ope_analysis(n_seeds: int = 10, n_users: int = 12, n_days: int = 28,
                 clip: float = 10.0) -> dict:
    """Fase C. Alvo: política GREEDY (posterior média) do TS warm-started.
    Logs: MRT em personas held-out. Como o simulador expõe p_true, comparamos
    cada estimador com dois valores verdadeiros:
      - true_myopic: E[p_true(π(x))] na TRAJETÓRIA DE ESTADOS da política de log
        (o que a OPE de bandit estima de fato);
      - true_rollout: valor de π rodando no mundo (com a SUA trajetória de
        fadiga). A diferença é o viés de estado — limitação declarada da OPE.
    """
    demo, params = Demographics(), ResponseParams()
    cfg = SimConfig(n_users=n_users, n_days=n_days, llm_mode="stub")
    est = {k: [] for k in ("ips", "snips", "dr", "replay",
                            "true_myopic", "true_rollout", "logging_value")}

    for s in range(n_seeds):
        warm_world, _ = build_world(cfg, demo, seed_offset=1000 + s)
        warm_res = rollout(warm_world, MRTLoggingPolicy(cfg.p_notify), params, cfg,
                           outcome_seed=90_000 + s)
        ts = DisjointLogisticTS(cost=BEST_COST, seed=500 + s)
        ts.warm_start(warm_res.rows)

        log_world, _ = build_world(cfg, demo, seed_offset=s)
        log_res = rollout(log_world, MRTLoggingPolicy(cfg.p_notify), params, cfg,
                          outcome_seed=80_000 + s)
        rows = [r for r in log_res.rows if int(r["eligible"]) == 1]

        ips_terms, snips_w, dr_terms, rep_r, rep_n = [], [], [], 0.0, 0
        true_pi, log_val = [], []
        for r in rows:
            a_log = int(r["action"])
            prop = float(r["propensity"])
            rew = int(r["action_performed"])
            a_pi = _greedy_action(ts, r)
            x = ts._vec(r)
            q_hat = _sigmoid(ts.m @ x)                    # modelo de recompensa (DR)
            match = float(a_log == a_pi)
            w = min(match / prop, clip)
            ips_terms.append(w * rew)
            snips_w.append(w)
            dr_terms.append(q_hat[a_pi] + w * (rew - q_hat[a_log]))
            if a_log == a_pi:
                rep_r += rew
                rep_n += 1
            true_pi.append(float(r[f"p_true_{a_pi}"]))
            log_val.append(float(r[f"p_true_{a_log}"]))

        est["ips"].append(float(np.mean(ips_terms)))
        est["snips"].append(float(np.sum(ips_terms) / max(np.sum(snips_w), 1e-9)))
        est["dr"].append(float(np.mean(dr_terms)))
        est["replay"].append(rep_r / max(rep_n, 1))
        est["true_myopic"].append(float(np.mean(true_pi)))
        est["logging_value"].append(float(np.mean(log_val)))

        pi_res = rollout(log_world, DisjointLogisticTSGreedy(ts), params, cfg,
                         outcome_seed=81_000 + s, collect_rows=True)
        pi_elig = [r for r in pi_res.rows if int(r["eligible"]) == 1]
        est["true_rollout"].append(
            float(np.mean([float(r[f"p_true_{int(r['action'])}"]) for r in pi_elig])))
        print(f"  [ope] seed {s + 1}/{n_seeds}")

    def agg(k):
        v = np.array(est[k])
        return {"mean": round(float(v.mean()), 4), "ci95": round(_ci95(v), 4)}

    report = {k: agg(k) for k in est}
    tm = report["true_myopic"]["mean"]
    report["abs_error_vs_true_myopic"] = {
        k: round(abs(report[k]["mean"] - tm), 4) for k in ("ips", "snips", "dr", "replay")}
    report["state_bias_true_rollout_minus_myopic"] = round(
        report["true_rollout"]["mean"] - tm, 4)
    return report


class DisjointLogisticTSGreedy:
    """Wrapper: executa a política greedy (posterior média) de um TS treinado."""

    name = "ts_greedy"

    def __init__(self, ts: DisjointLogisticTS):
        self.ts = ts

    def start_user(self, user_id): pass
    def update(self, user_id, obs, arm, reward): pass
    def end_day(self): pass

    def decide(self, user_id, obs, eligible, rng):
        if not eligible:
            return 0, 1.0
        return _greedy_action(self.ts, obs), 1.0


# ----------------------------------------------------------------------------
def reward_model_quality(n_seeds: int = 10, n_users: int = 12, n_days: int = 28) -> dict:
    """Fase A: AUC (Mann-Whitney) e calibração (ECE em decis) do modelo de
    recompensa warm-started, avaliado em logs MRT de personas held-out."""
    demo, params = Demographics(), ResponseParams()
    cfg = SimConfig(n_users=n_users, n_days=n_days, llm_mode="stub")
    aucs, eces = [], []

    for s in range(n_seeds):
        warm_world, _ = build_world(cfg, demo, seed_offset=1000 + s)
        warm_res = rollout(warm_world, MRTLoggingPolicy(cfg.p_notify), params, cfg,
                           outcome_seed=90_000 + s)
        ts = DisjointLogisticTS(cost=BEST_COST, seed=500 + s)
        ts.warm_start(warm_res.rows)

        test_world, _ = build_world(cfg, demo, seed_offset=s)
        test_res = rollout(test_world, MRTLoggingPolicy(cfg.p_notify), params, cfg,
                           outcome_seed=80_000 + s)
        rows = [r for r in test_res.rows if int(r["eligible"]) == 1]
        scores = np.array([float(_sigmoid(ts.m[int(r["action"])] @ ts._vec(r))) for r in rows])
        y = np.array([int(r["action_performed"]) for r in rows])

        pos, neg = scores[y == 1], scores[y == 0]
        ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
        auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        aucs.append(float(auc))

        bins = np.clip((scores * 10).astype(int), 0, 9)
        ece = sum(np.abs(scores[bins == b].mean() - y[bins == b].mean()) * np.sum(bins == b)
                  for b in range(10) if np.any(bins == b)) / len(y)
        eces.append(float(ece))
        print(f"  [fase A] seed {s + 1}/{n_seeds}")

    return {"auc": {"mean": round(float(np.mean(aucs)), 4), "ci95": round(_ci95(np.array(aucs)), 4)},
            "ece": {"mean": round(float(np.mean(eces)), 4), "ci95": round(_ci95(np.array(eces)), 4)}}


# ----------------------------------------------------------------------------
def run_all(out_dir: str = "results") -> dict:
    report = {}
    print("[1/4] Fase A — qualidade do modelo de recompensa (held-out)")
    report["fase_a_reward_model"] = reward_model_quality()
    print("[2/4] Ablations")
    report["ablations"] = ablations()
    print("[3/4] OPE (Fase C)")
    report["ope"] = ope_analysis()
    print("[4/4] Sensibilidade (RQ4)")
    report["sensitivity"] = sensitivity()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "f2_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
