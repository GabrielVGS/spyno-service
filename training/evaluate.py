"""Compara o PPO treinado às demais políticas no mesmo mundo simulado.

Mesmas personas, mesmas sementes de desfecho, mesma função de resposta: a única
coisa que varia é a política. É a condição mínima para que a diferença possa ser
atribuída ao método.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SIM = Path(os.getenv("SPYNO_SIM_PATH",
                      Path(__file__).resolve().parents[2] / "spyno-sim"))
for p in (str(_SIM), str(Path(__file__).resolve().parents[1])):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--users", type=int, default=24)
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--seed", type=int, default=777, help="mundo de avaliação, disjunto do treino")
    ap.add_argument("--checkpoint", default="checkpoints/ppo_observable.zip")
    args = ap.parse_args()

    from spyno_sim.config import Demographics
    from spyno_sim.response import ResponseParams
    from spyno_sim.simulate import SimConfig, build_world, rollout

    from spyno_service.policies.ppo import PPOPolicy
    from spyno_service.policies.registry import build as build_policy

    cfg = SimConfig(n_users=args.users, n_days=args.days, seed=args.seed,
                    llm_mode="stub")
    world, _ = build_world(cfg, Demographics())
    params = ResponseParams()

    candidates = {
        "never": build_policy("never"),
        "always": build_policy("always"),
        "ts_hybrid": build_policy("ts_hybrid"),
        "ts_disjoint": build_policy("ts_disjoint"),
    }
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        candidates["ppo"] = PPOPolicy(checkpoint=ckpt, deterministic=False)

    rows = []
    for name, policy in candidates.items():
        res = rollout(world, policy, params, cfg,
                      outcome_seed=cfg.seed + 10_000, collect_rows=False)
        m = res.metrics(args.users, args.days)
        rows.append({"policy": name, **m})

    campos = ["policy", "trainings_per_user_week", "notifications_per_user_week",
              "cum_regret", "notifications_per_training"]
    disponiveis = [c for c in campos if c == "policy" or c in rows[0]]
    largura = max(len(r["policy"]) for r in rows) + 2
    print("\n" + "policy".ljust(largura) + "  ".join(f"{c:>26}" for c in disponiveis[1:]))
    for r in rows:
        print(r["policy"].ljust(largura)
              + "  ".join(f"{r.get(c, float('nan')):>26.4f}" for c in disponiveis[1:]))
    print("\nmétricas completas:")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
