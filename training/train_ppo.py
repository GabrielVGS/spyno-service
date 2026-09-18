"""Treina o PPO sobre o vetor observável e salva um checkpoint servível.

Os cinco checkpoints do trabalho anterior observam motivação, habilidade e sono
exatos, sinais que a formulação atual não coleta. Este treino produz o modelo
que a política `ppo` do serviço consegue carregar: mesma observação, mesmo
espaço de ações e mesma utilidade dos bandits.

    python training/train_ppo.py --steps 200000 --out checkpoints/ppo_observable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# O simulador vive num diretório irmão e não é instalável como pacote.
_SIM = Path(os.getenv("SPYNO_SIM_PATH",
                      Path(__file__).resolve().parents[2] / "spyno-sim"))
if _SIM.is_dir() and str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.ppo_env import EnvConfig, SpynoNotificationEnv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--users", type=int, default=12)
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost", type=float, default=0.05,
                    help="custo por notificação; deve ser o mesmo dos bandits")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=2048,
                    help="passos por atualização; valores muito baixos colapsam a política")
    ap.add_argument("--ent-coef", type=float, default=0.01,
                    help="pressão de exploração; com 0 o agente colapsa em abster-se")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="checkpoints/ppo_observable")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(SpynoNotificationEnv(EnvConfig(
        n_users=args.users, n_days=args.days, seed=args.seed, cost=args.cost,
    )))

    model = PPO("MlpPolicy", env, learning_rate=args.lr, n_steps=args.n_steps,
                batch_size=args.batch_size, ent_coef=args.ent_coef,
                seed=args.seed, verbose=1, device="cpu")
    model.learn(total_timesteps=args.steps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))

    meta = {
        "observation_dim": int(model.observation_space.shape[0]),
        "n_actions": int(model.action_space.n),
        "steps": args.steps,
        "cost": args.cost,
        "learning_rate": args.lr,
        "n_steps": args.n_steps,
        "ent_coef": args.ent_coef,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "users": args.users,
        "days": args.days,
        "note": "treinado sobre o vetor observável; não usa motivação, "
                "habilidade nem sono exatos",
    }
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    env.close()
    print(f"checkpoint salvo em {out}.zip")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
