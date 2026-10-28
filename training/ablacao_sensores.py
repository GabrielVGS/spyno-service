"""Quanto custa não ter os sensores de localização e movimento?

Três condições sobre os mesmos mundos e as mesmas sementes de desfecho:

  completo   os 15 atributos, com os sensores como no simulador
  constante  os dois sensores fixados em 0,5, que é o paliativo implementado
             hoje no backend
  removido   os dois sensores zerados, equivalente a treinar com 13 atributos
             num modelo linear

Comparação pareada por semente: cada mundo é percorrido pelas três condições.

ATENÇÃO (2026-10-25): no esquema v2, `build_policy` devolve a política com o
adaptador `_CompatibilidadeHerdada`, que zera os sensores depois de `Substitui`.
As três condições rodam, portanto, SEM sensores, e o script não mede o que diz.
Além disso, é exploratório: cold start, 10 sementes, IC normal sem Holm, sem
JSON de saída. Não use o resultado como evidência; ver o README ("Warm-start").
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

_SIM = Path(os.getenv("SPYNO_SIM_PATH", Path(__file__).resolve().parents[2] / "spyno-sim"))
for p in (str(_SIM), str(Path(__file__).resolve().parents[1])):
    if p not in sys.path:
        sys.path.insert(0, p)

from spyno_sim.config import Demographics  # noqa: E402
from spyno_sim.response import ResponseParams  # noqa: E402
from spyno_sim.simulate import SimConfig, build_world, rollout  # noqa: E402

from spyno_service.policies.registry import build as build_policy  # noqa: E402

SENSORES = ("location_home", "motion_stationary")


class Substitui:
    def __init__(self, inner, valor):
        self.inner, self.valor = inner, valor
        self.name = inner.name

    def _mask(self, obs):
        return {**obs, **{k: self.valor for k in SENSORES}}

    def start_user(self, u):
        self.inner.start_user(u)

    def decide(self, u, obs, e, rng=None):
        return self.inner.decide(u, self._mask(obs), e, rng)

    def update(self, u, obs, arm, r):
        self.inner.update(u, self._mask(obs), arm, r)

    def end_day(self):
        self.inner.end_day()


def ic95(xs: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return (statistics.mean(xs), 0.0)
    m = statistics.mean(xs)
    return m, 1.96 * statistics.stdev(xs) / len(xs) ** 0.5


def main() -> None:
    sementes = [int(s) for s in os.getenv("SEMENTES", "1,2,3,4,5,6,7,8,9,10").split(",")]
    n_users, n_days = 20, 28
    politica = os.getenv("POLITICA", "ts_hybrid")

    resultados: dict[str, list[dict]] = {"completo": [], "constante": [], "removido": []}

    for s in sementes:
        cfg = SimConfig(n_users=n_users, n_days=n_days, seed=1000 + s, llm_mode="stub")
        world, _ = build_world(cfg, Demographics())
        params = ResponseParams()
        for rotulo, envolve in (("completo", None), ("constante", 0.5), ("removido", 0.0)):
            pol = build_policy(politica)
            if envolve is not None:
                pol = Substitui(pol, envolve)
            r = rollout(world, pol, params, cfg, outcome_seed=cfg.seed + 10_000,
                        collect_rows=False)
            m = r.metrics(n_users, n_days)
            resultados[rotulo].append({
                "treinos": m["trainings_per_user_week"],
                "notifs": r.notifications / n_users / (n_days / 7),
                "regret": m["cum_regret"],
            })

    print(f"\npolítica: {politica} | {len(sementes)} sementes pareadas | "
          f"{n_users} usuários × {n_days} dias\n")
    print("%-11s %18s %18s %18s" % ("condição", "treinos/usr·sem", "notifs/usr·sem", "regret acum."))
    for rotulo, linhas in resultados.items():
        celulas = []
        for chave in ("treinos", "notifs", "regret"):
            m, e = ic95([l[chave] for l in linhas])
            celulas.append(f"{m:9.3f} ± {e:.3f}")
        print("%-11s %18s %18s %18s" % (rotulo, *celulas))

    print("\ndiferenças pareadas contra 'completo' (IC 95%):")
    for rotulo in ("constante", "removido"):
        for chave in ("treinos", "notifs", "regret"):
            difs = [resultados[rotulo][i][chave] - resultados["completo"][i][chave]
                    for i in range(len(sementes))]
            m, e = ic95(difs)
            cruza = "  (cruza zero)" if abs(m) < e else "  *"
            print(f"  {rotulo:10s} {chave:8s} {m:+8.3f} ± {e:.3f}{cruza}")


if __name__ == "__main__":
    main()
