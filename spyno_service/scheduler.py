"""Agendador: pontos de decisão, fechamento de janela e retreino diário.

Nenhuma dessas tarefas precisa de um broker. Na escala do Spyno, algumas
milhares de decisões por dia, o estado vive na própria tabela de decisões e um
agendador em processo dá conta. Trocar isso por fila distribuída acrescentaria
entrega exatamente-uma-vez ao que precisa ser demonstrado, sem melhorar nada.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

log = logging.getLogger("spyno.scheduler")


def build_scheduler(ctx: dict, on_decision_point=None) -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone="America/Sao_Paulo")
    cfg, store, policy = ctx["config"], ctx["store"], ctx["policy"]

    @sched.scheduled_job(CronTrigger(minute=5))
    def _close_windows():
        n = store.close_expired_windows(datetime.now(timezone.utc))
        if n:
            log.info("janelas fechadas como censura: %d", n)

    # Atenção: em produção o fechamento do dia é do backend, via POST
    # /v1/end-day, que é idempotente por data. Este job chama a política direto
    # e não passa por essa proteção; não ligue os dois ao mesmo tempo.
    @sched.scheduled_job(CronTrigger(hour=3, minute=0))
    def _end_day():
        try:
            policy.end_day()
            log.info("lote do dia fechado")
        except Exception:
            log.exception("end_day falhou")

    if on_decision_point is not None:
        # Um ponto de decisão por hora dentro da janela permitida. A
        # elegibilidade fina continua do lado do backend.
        @sched.scheduled_job(CronTrigger(hour=f"{cfg.quiet_end_h}-{cfg.quiet_start_h - 1}",
                                         minute=0))
        def _tick():
            try:
                on_decision_point(datetime.now(timezone.utc))
            except Exception:
                log.exception("ponto de decisão falhou")

    return sched
