"""Restrições aplicadas fora da política.

Tudo o que o aprendizado não pode relaxar mora aqui: horário de silêncio, cota
diária e intervalo mínimo entre notificações. A política decide o que gostaria
de fazer; esta camada decide o que será executado, e o registro guarda as duas
coisas separadamente. Confundir as duas invalidaria qualquer análise posterior,
porque um envio bloqueado não é uma abstenção escolhida pela política.
"""

from __future__ import annotations

from .config import CONFIG
from .contracts import Context


def in_quiet_hours(hour: int, cfg=None) -> bool:
    cfg = cfg or CONFIG
    start, end = cfg.quiet_start_h, cfg.quiet_end_h
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def apply(recommended: int, ctx: Context, cfg=None) -> tuple[int, str | None]:
    """Devolve (ação executada, motivo do bloqueio)."""
    cfg = cfg or CONFIG
    if recommended == 0:
        return 0, None
    if in_quiet_hours(ctx.hour, cfg):
        return 0, "quiet_hours"
    if ctx.notifs_today >= cfg.daily_cap:
        return 0, "daily_cap"
    if ctx.minutes_since_last is not None and ctx.minutes_since_last < cfg.cooldown_min:
        return 0, "cooldown"
    return recommended, None
