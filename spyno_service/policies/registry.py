"""Registro de políticas: um nome, uma fábrica.

Trocar a política ativa é mudar `SPYNO_POLICY`. O contrato da API não muda, o
que é exatamente a propriedade que a questão de viabilidade exige demonstrar.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Callable

from ..config import CONFIG
from . import bandit
from .base import Policy

FACTORIES: dict[str, Callable[..., Policy]] = {
    "ts_hybrid": bandit.ts_hybrid,
    "ts_disjoint": bandit.ts_disjoint,
    "ts_binary": bandit.ts_binary,
    "never": bandit.never,
    "always": bandit.always,
    "fbm_threshold": bandit.fbm_threshold,
    "fbm_obs_threshold": bandit.fbm_threshold_observable,
}


def _ppo(**kwargs) -> Policy:
    from .ppo import PPOPolicy

    ckpt = kwargs.pop("checkpoint", Path(CONFIG.checkpoint_dir) / "ppo_observable.zip")
    return PPOPolicy(checkpoint=ckpt, **kwargs)


FACTORIES["ppo"] = _ppo


def available() -> list[str]:
    return sorted(FACTORIES)


def build(name: str | None = None, **kwargs) -> Policy:
    name = name or CONFIG.policy
    try:
        factory = FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"política '{name}' desconhecida. Disponíveis: {', '.join(available())}"
        ) from None
    return factory(**kwargs)


def version(policy: Policy) -> str:
    """Identidade reproduzível da política servida.

    Combina o nome, a assinatura da fábrica e o estado inicial dos parâmetros.
    Serve para amarrar cada decisão registrada ao modelo que a produziu: sem
    isso, não é possível saber meses depois qual política gerou cada envio.
    """
    parts = [type(policy).__name__, getattr(policy, "name", "?")]
    snap = getattr(policy, "snapshot", None)
    if callable(snap):
        try:
            parts.append(repr(sorted(snap().items())))
        except Exception:
            parts.append("snapshot-indisponivel")
    else:
        try:
            parts.append(str(inspect.signature(type(policy).__init__)))
        except (TypeError, ValueError):
            pass
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]
    return f"{getattr(policy, 'name', 'policy')}-{digest}"
