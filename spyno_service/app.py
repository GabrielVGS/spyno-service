"""Serviço de decisão. Dois verbos, um contrato, política plugável."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, Request

from .config import CONFIG
from .contracts import (ARMS, DecideRequest, DecideResponse, FeedbackRequest,
                        FeedbackResponse, Health)
from .guardrails import apply as apply_guardrails
from .policies import propensities
from .policies.registry import available, build, version
from .store import Store, make_decision_id

log = logging.getLogger("spyno.service")


def create_app(policy=None, store=None, config=None) -> FastAPI:
    """Fábrica da aplicação.

    As dependências entram por parâmetro para que o serviço seja exercitável
    sem Postgres e sem a política real. É isso que torna os testes de contrato
    executáveis em qualquer ambiente, e é a razão de não haver estado global.
    """
    cfg = config or CONFIG
    chosen = policy if policy is not None else build(cfg.policy)

    aquecer = getattr(chosen, "warmup", None)
    if callable(aquecer):
        try:
            aquecer()
        except Exception:
            log.exception("aquecimento da política falhou; ela cairá na contingência")

    app = FastAPI(
        title="Spyno — serviço de decisão",
        description="Interface comum para políticas adaptativas de notificação.",
        version="0.1.0",
    )
    app.state.ctx = {
        "config": cfg,
        "policy": chosen,
        "policy_version": version(chosen),
        "store": store if store is not None else Store(cfg.database_url),
        "rng": np.random.default_rng(7),
        "pool": ThreadPoolExecutor(max_workers=4),
        "seen_users": set(),
    }
    log.info("política ativa: %s (%s) | shadow=%s",
             cfg.policy, app.state.ctx["policy_version"], cfg.shadow)

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        ctx = request.app.state.ctx
        return Health(
            status="ok",
            policy=ctx["config"].policy,
            policy_version=ctx["policy_version"],
            shadow=ctx["config"].shadow,
            decisions=ctx["store"].count(),
            pending_outcomes=ctx["store"].pending_outcomes(),
        )

    @app.get("/v1/policies")
    def policies(request: Request) -> dict:
        return {"active": request.app.state.ctx["config"].policy,
                "available": available()}

    @app.get("/v1/features")
    def features() -> dict:
        """Ordem do vetor observável, para o consumidor conferir o contrato."""
        from .features import FEATURES

        return {"features": list(FEATURES)}

    @app.post("/v1/decide", response_model=DecideResponse)
    def decide(req: DecideRequest, request: Request) -> DecideResponse:
        ctx = request.app.state.ctx
        cfg, store, pol = ctx["config"], ctx["store"], ctx["policy"]
        policy_version = ctx["policy_version"]

        decision_id = make_decision_id(req.user_key, req.decision_point, policy_version)

        # Idempotência: o mesmo ponto de decisão devolve a decisão já tomada,
        # sem nova amostragem. Sem isso um retry viraria exploração extra e
        # corromperia a propensão registrada.
        existing = store.get(decision_id)
        if existing is not None:
            return _to_response(existing)

        obs = req.context.model_dump()
        started = time.perf_counter()
        fallback, fallback_reason = False, None

        if req.user_key not in ctx["seen_users"]:
            try:
                pol.start_user(req.user_key)
            except Exception:
                log.exception("start_user falhou para %s", req.user_key)
            ctx["seen_users"].add(req.user_key)

        try:
            future = ctx["pool"].submit(pol.decide, req.user_key, obs,
                                        req.eligible, ctx["rng"])
            recommended, _ = future.result(timeout=cfg.decide_timeout_s)
        except FutureTimeout:
            recommended, fallback, fallback_reason = 0, True, "timeout"
        except Exception as exc:
            log.exception("política falhou")
            recommended, fallback, fallback_reason = 0, True, type(exc).__name__

        executed, guardrail_reason = apply_guardrails(recommended, req.context, cfg)

        # Em modo de observação a política decide e o registro guarda, mas nada
        # é enviado. Só a propensão da ação efetivamente executada serve como
        # propensão de coleta.
        if cfg.shadow:
            executed = 0
            guardrail_reason = guardrail_reason or "shadow"

        if fallback:
            propensity, method = 1.0, "fallback"
        else:
            propensity, method = propensities(pol, obs, executed, cfg.propensity_samples)

        record = {
            "decision_id": decision_id,
            "user_key": req.user_key,
            "decision_point": req.decision_point,
            "context": obs,
            "recommended_action": recommended,
            "executed_action": executed,
            "guardrail_reason": guardrail_reason,
            "propensity": propensity,
            "propensity_method": method,
            "policy": cfg.policy,
            "policy_version": policy_version,
            "shadow": cfg.shadow,
            "fallback": fallback,
            "fallback_reason": fallback_reason,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        store.insert(record, cfg.outcome_window_min)
        return _to_response(record)

    @app.post("/v1/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest, request: Request) -> FeedbackResponse:
        ctx = request.app.state.ctx
        store, pol = ctx["store"], ctx["policy"]
        applied, reason, outcome = store.apply_outcome(
            req.decision_id, req.outcome, req.value, req.observed_at
        )
        # A política só aprende com resposta dentro da janela. Um desfecho
        # censurado é registrado, mas não treina.
        if applied and outcome == "responded":
            record = store.get(req.decision_id)
            if record is not None:
                try:
                    pol.update(record["user_key"], json.loads(record["context"]),
                               record["executed_action"], req.value)
                except Exception:
                    log.exception("update da política falhou para %s", req.decision_id)
        return FeedbackResponse(decision_id=req.decision_id, applied=applied, reason=reason)

    @app.post("/v1/close-windows")
    def close_windows(request: Request) -> dict:
        """Fecha as janelas de atribuição vencidas. Chamado pelo agendador."""
        closed = request.app.state.ctx["store"].close_expired_windows(
            datetime.now(timezone.utc))
        return {"censored": closed}

    @app.post("/v1/end-day")
    def end_day(request: Request) -> dict:
        """Fecha o lote do dia: a posterior de hoje é o prior de amanhã."""
        try:
            request.app.state.ctx["policy"].end_day()
        except Exception as exc:
            log.exception("end_day falhou")
            return {"ok": False, "error": type(exc).__name__}
        return {"ok": True}

    return app


def _to_response(record: dict) -> DecideResponse:
    return DecideResponse(
        decision_id=record["decision_id"],
        recommended_action=record["recommended_action"],
        recommended_arm=ARMS[record["recommended_action"]],
        executed_action=record["executed_action"],
        propensity=record["propensity"],
        propensity_method=record["propensity_method"],
        policy=record["policy"],
        policy_version=record["policy_version"],
        shadow=bool(record["shadow"]),
        fallback=bool(record["fallback"]),
        fallback_reason=record["fallback_reason"],
        guardrail_reason=record["guardrail_reason"],
        latency_ms=record["latency_ms"],
    )
