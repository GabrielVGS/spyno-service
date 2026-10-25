"""Serviço de decisão. Dois verbos, um contrato, política plugável."""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request

from .config import CONFIG
from .contracts import (ARMS, ContractInfo, Context, DecideRequest, DecideResponse,
                        EndDayRequest, EndDayResponse, FeedbackRequest,
                        FeedbackResponse, Health)
from .features import FEATURES, POLICY_SCHEMA
from .guardrails import apply as apply_guardrails
from .policies import propensities
from .policies.clipping import ClippedPolicy
from .policies.bandit import IncompatibleSnapshotError
from .policies.registry import available, build, version
from .posterior import apply_warmstart, digest, has_posterior, snapshot_of
from .store import Store, make_decision_id

#: Versão do contrato HTTP. O backend confere em GET /v1/contract.
#: 2.2: campos aditivos sobre a origem da posterior (posterior_init,
#: warmstart_sha256 no contrato; posterior_source no /health).
#: 2.3: campos aditivos sobre o congelamento da posterior em shadow
#: (learning_frozen no contrato, no /health, no /v1/policy-state, no decide e no
#: end-day; shadow_learn no contrato).
CONTRACT_VERSION = "2.3"

_PROPOSED = ("notifs_today", "fatigue", "minutes_since_last")

log = logging.getLogger("spyno.service")


def create_app(policy=None, store=None, config=None) -> FastAPI:
    """Fábrica da aplicação.

    As dependências entram por parâmetro para que o serviço seja exercitável
    sem Postgres e sem a política real. É isso que torna os testes de contrato
    executáveis em qualquer ambiente, e é a razão de não haver estado global.
    """
    cfg = config or CONFIG
    chosen = policy if policy is not None else build(cfg.policy)
    store = store if store is not None else Store(cfg.database_url)

    # 1) Estado inicial da linhagem: warm-start, se configurado; senão o prior.
    #    Artefato incompatível derruba a subida: quem pediu o warm-start não
    #    pode receber um prior sem saber.
    init, warm = "prior", None
    if cfg.warmstart_snapshot:
        warm = apply_warmstart(chosen, cfg.policy, cfg.warmstart_snapshot)
        init = "warmstart"
        log.info("warm-start carregado de %s (sha256 %s, posterior %s)",
                 warm["path"], warm["sha256"][:12], warm["posterior_digest"])

    aquecer = getattr(chosen, "warmup", None)
    if callable(aquecer):
        try:
            aquecer()
        except Exception:
            log.exception("aquecimento da política falhou; ela cairá na contingência")

    # 2) A identidade é a do estado inicial (ver posterior.py): o warm-start
    #    entra no hash, a posterior persistida não.
    policy_version = version(chosen)

    # 3) Posterior persistida da MESMA versão tem precedência.
    persistable = has_posterior(chosen)
    source = init if persistable else None
    persisted_day, skipped = None, []
    if persistable:
        row = store.load_posterior(policy_version)
        if row is not None:
            try:
                chosen.load_snapshot(row["snapshot"])
            except IncompatibleSnapshotError as exc:
                raise RuntimeError(
                    f"posterior persistida de {policy_version} (dia {row['day']}) não "
                    f"carrega: {exc}") from exc
            if digest(snapshot_of(chosen)) != row["digest"]:
                raise RuntimeError(
                    f"posterior persistida de {policy_version} não confere com o digest "
                    f"gravado ({row['digest']})")
            source, persisted_day = "persisted", row["day"].isoformat()
        skipped = [v for v in store.persisted_versions()
                   if v["policy_version"] != policy_version]
        if skipped:
            log.warning("posteriors persistidas de outras versões NÃO carregadas "
                        "(servindo %s): %s", policy_version,
                        ", ".join(f"{v['policy_version']}@{v['day']}" for v in skipped))

    app = FastAPI(
        title="Spyno — serviço de decisão",
        description="Interface comum para políticas adaptativas de notificação.",
        version="0.1.0",
    )
    app.state.ctx = {
        "config": cfg,
        "policy": chosen,
        "policy_version": policy_version,
        "store": store,
        "rng": np.random.default_rng(7),
        "pool": ThreadPoolExecutor(max_workers=4),
        "seen_users": set(),
        "pending_updates": 0,
        "end_day_lock": threading.Lock(),
        "posterior": {"source": source, "init": init if persistable else None,
                      "warmstart": warm, "persisted_day": persisted_day,
                      "skipped_persisted": skipped},
    }
    log.info("política ativa: %s (%s) | shadow=%s | aprendizado %s | posterior=%s%s",
             cfg.policy, policy_version, cfg.shadow,
             "CONGELADO" if cfg.learning_frozen else "ativo", source,
             f" (dia {persisted_day})" if persisted_day else "")

    def require_auth(request: Request) -> None:
        """Bearer token em tempo constante; sem token configurado, 503 (fail closed)."""
        c = request.app.state.ctx["config"]
        if c.auth_disabled:
            return
        if not c.decision_token:
            raise HTTPException(503, "autenticação não configurada: defina "
                                     "SPYNO_DECISION_TOKEN (ou SPYNO_DECISION_AUTH=disabled "
                                     "em desenvolvimento)")
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        ok = scheme.lower() == "bearer" and hmac.compare_digest(
            token.strip().encode(), c.decision_token.encode())
        if not ok:
            raise HTTPException(401, "token inválido",
                                headers={"WWW-Authenticate": "Bearer"})

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_auth)])

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
            posterior_source=ctx["posterior"]["source"],
            learning_frozen=ctx["config"].learning_frozen,
        )

    @app.get("/v1/contract", response_model=ContractInfo)
    def contract(request: Request) -> ContractInfo:
        """Sem autenticação: o backend lê na subida e recusa iniciar se divergir."""
        ctx = request.app.state.ctx
        c = ctx["config"]
        pol = ctx["policy"]
        w_max = (pol.clip.max_importance_weight(len(pol.send_arms))
                 if isinstance(pol, ClippedPolicy) else None)
        return ContractInfo(
            contract_version=CONTRACT_VERSION,
            policy=c.policy,
            policy_version=ctx["policy_version"],
            policy_schema=POLICY_SCHEMA,
            outcome_window_min=c.outcome_window_min,
            shadow=c.shadow,
            features=list(FEATURES),
            arms=list(ARMS),
            clip_min=c.clip_min, clip_max=c.clip_max, arm_mix=c.arm_mix,
            arm_floor=c.arm_floor, max_importance_weight=w_max,
            daily_cap=c.daily_cap, cooldown_min=c.cooldown_min,
            quiet_start_h=c.quiet_start_h, quiet_end_h=c.quiet_end_h,
            posterior_init=ctx["posterior"]["init"],
            warmstart_sha256=(ctx["posterior"]["warmstart"] or {}).get("sha256"),
            learning_frozen=c.learning_frozen, shadow_learn=c.shadow_learn,
        )

    @v1.get("/policies")
    def policies(request: Request) -> dict:
        return {"active": request.app.state.ctx["config"].policy,
                "available": available()}

    @v1.get("/features")
    def features() -> dict:
        """Ordem do vetor observável, para o consumidor conferir o contrato."""
        return {"features": list(FEATURES), "policy_schema": POLICY_SCHEMA}

    @v1.get("/policy-state")
    def policy_state(request: Request) -> dict:
        """Impressão digital da posterior, para auditoria e verificação ponta a ponta.

        Não expõe os pesos: só um hash do snapshot (m, q, custo), os updates
        acumulados desde o último end-day e os dias já fechados. É o que permite
        confirmar de fora que um /v1/end-day com updates mudou a posterior e que
        um end-day repetido não a muda.
        """
        ctx = request.app.state.ctx
        posterior_digest = None
        try:
            snap = snapshot_of(ctx["policy"])
            posterior_digest = None if snap is None else digest(snap)
        except Exception:
            log.exception("snapshot da política falhou")
        post = ctx["posterior"]
        return {"policy_version": ctx["policy_version"], "posterior_digest": posterior_digest,
                "pending_updates": ctx["pending_updates"],
                "days_closed": ctx["store"].closed_days(),
                # origem na subida: persisted | warmstart | prior (None sem posterior)
                "posterior_source": post["source"],
                "posterior_init": post["init"],
                "persisted_day": post["persisted_day"],
                "warmstart": post["warmstart"],
                "skipped_persisted": post["skipped_persisted"],
                # em shadow (sem SPYNO_SHADOW_LEARN) o digest não muda
                "learning_frozen": ctx["config"].learning_frozen}

    @v1.post("/decide", response_model=DecideResponse)
    def decide(req: DecideRequest, request: Request) -> DecideResponse:
        ctx = request.app.state.ctx
        cfg, store, pol = ctx["config"], ctx["store"], ctx["policy"]
        policy_version = ctx["policy_version"]

        decision_id = make_decision_id(req.user_key, req.decision_point, policy_version)

        # Idempotência: o mesmo ponto de decisão (mesma hora) devolve a decisão
        # já tomada, sem nova amostragem. Sem isso um retry ou um tick duplicado
        # viraria exploração extra e corromperia a propensão registrada.
        existing = store.get(decision_id)
        if existing is not None:
            return _to_response(existing)

        raw = req.context.model_dump()
        obs, basis = _effective_context(req.context, cfg.shadow)
        eff_ctx = Context(**obs)
        started = time.perf_counter()
        fallback, fallback_reason = False, None

        if req.user_key not in ctx["seen_users"]:
            try:
                pol.start_user(req.user_key)
            except Exception:
                log.exception("start_user falhou para %s", req.user_key)
            ctx["seen_users"].add(req.user_key)

        pi = None
        try:
            if isinstance(pol, ClippedPolicy):
                future = ctx["pool"].submit(pol.decide_with_distribution, req.user_key,
                                            obs, req.eligible, ctx["rng"])
                recommended, pi = future.result(timeout=cfg.decide_timeout_s)
            else:
                future = ctx["pool"].submit(pol.decide, req.user_key, obs,
                                            req.eligible, ctx["rng"])
                recommended, _ = future.result(timeout=cfg.decide_timeout_s)
        except FutureTimeout:
            recommended, fallback, fallback_reason, pi = 0, True, "timeout", None
        except Exception as exc:
            log.exception("política falhou")
            recommended, fallback, fallback_reason, pi = 0, True, type(exc).__name__, None

        executed, guardrail_reason = apply_guardrails(recommended, eff_ctx, cfg)
        # Ação proposta: a que TERIA sido executada sem o shadow, isto é, a
        # recomendada depois dos guardrails reais. É sobre ela que o backend
        # calcula os contadores *_proposed; fora do shadow é a executada.
        proposed = executed
        # O guardrail depende só do contexto. Se ele bloquearia qualquer envio,
        # a abstenção executada tem probabilidade 1, e o ponto não é elegível
        # para treino (no simulador, cota estourada e sono tornam o ponto
        # inelegível).
        _, would_block = apply_guardrails(1, eff_ctx, cfg)

        # Em modo de observação a política decide e o registro guarda, mas nada
        # é enviado.
        if cfg.shadow:
            executed = 0
            guardrail_reason = guardrail_reason or "shadow"
        forced_zero = cfg.shadow or would_block is not None

        if fallback:
            propensity, rec_prop, method = 1.0, 1.0, "fallback"
        elif pi is not None:
            method = "exact"
            rec_prop = float(pi[recommended])
            propensity = 1.0 if forced_zero else float(pi[executed])
        else:
            rec_prop, _ = propensities(pol, obs, recommended, cfg.propensity_samples)
            if forced_zero:
                propensity, method = 1.0, "exact"
            else:
                propensity, method = propensities(pol, obs, executed, cfg.propensity_samples)

        # `update_eligible` diz se o ponto SERIA treinável (elegível, sem guardrail
        # bloqueando, sem contingência), com ou sem congelamento: é a seleção que
        # a OPE precisa reproduzir. Se ele treina de fato depende também de
        # `learning_frozen` (ver /v1/feedback).
        update_eligible = bool(req.eligible and not fallback and would_block is None)

        record = {
            "decision_id": decision_id,
            "user_key": req.user_key,
            "decision_point": req.decision_point,
            "context": raw,
            "policy_context": obs,
            "counter_basis": basis,
            "eligible": req.eligible,
            "update_eligible": update_eligible,
            "learning_frozen": cfg.learning_frozen,
            "recommended_action": recommended,
            "executed_action": executed,
            "proposed_action": proposed,
            "guardrail_reason": guardrail_reason,
            "propensity": propensity,
            "recommended_propensity": rec_prop,
            "arm_probs": None if pi is None else [float(p) for p in pi],
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

    @v1.post("/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest, request: Request) -> FeedbackResponse:
        ctx = request.app.state.ctx
        store, pol = ctx["store"], ctx["policy"]
        applied, reason, outcome = store.apply_outcome(
            req.decision_id, req.outcome, req.value, req.observed_at
        )
        if not applied:
            return FeedbackResponse(decision_id=req.decision_id, applied=False, reason=reason)

        # y é "treinou dentro da janela", para qualquer braço (inclusive o 0).
        # Censura e falha de entrega só fecham o pendente. Uma resposta fora da
        # janela já chega aqui como censura.
        y = {"responded": 1, "not_responded": 0}.get(outcome)
        record = store.get(req.decision_id)
        trained, skip = False, None
        if y is None:
            skip = f"sem_treino_{outcome}"
        elif record["policy_version"] != ctx["policy_version"]:
            # Decisão de outra versão: outra escala de features e outro
            # clipping. O desfecho fica no log, mas não treina a posterior atual.
            skip = "policy_version_diferente"
        elif not record.get("update_eligible"):
            skip = "ponto_nao_elegivel"
        elif _decided_frozen(record):
            # Decidido com a posterior congelada (shadow): o desfecho fica no log,
            # para a OPE, mas nunca treina, nem depois de sair do shadow.
            skip = "decisao_em_shadow_congelado"
        elif ctx["config"].learning_frozen:
            # Decisão treinável, mas o serviço agora está congelado: nada entra
            # no buffer enquanto durar o shadow.
            skip = "aprendizado_congelado"
        else:
            obs = json.loads(record.get("policy_context") or record["context"])
            try:
                # Sempre a ação EXECUTADA. Em shadow com SPYNO_SHADOW_LEARN=1 ela
                # é 0: (x, 0, y) é dado do braço 0. A ação proposta nunca treina,
                # porque o desfecho não dependeu dela.
                pol.update(record["user_key"], obs, record["executed_action"], y)
                trained = True
                ctx["pending_updates"] += 1
            except Exception:
                log.exception("update da política falhou para %s", req.decision_id)
                skip = "erro_no_update"
        store.mark_trained(req.decision_id, trained, skip)
        return FeedbackResponse(decision_id=req.decision_id, applied=True,
                                trained=trained, train_skip_reason=skip)

    @v1.post("/close-windows")
    def close_windows(request: Request) -> dict:
        """Fecha as janelas de atribuição vencidas como censura (sem treino)."""
        closed = request.app.state.ctx["store"].close_expired_windows(
            datetime.now(timezone.utc))
        return {"censored": closed}

    @v1.post("/end-day", response_model=EndDayResponse)
    def end_day(req: EndDayRequest, request: Request) -> EndDayResponse:
        """Fecha o lote do dia local `date`: a posterior de hoje é o prior de amanhã.

        O backend chama uma vez por dia. Idempotente por data: uma segunda
        chamada com a mesma data não refaz o ajuste. O buffer de updates do dia
        é consumido pelo ajuste e esvaziado.

        Com a posterior congelada (shadow), o dia é fechado em `policy_days`
        com 0 updates, `policy.end_day()` não é chamado e nada é gravado em
        `policy_posteriors`: a posterior persistida só nasce de aprendizado, e
        uma linha gravada em shadow seria só uma cópia do estado inicial.
        """
        ctx = request.app.state.ctx
        store = ctx["store"]
        frozen = ctx["config"].learning_frozen
        with ctx["end_day_lock"]:
            if store.day_closed(req.date):
                return EndDayResponse(ok=True, date=req.date, already_closed=True,
                                      learning_frozen=frozen)
            if frozen:
                try:
                    closed = store.close_day(req.date, ctx["policy_version"], 0)
                except Exception as exc:
                    log.exception("fechamento do dia %s não foi gravado", req.date)
                    return EndDayResponse(ok=False, date=req.date, learning_frozen=True,
                                          error=f"persistencia_{type(exc).__name__}")
                return EndDayResponse(ok=True, date=req.date, already_closed=not closed,
                                      learning_frozen=True)
            n = ctx["pending_updates"]
            try:
                ctx["policy"].end_day()
            except Exception as exc:
                log.exception("end_day falhou")
                return EndDayResponse(ok=False, date=req.date, error=type(exc).__name__)
            snap = snapshot_of(ctx["policy"]) if has_posterior(ctx["policy"]) else None
            # Fechamento e posterior na mesma transação (Store.close_day). Se ela
            # falhar, o dia fica aberto e nada é gravado, mas a posterior em
            # memória já avançou e o buffer foi consumido. `pending_updates` NÃO
            # é zerado: ele conta os updates desde o último fechamento gravado.
            # Na nova tentativa, os feedbacks chegados no intervalo (que estão no
            # buffer) também entram no ajuste e no dia `date`: a posterior gravada
            # é a de antes da falha MAIS esses updates, e `updates_applied` soma os
            # dois lotes.
            try:
                closed = store.close_day(req.date, ctx["policy_version"], n, snapshot=snap,
                                         digest=None if snap is None else digest(snap))
            except Exception as exc:
                log.exception("fechamento do dia %s não foi gravado", req.date)
                return EndDayResponse(ok=False, date=req.date,
                                      error=f"persistencia_{type(exc).__name__}")
            if not closed:
                return EndDayResponse(ok=True, date=req.date, already_closed=True)
            if snap is not None:
                ctx["posterior"]["persisted_day"] = req.date.isoformat()
            ctx["pending_updates"] = 0
        return EndDayResponse(ok=True, date=req.date, updates_applied=n)

    app.include_router(v1)
    return app


def _effective_context(context: Context, shadow: bool) -> tuple[dict, str]:
    """Contexto que a política e os guardrails veem, e a base dos contadores.

    Em shadow, se o backend mandou `notifs_today_proposed` e `fatigue_proposed`,
    os contadores sobre a ação proposta substituem os reais (que ficam presos em
    zero, porque nada é enviado). Sem eles, usam-se os reais e o log diz isso.
    """
    raw = context.model_dump()
    obs = {k: v for k, v in raw.items() if not k.endswith("_proposed")}
    if shadow and context.notifs_today_proposed is not None \
            and context.fatigue_proposed is not None:
        for k in _PROPOSED:
            obs[k] = raw[f"{k}_proposed"]
        return obs, "proposed"
    if shadow:
        log.warning("shadow sem contadores *_proposed: features usam os contadores reais")
    return obs, "executed"


def _to_response(record: dict) -> DecideResponse:
    return DecideResponse(
        decision_id=record["decision_id"],
        recommended_action=record["recommended_action"],
        recommended_arm=ARMS[record["recommended_action"]],
        executed_action=record["executed_action"],
        proposed_action=_proposed_action(record),
        propensity=record["propensity"],
        recommended_propensity=record.get("recommended_propensity") or record["propensity"],
        arm_probabilities=_json_or_none(record.get("arm_probs")),
        propensity_method=record["propensity_method"],
        counter_basis=record.get("counter_basis") or "executed",
        update_eligible=bool(record.get("update_eligible")),
        learning_frozen=_decided_frozen(record),
        policy=record["policy"],
        policy_version=record["policy_version"],
        shadow=bool(record["shadow"]),
        fallback=bool(record["fallback"]),
        fallback_reason=record["fallback_reason"],
        guardrail_reason=record["guardrail_reason"],
        latency_ms=record["latency_ms"],
    )


def _decided_frozen(record: dict) -> bool:
    """A decisão foi tomada com a posterior congelada.

    Linhas anteriores à coluna (nula) contam como congeladas se eram de shadow:
    é a regra vigente, e o padrão desde que ela existe.
    """
    frozen = record.get("learning_frozen")
    return bool(record["shadow"]) if frozen is None else bool(frozen)


def _proposed_action(record: dict) -> int:
    """Ação proposta gravada; para linhas anteriores à coluna, reconstruída.

    A reconstrução é a mesma regra: em shadow, a recomendada só conta como
    proposta se nenhum guardrail a bloqueou (guardrail_reason == "shadow").
    """
    if record.get("proposed_action") is not None:
        return int(record["proposed_action"])
    if not record["shadow"]:
        return int(record["executed_action"])
    return int(record["recommended_action"]) if record["guardrail_reason"] == "shadow" else 0


def _json_or_none(value):
    if value is None or isinstance(value, list):
        return value
    return json.loads(value)
