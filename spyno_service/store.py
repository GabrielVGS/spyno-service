"""Registro de decisões: a fonte de verdade para reprodução e avaliação.

O log é somente-anexação no que importa: a decisão nunca é reescrita. O desfecho
entra depois, num par de colunas próprias, e uma segunda chamada com o mesmo
`decision_id` não altera nada. Sem essa garantia, um reenvio de rede
contabilizaria a mesma resposta duas vezes.

A mesma tabela serve de fila para o fechamento da janela de atribuição. Nessa
escala, algumas milhares de decisões por dia, uma tabela com estado é a escolha
correta e dispensa um broker.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import (Boolean, Column, Date, DateTime, Float, Integer, MetaData,
                        String, Table, Text, create_engine, func, inspect, select,
                        text, update)

from .config import CONFIG

log = logging.getLogger("spyno.store")

METADATA = MetaData()

decisions = Table(
    "decisions", METADATA,
    Column("decision_id", String(64), primary_key=True),
    Column("user_key", String(128), nullable=False, index=True),
    Column("decision_point", DateTime(timezone=True), nullable=False),
    Column("context", Text, nullable=False),
    Column("recommended_action", Integer, nullable=False),
    Column("executed_action", Integer, nullable=False),
    # a que teria sido executada sem o shadow (recomendada após os guardrails);
    # nula nas linhas gravadas antes da coluna (ver app._proposed_action)
    Column("proposed_action", Integer),
    Column("guardrail_reason", String(32)),
    Column("propensity", Float, nullable=False),
    Column("propensity_method", String(16), nullable=False),
    Column("policy", String(64), nullable=False),
    Column("policy_version", String(128), nullable=False),
    Column("shadow", Boolean, nullable=False),
    Column("fallback", Boolean, nullable=False),
    Column("fallback_reason", String(64)),
    Column("latency_ms", Float, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    # desfecho, preenchido depois e uma única vez
    Column("outcome", String(24)),
    Column("outcome_value", Integer),
    Column("outcome_at", DateTime(timezone=True)),
    Column("window_closes_at", DateTime(timezone=True), nullable=False),
    # ---- esquema v2 (nulas nas linhas gravadas antes dele) -------------------
    # elegibilidade informada pelo backend
    Column("eligible", Boolean),
    # o desfecho deste ponto pode treinar: elegível, sem guardrail bloqueando e
    # sem contingência (ver app.py)
    Column("update_eligible", Boolean),
    # o que a política efetivamente viu (contexto com os contadores escolhidos);
    # é com ele que o update é feito
    Column("policy_context", Text),
    Column("counter_basis", String(16)),
    # propensão da ação recomendada e distribuição clipada sobre os braços
    Column("recommended_propensity", Float),
    Column("arm_probs", Text),
    # auditoria do treino
    Column("trained", Boolean),
    Column("train_skip_reason", String(48)),
)

#: Colunas acrescentadas no esquema v2. Bancos criados antes recebem um
#: ALTER TABLE explícito na abertura (ver Store._migrar).
V2_COLUMNS = ("eligible", "update_eligible", "policy_context", "counter_basis",
              "recommended_propensity", "arm_probs", "trained", "train_skip_reason",
              "proposed_action")

#: Dias já fechados por /v1/end-day: é o que torna o fechamento idempotente.
policy_days = Table(
    "policy_days", METADATA,
    Column("day", Date, primary_key=True),
    Column("policy_version", String(128), nullable=False),
    Column("updates_applied", Integer, nullable=False),
    Column("closed_at", DateTime(timezone=True), nullable=False),
)


def _utc(dt: datetime) -> datetime:
    """Normaliza para UTC ciente de fuso.

    Tudo é gravado em UTC. Sem isso, o SQLite devolve datetimes sem fuso e a
    comparação com a janela de atribuição passa a depender do fuso de quem
    escreveu: uma resposta válida de São Paulo era marcada como censurada
    porque 15h20 com fuso -03 vira 18h20 UTC e era comparada contra um limite
    de 18h00 lido como se já fosse UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def decision_hour(decision_point: datetime) -> datetime:
    """Instante truncado na hora, em UTC: a unidade do ponto de decisão."""
    return _utc(decision_point).replace(minute=0, second=0, microsecond=0)


def make_decision_id(user_key: str, decision_point: datetime, policy_version: str) -> str:
    """Identificador determinístico: o mesmo ponto devolve a mesma decisão.

    É o que torna a operação idempotente. Uma repetição de rede não pode virar
    exploração adicional, porque isso corromperia a propensão registrada. O
    instante é truncado na hora (UTC): dois ticks do mesmo usuário na mesma hora
    são o mesmo ponto de decisão, mesmo que difiram em minutos ou milissegundos.
    """
    raw = f"{user_key}|{decision_hour(decision_point).isoformat()}|{policy_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class Store:
    def __init__(self, url: str | None = None):
        url = url or CONFIG.database_url
        kwargs: dict = {"future": True}
        if ":memory:" in url:
            # SQLite em memória cria um banco por conexão. Sem pool estático, a
            # tabela criada aqui desaparece na requisição seguinte.
            from sqlalchemy.pool import StaticPool

            kwargs.update(poolclass=StaticPool,
                          connect_args={"check_same_thread": False})
        self.engine = create_engine(url, **kwargs)
        METADATA.create_all(self.engine)
        self.migrated_columns = self._migrar()

    def _migrar(self) -> list[str]:
        """Acrescenta as colunas do esquema v2 a uma tabela criada antes dele.

        `create_all` não altera tabela existente. Sem isto, o primeiro insert num
        banco antigo falharia. As linhas antigas ficam com as colunas novas nulas
        e, por terem outra `policy_version`, nunca treinam a política atual.
        """
        existentes = {c["name"] for c in inspect(self.engine).get_columns("decisions")}
        faltando = [c for c in V2_COLUMNS if c not in existentes]
        if faltando:
            with self.engine.begin() as conn:
                for nome in faltando:
                    col = decisions.c[nome]
                    tipo = col.type.compile(dialect=self.engine.dialect)
                    conn.execute(text(f"ALTER TABLE decisions ADD COLUMN {nome} {tipo}"))
            log.warning("tabela decisions migrada para o esquema v2: %s", ", ".join(faltando))
        return faltando

    # ---- decisões --------------------------------------------------------

    def get(self, decision_id: str) -> dict | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(decisions).where(decisions.c.decision_id == decision_id)
            ).mappings().first()
        return dict(row) if row else None

    def insert(self, record: dict, outcome_window_min: int | None = None) -> None:
        payload = dict(record)
        payload["context"] = json.dumps(payload["context"], default=str)
        for k in ("policy_context", "arm_probs"):
            if payload.get(k) is not None and not isinstance(payload[k], str):
                payload[k] = json.dumps(payload[k], default=str)
        payload.setdefault("created_at", datetime.now(timezone.utc))
        window = outcome_window_min or CONFIG.outcome_window_min
        payload["decision_point"] = _utc(payload["decision_point"])
        payload["created_at"] = _utc(payload["created_at"])
        payload.setdefault(
            "window_closes_at",
            payload["decision_point"] + timedelta(minutes=window),
        )
        payload["window_closes_at"] = _utc(payload["window_closes_at"])
        with self.engine.begin() as conn:
            conn.execute(decisions.insert().values(**payload))

    # ---- desfechos -------------------------------------------------------

    def apply_outcome(self, decision_id: str, outcome: str, value: int,
                      observed_at: datetime) -> tuple[bool, str | None, str | None]:
        """Anexa o desfecho uma única vez.

        Devolve (aplicado, motivo da recusa, desfecho efetivamente gravado). O
        terceiro valor importa porque uma resposta chegada fora da janela é
        gravada como censura, e quem chamou precisa saber disso para não treinar
        a política com ela.
        """
        with self.engine.begin() as conn:
            row = conn.execute(
                select(decisions.c.outcome, decisions.c.window_closes_at)
                .where(decisions.c.decision_id == decision_id)
            ).first()
            if row is None:
                return False, "decisao_inexistente", None
            if row.outcome is not None:
                return False, "desfecho_ja_aplicado", None
            observed_utc = _utc(observed_at)
            if row.window_closes_at is not None and observed_utc > _utc(row.window_closes_at):
                outcome, value = "censored", 0
            conn.execute(
                update(decisions)
                .where(decisions.c.decision_id == decision_id)
                .values(outcome=outcome, outcome_value=value, outcome_at=observed_utc)
            )
        return True, None, outcome

    def mark_trained(self, decision_id: str, trained: bool, reason: str | None) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(decisions).where(decisions.c.decision_id == decision_id)
                .values(trained=trained, train_skip_reason=reason)
            )

    # ---- fechamento do dia -----------------------------------------------

    def day_closed(self, day) -> bool:
        with self.engine.begin() as conn:
            return conn.execute(
                select(policy_days.c.day).where(policy_days.c.day == day)
            ).first() is not None

    def close_day(self, day, policy_version: str, updates_applied: int) -> bool:
        """Registra o fechamento. Devolve False se o dia já estava fechado."""
        from sqlalchemy.exc import IntegrityError

        try:
            with self.engine.begin() as conn:
                conn.execute(policy_days.insert().values(
                    day=day, policy_version=policy_version,
                    updates_applied=updates_applied,
                    closed_at=datetime.now(timezone.utc)))
        except IntegrityError:
            return False
        return True

    def close_expired_windows(self, now: datetime | None = None) -> int:
        """Marca como censuradas as decisões cuja janela passou sem resposta.

        É o job periódico que substitui a fila: o estado vive na própria tabela.
        """
        now = _utc(now or datetime.now(timezone.utc))
        with self.engine.begin() as conn:
            result = conn.execute(
                update(decisions)
                .where(decisions.c.outcome.is_(None))
                .where(decisions.c.window_closes_at < now)
                .values(outcome="censored", outcome_value=0, outcome_at=now)
            )
        return result.rowcount or 0

    def pending_outcomes(self) -> int:
        with self.engine.begin() as conn:
            return int(conn.execute(
                select(func.count()).select_from(decisions)
                .where(decisions.c.outcome.is_(None))
            ).scalar_one())

    def count(self) -> int:
        with self.engine.begin() as conn:
            return int(conn.execute(
                select(func.count()).select_from(decisions)
            ).scalar_one())

    def labelled(self, limit: int = 10_000) -> list[dict]:
        """Decisões com desfecho, para o retreino em lote e para a avaliação."""
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(decisions).where(decisions.c.outcome.isnot(None))
                .order_by(decisions.c.created_at).limit(limit)
            ).mappings().all()
        out = []
        for r in rows:
            d = dict(r)
            d["context"] = json.loads(d["context"])
            out.append(d)
        return out
