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
from datetime import datetime, timedelta, timezone

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, MetaData,
                        String, Table, Text, create_engine, func, select, update)

from .config import CONFIG

METADATA = MetaData()

decisions = Table(
    "decisions", METADATA,
    Column("decision_id", String(64), primary_key=True),
    Column("user_key", String(128), nullable=False, index=True),
    Column("decision_point", DateTime(timezone=True), nullable=False),
    Column("context", Text, nullable=False),
    Column("recommended_action", Integer, nullable=False),
    Column("executed_action", Integer, nullable=False),
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


def make_decision_id(user_key: str, decision_point: datetime, policy_version: str) -> str:
    """Identificador determinístico: o mesmo ponto devolve a mesma decisão.

    É o que torna a operação idempotente. Uma repetição de rede não pode virar
    exploração adicional, porque isso corromperia a propensão registrada.
    """
    raw = f"{user_key}|{decision_point.isoformat()}|{policy_version}"
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
