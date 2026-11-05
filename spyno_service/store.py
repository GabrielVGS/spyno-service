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

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    inspect,
    select,
    text,
    update,
)

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
    # o serviço estava com a posterior congelada (shadow sem SPYNO_SHADOW_LEARN)
    # quando decidiu: o desfecho deste ponto nunca treina, mas fica no log para
    # a OPE; `update_eligible` continua dizendo se o ponto seria treinável
    Column("learning_frozen", Boolean),
    # desde o contrato 2.3: instante em que uma decisão gravada fora do shadow
    # foi repetida com o serviço em shadow e a resposta forçou a abstenção. A
    # decisão original fica como estava; a execução efetiva desta linha passa a
    # ser 0 (propensão 1) e o desfecho dela nunca treina
    Column("shadow_override_at", DateTime(timezone=True)),
)

#: Colunas acrescentadas no esquema v2. Bancos criados antes recebem um
#: ALTER TABLE explícito na abertura (ver Store._migrar).
V2_COLUMNS = ("eligible", "update_eligible", "policy_context", "counter_basis",
              "recommended_propensity", "arm_probs", "trained", "train_skip_reason",
              "proposed_action", "learning_frozen", "shadow_override_at")

#: Dias já fechados por /v1/end-day: é o que torna o fechamento idempotente.
policy_days = Table(
    "policy_days", METADATA,
    Column("day", Date, primary_key=True),
    Column("policy_version", String(128), nullable=False),
    Column("updates_applied", Integer, nullable=False),
    Column("closed_at", DateTime(timezone=True), nullable=False),
)

#: Última posterior de cada `policy_version`, gravada na mesma transação que o
#: fechamento do dia (ver Store.close_day). Uma linha por versão: a subida só
#: carrega a da versão servida (ver posterior.py).
policy_posteriors = Table(
    "policy_posteriors", METADATA,
    Column("policy_version", String(128), primary_key=True),
    # JSON do snapshot (m, q, custo, esquema, features)
    Column("snapshot", Text, nullable=False),
    Column("digest", String(16), nullable=False),
    Column("schema", String(16), nullable=False),
    # dia cujo fechamento produziu esta posterior
    Column("day", Date, nullable=False),
    Column("saved_at", DateTime(timezone=True), nullable=False),
    # desde o contrato 2.3: a posterior foi aprendida com o serviço em shadow
    # (SPYNO_SHADOW_LEARN=1). Nula nas linhas gravadas antes da coluna: origem
    # desconhecida (ver app.create_app, que confere o histórico de decisões)
    Column("learned_in_shadow", Boolean),
)

#: Colunas acrescentadas a `policy_posteriors` depois da criação (contrato 2.3).
POSTERIOR_COLUMNS = ("learned_in_shadow",)


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
        faltando = self._acrescentar(decisions, V2_COLUMNS)
        if faltando:
            log.warning("tabela decisions migrada para o esquema v2: %s", ", ".join(faltando))
        extra = self._acrescentar(policy_posteriors, POSTERIOR_COLUMNS)
        if extra:
            log.warning("tabela policy_posteriors migrada (contrato 2.3): %s", ", ".join(extra))
        return faltando

    def _acrescentar(self, tabela: Table, colunas) -> list[str]:
        existentes = {c["name"] for c in inspect(self.engine).get_columns(tabela.name)}
        faltando = [c for c in colunas if c not in existentes]
        if faltando:
            with self.engine.begin() as conn:
                for nome in faltando:
                    tipo = tabela.c[nome].type.compile(dialect=self.engine.dialect)
                    conn.execute(text(f"ALTER TABLE {tabela.name} ADD COLUMN {nome} {tipo}"))
        return faltando

    # ---- decisões --------------------------------------------------------

    def get(self, decision_id: str) -> dict | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(decisions).where(decisions.c.decision_id == decision_id)
            ).mappings().first()
        return dict(row) if row else None

    def insert(self, record: dict, outcome_window_min: int | None = None) -> bool:
        """Grava a decisão. Devolve False se o `decision_id` já estava gravado.

        A decisão gravada nunca é sobrescrita: numa corrida entre duas chamadas
        do mesmo ponto (outro processo, ou uma chamada que escapou do lock por
        `decision_id` do app), vale a que chegou primeiro ao banco, e quem chamou
        relê a linha (ver app.decide). Qualquer outra violação de integridade
        sobe.
        """
        from sqlalchemy.exc import IntegrityError

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
        try:
            with self.engine.begin() as conn:
                conn.execute(decisions.insert().values(**payload))
        except IntegrityError:
            # A transação foi desfeita. Se a linha existe, a violação foi a da
            # chave primária: outra chamada gravou este ponto primeiro. No
            # PostgreSQL o INSERT concorrente espera o commit da outra transação
            # antes de falhar, e a releitura (nova transação) já a enxerga.
            if self.get(payload["decision_id"]) is not None:
                return False
            raise
        return True

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

    def mark_shadow_override(self, decision_id: str, when: datetime | None = None) -> None:
        """Registra (uma vez) que a resposta desta decisão foi forçada a 0 pelo shadow."""
        with self.engine.begin() as conn:
            conn.execute(
                update(decisions).where(decisions.c.decision_id == decision_id)
                .where(decisions.c.shadow_override_at.is_(None))
                .values(shadow_override_at=_utc(when or datetime.now(timezone.utc)))
            )

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

    def close_day(self, day, policy_version: str, updates_applied: int,
                  snapshot: dict | None = None, digest: str | None = None,
                  learned_in_shadow: bool = False) -> bool:
        """Registra o fechamento e, se houver, a posterior resultante.

        As duas escritas vão na mesma transação: ou o dia fica fechado com a
        posterior dele gravada, ou nada muda. Devolve False se o dia já estava
        fechado (nesse caso a posterior também não é regravada).
        """
        from sqlalchemy.exc import IntegrityError

        now = datetime.now(timezone.utc)
        try:
            with self.engine.begin() as conn:
                conn.execute(policy_days.insert().values(
                    day=day, policy_version=policy_version,
                    updates_applied=updates_applied, closed_at=now))
                if snapshot is not None:
                    values = {"snapshot": json.dumps(snapshot, default=float),
                              "digest": digest, "schema": snapshot.get("schema", "v1"),
                              "day": day, "saved_at": now,
                              "learned_in_shadow": bool(learned_in_shadow)}
                    done = conn.execute(
                        update(policy_posteriors)
                        .where(policy_posteriors.c.policy_version == policy_version)
                        .values(**values)).rowcount
                    if not done:
                        conn.execute(policy_posteriors.insert().values(
                            policy_version=policy_version, **values))
        except IntegrityError:
            # Só a corrida no fechamento do mesmo dia é "já fechado"; qualquer
            # outra violação (posterior inválida) sobe, com a transação desfeita.
            if self.day_closed(day):
                return False
            raise
        return True

    def load_posterior(self, policy_version: str) -> dict | None:
        """Posterior persistida desta versão: {snapshot, digest, schema, day, saved_at}."""
        with self.engine.begin() as conn:
            row = conn.execute(
                select(policy_posteriors)
                .where(policy_posteriors.c.policy_version == policy_version)
            ).mappings().first()
        if row is None:
            return None
        out = dict(row)
        out["snapshot"] = json.loads(out["snapshot"])
        return out

    def persisted_versions(self) -> list[dict]:
        """Versões com posterior persistida (sem os pesos), da mais recente à mais antiga."""
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(policy_posteriors.c.policy_version, policy_posteriors.c.day,
                       policy_posteriors.c.digest, policy_posteriors.c.learned_in_shadow)
                .order_by(policy_posteriors.c.saved_at.desc())
            ).mappings().all()
        return [{"policy_version": r["policy_version"], "day": r["day"].isoformat(),
                 "digest": r["digest"], "learned_in_shadow": r["learned_in_shadow"]}
                for r in rows]

    def shadow_trained(self, policy_version: str) -> int:
        """Decisões de shadow desta versão que treinaram (o shadow aprendia).

        É o que permite reconhecer, numa linha de posterior anterior à coluna
        `learned_in_shadow`, que o histórico da linhagem passou por shadow com
        aprendizado (o comportamento de antes do contrato 2.3).
        """
        with self.engine.begin() as conn:
            return int(conn.execute(
                select(func.count()).select_from(decisions)
                .where(decisions.c.policy_version == policy_version)
                .where(decisions.c.shadow.is_(True))
                .where(decisions.c.trained.is_(True))
            ).scalar_one())

    def closed_days(self) -> list[dict]:
        """Dias fechados por /v1/end-day, em ordem."""
        with self.engine.begin() as conn:
            rows = conn.execute(select(policy_days).order_by(policy_days.c.day)).mappings().all()
        return [{"day": r["day"].isoformat(), "updates_applied": r["updates_applied"]}
                for r in rows]

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
