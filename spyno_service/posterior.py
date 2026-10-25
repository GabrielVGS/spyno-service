"""Origem e persistência da posterior servida.

Ordem de precedência na subida (ver `app.create_app` e o README):

1. ``persisted``: a posterior gravada no banco pelo último ``/v1/end-day``
   bem-sucedido **da mesma** ``policy_version``;
2. ``warmstart``: o artefato de `SPYNO_WARMSTART_SNAPSHOT`, gerado por
   ``training/build_warmstart.py``;
3. ``prior``: a posterior inicial da fábrica.

A ``policy_version`` é calculada sobre o estado *inicial* (prior ou
warm-start), antes de aplicar a posterior persistida. Assim:

- o warm-start muda a ``policy_version`` (é outro modelo inicial, e decisões
  tomadas a partir do prior não devem treinar uma posterior aquecida, nem o
  contrário);
- a posterior persistida não muda a ``policy_version``: ela é a continuação da
  mesma linhagem, e um reinício não pode invalidar o ``decision_id`` das
  decisões já tomadas nem descartar os desfechos que ainda vão chegar.

Uma posterior persistida de outra ``policy_version`` (outro esquema, outro
clipping, outro warm-start) nunca é carregada; a recusa vai para o log e para
``/v1/policy-state``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .features import FEATURES, POLICY_SCHEMA
from .policies.bandit import IncompatibleSnapshotError

#: Formato do artefato de warm-start (``training/build_warmstart.py``).
WARMSTART_FORMAT = "spyno-warmstart/1"

ORIGINS = ("persisted", "warmstart", "prior")


def digest(snapshot: dict) -> str:
    """Impressão digital de um snapshot (m, q, custo, esquema). É a de /v1/policy-state."""
    raw = json.dumps(snapshot, sort_keys=True, default=float)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def has_posterior(policy) -> bool:
    """A política tem posterior salvável e recarregável (os TS do spyno-decision)."""
    return callable(getattr(policy, "snapshot", None)) and \
        callable(getattr(policy, "load_snapshot", None))


def snapshot_of(policy) -> dict | None:
    """Snapshot da política, ou None se ela não tem posterior (never, always, PPO)."""
    snap = getattr(policy, "snapshot", None)
    if not callable(snap):
        return None
    return snap()


def read_warmstart(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SPYNO_WARMSTART_SNAPSHOT aponta para {p}, que não existe")
    raw = p.read_bytes()
    art = json.loads(raw)
    art["_sha256"] = hashlib.sha256(raw).hexdigest()
    art["_path"] = str(p)
    return art


def check_warmstart(art: dict, policy_name: str, cost: float) -> None:
    """Recusa um artefato que não foi gerado para a política servida.

    O esquema e a dimensão são conferidos depois, por `load_snapshot`
    (`bandit._check_snapshot`). Aqui ficam o que o snapshot sozinho não
    distingue: `ts_disjoint` e `ts_binary` têm posteriors de mesma forma, e
    `load_snapshot` sobrescreveria o custo sem avisar.
    """
    fmt = art.get("format")
    if fmt != WARMSTART_FORMAT:
        raise IncompatibleSnapshotError(
            f"artefato de warm-start em formato '{fmt}', esperado '{WARMSTART_FORMAT}'")
    if art.get("schema", "v1") != POLICY_SCHEMA:
        raise IncompatibleSnapshotError(
            f"warm-start gerado no esquema '{art.get('schema', 'v1')}', mas o serviço "
            f"serve o esquema '{POLICY_SCHEMA}'. Regere com training/build_warmstart.py")
    if list(art.get("features", [])) != list(FEATURES):
        raise IncompatibleSnapshotError(
            "warm-start gerado com outro vetor de features: "
            f"{art.get('features')} != {list(FEATURES)}")
    if art.get("policy") != policy_name:
        raise IncompatibleSnapshotError(
            f"warm-start gerado para '{art.get('policy')}', mas a política ativa é "
            f"'{policy_name}'")
    if not math.isclose(float(art.get("cost", float("nan"))), float(cost),
                        rel_tol=0.0, abs_tol=1e-12):
        raise IncompatibleSnapshotError(
            f"warm-start gerado com custo {art.get('cost')}, mas a política usa {cost}")


def apply_warmstart(policy, policy_name: str, path: str | Path) -> dict:
    """Carrega o artefato na política. Falha alto: quem pediu o warm-start quer ele."""
    load = getattr(policy, "load_snapshot", None)
    if not callable(load):
        raise IncompatibleSnapshotError(
            f"a política '{policy_name}' não tem posterior; SPYNO_WARMSTART_SNAPSHOT "
            "só vale para ts_hybrid, ts_disjoint e ts_binary")
    art = read_warmstart(path)
    check_warmstart(art, policy_name, getattr(policy, "cost", float("nan")))
    load(art)
    prov = art.get("provenance", {})
    return {"path": art["_path"], "sha256": art["_sha256"],
            "posterior_digest": digest(snapshot_of(policy)),
            "recipe": art.get("recipe", {}).get("summary"),
            "spyno_sim_commit": prov.get("spyno_sim", {}).get("commit"),
            "spyno_decision": prov.get("spyno_decision", {}).get("origin")}
