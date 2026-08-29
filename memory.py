"""Memory module for the Agentic Incident Postmortem Synthesizer — Phase 4.

Wraps `rag.py` with the consult-only policy: recalled prior incidents are
*advisory only* and must never be cited as ground-truth evidence for the current
incident. Enforced two ways (mirrors the verifier's `from_recalled_incident`
rule in `store.py`):
  1. `recall_incidents` returns `ConsultedIncident` objects (default `applied=False`).
  2. `validate_consult_only` / `assert_consult_only` reject any postmortem whose
     `claims[].evidence_refs` contain a recalled incident id (the hard ban).

`embed_approved(postmortem)` rebuilds the EXACT `incident_memory` document shape
produced by `generate_incidents.memory_doc` (single source of truth) so live
embedded incidents stay compatible with the pre-seeded `memory_seed.json` fixtures.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from schemas import ConsultedIncident, Postmortem

from generate_incidents import memory_doc
from rag import create_memory_collection


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _postmortem_to_incident(postmortem: Postmortem, time_approved: str) -> Dict[str, Any]:
    """Map an approved Postmortem back into the incident shape `memory_doc` expects."""
    return {
        "id": postmortem.incident_id,
        "description": postmortem.summary,
        "evidence": [{"content": t.description} for t in postmortem.timeline],
        "true_root_cause": postmortem.root_cause,
        # A postmortem carries no red-herring label; empty keeps the shape identical.
        "red_herring": "",
        "window_end": time_approved,
    }


def embed_approved(postmortem: Postmortem,
                   time_approved: Optional[str] = None) -> Dict[str, Any]:
    """Build the `incident_memory` doc for an approved postmortem.

    Delegates to `generate_incidents.memory_doc` so the shape matches fixtures;
    only `time_approved` and the (empty) red-herring are reconciled afterward.
    """
    time_approved = time_approved or _now_iso()
    inc = _postmortem_to_incident(postmortem, time_approved)
    doc = memory_doc(inc)
    # Real approval time overrides the synthetic window_end.
    doc["metadata"]["time_approved"] = time_approved
    # Drop the empty token produced by the empty red_herring.
    doc["metadata"]["symptom_keywords"] = ",".join(
        t for t in doc["metadata"]["symptom_keywords"].split(",") if t
    )
    return doc


async def store_approved(postmortem: Postmortem,
                         collection: Any,
                         time_approved: Optional[str] = None) -> Dict[str, Any]:
    """Embed + persist an approved postmortem into the `incident_memory` collection."""
    doc = embed_approved(postmortem, time_approved)
    await collection.add(
        id=doc["incident_id"], document=doc["document"], metadata=doc["metadata"]
    )
    return doc


async def recall_incidents(symptom_text: str,
                           collection: Any,
                           n: int = 5,
                           where: Optional[Dict[str, Any]] = None) -> List[ConsultedIncident]:
    """Recall up to `n` prior incidents from `incident_memory` (consult-only)."""
    return await collection.recall(symptom_text, n=n, where=where)


def validate_consult_only(postmortem: Postmortem,
                          recalled_incident_ids: List[str]) -> List[Dict[str, Any]]:
    """Return violations where a recalled incident id leaked into `evidence_refs`.

    A recalled incident id (e.g. "INC-014") must appear only via
    `consulted_incidents` / `from_prior_incident`, never as an evidence ref.
    """
    banned = set(recalled_incident_ids)
    violations: List[Dict[str, Any]] = []
    for i, claim in enumerate(postmortem.claims):
        leaked = [r for r in claim.evidence_refs if r in banned]
        if leaked:
            violations.append(
                {"claim_index": i, "statement": claim.statement, "leaked_refs": leaked}
            )
    return violations


def assert_consult_only(postmortem: Postmortem,
                        recalled_incident_ids: List[str]) -> None:
    """Raise ValueError if any recalled incident id is cited as evidence."""
    violations = validate_consult_only(postmortem, recalled_incident_ids)
    if violations:
        detail = "; ".join(
            f"claim#{v['claim_index']} leaked {v['leaked_refs']}" for v in violations
        )
        raise ValueError(f"Consult-only violation: {detail}")


def get_collection(embed_fn: Optional[Any] = None, backend: str = "auto",
                   persist_dir: Optional[str] = None) -> Any:
    """Convenience accessor for the shared `incident_memory` collection."""
    return create_memory_collection(
        name="incident_memory", embed_fn=embed_fn, backend=backend,
        persist_dir=persist_dir,
    )
