"""Memory module for the Agentic Incident Postmortem Synthesizer — Phase 4.

Wraps `rag.py` with the consult-only policy: recalled prior incidents are
*advisory only* and must never be cited as ground-truth evidence for the current
incident. Enforced two ways (mirrors the verifier's `from_recalled_incident`
rule in `store.py`):
  1. `recall_incidents` returns `ConsultedIncident` objects (default `applied=False`).
  2. `validate_consult_only` / `assert_consult_only` reject any postmortem whose
     `claims[].evidence_refs` contain a recalled incident id (the hard ban).

`embed_postmortem(postmortem)` rebuilds the EXACT `incident_memory` document shape
produced by `generate_incidents.memory_doc` (single source of truth) so live
embedded incidents stay compatible with the pre-seeded `memory_seed.json` fixtures.

Quality is stamped into the Chroma metadata: `verification_score` (the deterministic
backed-claim ratio) travels alongside `action_item_count` so recall can surface a
mix of good and imperfect priors. The hard consult-only ban is enforced separately
(`is_consult_only_leak`): a postmortem that leaks a recalled incident id into claim
evidence is NEVER embedded, regardless of quality.
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


def embed_postmortem(postmortem: Postmortem,
                     time_approved: Optional[str] = None,
                     verification_score: Optional[float] = None) -> Dict[str, Any]:
    """Build the `incident_memory` doc for a postmortem (approved OR imperfect).

    Delegates to `generate_incidents.memory_doc` so the shape matches fixtures;
    only `time_approved`, the (empty) red-herring, the real `action_item_count`,
    and the quality `verification_score` are reconciled afterward. The hard
    consult-only ban is NOT checked here — callers must refuse to embed leaks via
    `is_consult_only_leak` (see `memory_writer_node`).
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
    # Faithful action-item count (memory_doc hardcodes 1 for the seed shape).
    doc["metadata"]["action_item_count"] = len(postmortem.action_items)
    # Stamp deterministic quality so recall can surface imperfect priors too.
    if verification_score is not None:
        doc["metadata"]["verification_score"] = float(verification_score)
    return doc


async def store_memory(postmortem: Postmortem,
                       collection: Any,
                       time_approved: Optional[str] = None,
                       verification_score: Optional[float] = None) -> Dict[str, Any]:
    """Embed + persist a (non-leaking) postmortem into the `incident_memory` collection."""
    doc = embed_postmortem(postmortem, time_approved, verification_score)
    await collection.add(
        id=doc["incident_id"], document=doc["document"], metadata=doc["metadata"]
    )
    return doc


def is_consult_only_leak(postmortem: Postmortem,
                         recalled_incident_ids: List[str]) -> bool:
    """True if any claim cites a recalled incident id as evidence (hard ban).

    Covers both the explicit `from_recalled_incident` marker and a leaked id in
    `claims[].evidence_refs`. Such postmortems must NEVER be embedded.
    """
    banned = set(recalled_incident_ids)
    for claim in postmortem.claims:
        if claim.from_recalled_incident is not None:
            return True
        if set(claim.evidence_refs) & banned:
            return True
    return False


async def recall_incidents(symptom_text: str,
                           collection: Any,
                           n: int = 5,
                           where: Optional[Dict[str, Any]] = None) -> List[ConsultedIncident]:
    """Recall up to `n` prior incidents from `incident_memory` (consult-only)."""
    return await collection.recall(symptom_text, n=n, where=where)


def validate_consult_only(postmortem: Postmortem,
                           recalled_incident_ids: List[str]) -> List[Dict[str, Any]]:
    """Return violations where a recalled incident id leaked into a claim.

    A recalled incident id (e.g. "INC-014") must appear only via
    `consulted_incidents` / `from_prior_incident`, never as an evidence ref and
    never via `from_recalled_incident`. This is the single source of truth for
    the consult-only ban, matching `is_consult_only_leak` (memory writer) and the
    human gate.
    """
    banned = set(recalled_incident_ids)
    violations: List[Dict[str, Any]] = []
    for i, claim in enumerate(postmortem.claims):
        leaked = [r for r in claim.evidence_refs if r in banned]
        if claim.from_recalled_incident is not None and claim.from_recalled_incident in banned:
            leaked = leaked + [claim.from_recalled_incident]
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
