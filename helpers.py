"""Shared test helpers for the Agentic Incident Postmortem Synthesizer.

Centralises the offline `FakeLLMAdapter` and the `incident_memory` collection /
incident fixtures so `tests/test_agents.py` and `tests/test_graph_integration.py`
don't duplicate them. Importing this module seeds `INCS` from `generate_incidents`
and builds the `INC-001` `EVIDENCE` list used across the suite.
"""

from __future__ import annotations

from typing import List

from generate_incidents import generate_incidents
from rag import create_memory_collection
from schemas import (
    Claim,
    ConsultedIncident,
    Evidence,
    Postmortem,
    RootCauseCandidate,
    TimelineEvent,
    VerificationReport,
)
from store import (
    connect,
    init_db,
    insert_evidence,
    insert_incident,
)
from llm_adapter import LLMAdapter

INCS = {i["id"]: i for i in generate_incidents()}
INC = INCS["INC-001"]
EVIDENCE = [
    Evidence(
        id=e["id"],
        incident_id=INC["id"],
        ts=e["ts"],
        source=e["source"],
        source_url=e.get("source_url"),
        content=e["content"],
    )
    for e in INC["evidence"]
]


def base_state() -> dict:
    return {
        "incident_id": INC["id"],
        "window_start": INC["window_start"],
        "window_end": INC["window_end"],
        "description": INC["description"],
        "evidence": EVIDENCE,
    }


def seed(conn) -> None:
    init_db(conn)
    insert_incident(
        conn,
        {
            "id": INC["id"],
            "window_start": INC["window_start"],
            "window_end": INC["window_end"],
            "description": INC["description"],
        },
        status="running",
    )
    insert_evidence(conn, INC["id"], [e.model_dump() for e in EVIDENCE])


def _hash_embed(texts: List[str]):
    return LLMAdapter()._hash_embed(texts)


def make_collection():
    return create_memory_collection(
        name="incident_memory_test", embed_fn=_hash_embed, backend="memory"
    )


async def seed_prior(collection) -> None:
    """Embed one prior incident so recall returns a consulted hypothesis."""
    from generate_incidents import memory_doc

    doc = memory_doc(INCS["INC-002"])
    await collection.add(
        id=doc["incident_id"], document=doc["document"], metadata=doc["metadata"]
    )


# -------------------------------------------------------------- Fake LLM adapter
# Honesty note (Design Risk #4): the original FakeLLM returned
# `config_timeout_drop` for *every* incident, so the agent's
# red-herring rate was 1.0 by construction. This revision is incident-specific:
# it inspects the prompt for the incident id and returns that incident's
# `true_root_cause` as the top candidate/postmortem label. The remaining
# limitation is documented in the eval output — citation/structure, not a live
# LLM's reasoning, is what is being faithfuiy measured.

_INC_RE = __import__("re").compile(r"INC-\d+")


def _extract_incident_id(messages) -> str | None:
    for m in messages or []:
        txt = m.get("content", "") if isinstance(m, dict) else str(m)
        found = _INC_RE.search(txt)
        if found:
            return found.group(0)
    return None


def _good_postmortem_for(incident_id: str) -> Postmortem:
    # Preserve exact fixture for INC-001 so existing tests that compare
    # Fake output to _good_postmortem() remain stable.
    if incident_id == "INC-001":
        return _good_postmortem()
    inc = INCS.get(incident_id)
    if inc is None:
        return _good_postmortem()
    ev = inc["evidence"]
    # Use real evidence ids so verification passes for this incident.
    # Timeline: first two evidence items; claims: first and last.
    e1 = ev[0]["id"] if ev else "E1"
    e2 = ev[1]["id"] if len(ev) > 1 else e1
    e_last = ev[-1]["id"] if ev else "E1"
    # Try to pick a second claim that is verifiable (not hallucinated).
    return Postmortem(
        incident_id=incident_id,
        summary=f"Postmortem for {incident_id}: {inc['description']}",
        impact=f"Incident {incident_id} impact derived from evidence {e1}.",
        root_cause=inc.get("true_root_cause", "config_timeout_drop"),
        timeline=[
            TimelineEvent(ts=ev[0]["ts"], description=ev[0]["content"][:120], evidence_refs=[e1]),
            TimelineEvent(ts=ev[1]["ts"] if len(ev) > 1 else ev[0]["ts"], description=ev[1]["content"][:120] if len(ev) > 1 else ev[0]["content"][:120], evidence_refs=[e2]),
        ],
        action_items=[f"Mitigate {inc.get('true_root_cause', 'root cause')} per evidence {e1}."],
        claims=[
            Claim(statement=f"Root cause {inc.get('true_root_cause')} evidenced by {e1}", evidence_refs=[e1, e2] if e2 != e1 else [e1]),
            Claim(statement=f"Red-herring observation {e_last} was coincidental", evidence_refs=[e_last]),
        ],
        consulted_incidents=[],
    )


class FakeLLMAdapter:
    def __init__(self, bad_postmortem_first: bool = False):
        self.bad_postmortem_first = bad_postmortem_first
        self._pm_calls = 0
        self.calls = []

    async def chat(self, system, messages, model=None, params=None, schema=None):
        self.calls.append((schema, system, messages))
        kind = self._schema_name(schema)
        iid = _extract_incident_id(messages) or "INC-001"
        if kind == "postmortem":
            self._pm_calls += 1
            if self.bad_postmortem_first and self._pm_calls == 1:
                # Keep the incident-specific id but hallucinate E99.
                pm = _good_postmortem_for(iid)
                return pm.model_copy(
                    update={"claims": [Claim(statement="timeout drop caused failures", evidence_refs=["E1", "E99"])]}
                )
            return _good_postmortem_for(iid)
        if kind == "timeline":
            inc = INCS.get(iid)
            if inc and len(inc["evidence"]) >= 2:
                ev = inc["evidence"]
                return [
                    TimelineEvent(ts=ev[0]["ts"], description=ev[0]["content"][:120], evidence_refs=[ev[0]["id"]]),
                    TimelineEvent(ts=ev[1]["ts"], description=ev[1]["content"][:120], evidence_refs=[ev[1]["id"]]),
                ]
            return [
                TimelineEvent(ts="2026-08-20T14:02:00", description="deploy dropped timeout", evidence_refs=["E1"]),
                TimelineEvent(ts="2026-08-20T14:12:00", description="payment gateway timeout", evidence_refs=["E4"]),
            ]
        if kind == "candidates":
            inc = INCS.get(iid)
            if inc:
                ev = inc["evidence"]
                e_ids = [e["id"] for e in ev]
                # supporting = first two, contradicting = last (red-herring-ish)
                return [
                    RootCauseCandidate(
                        rank=1,
                        confidence=0.9,
                        hypothesis=f"Root cause is {inc.get('true_root_cause')}",
                        root_cause_label=inc.get("true_root_cause", "config_timeout_drop"),
                        supporting_evidence=e_ids[:2] if len(e_ids) >= 2 else e_ids[:1],
                        contradicting_evidence=[e_ids[-1]] if len(e_ids) >= 1 else [],
                        from_prior_incident=None,
                    )
                ]
            return [
                RootCauseCandidate(
                    rank=1,
                    confidence=0.9,
                    hypothesis="payment timeout config drop",
                    root_cause_label="config_timeout_drop",
                    supporting_evidence=["E1", "E4"],
                    contradicting_evidence=["E5"],
                    from_prior_incident=None,
                )
            ]
        return None

    async def embed(self, texts, model=None):
        return _hash_embed(texts)

    @staticmethod
    def _schema_name(schema):
        origin = getattr(schema, "__origin__", None)
        if origin is list:
            args = getattr(schema, "__args__", ())
            if args and args[0] is TimelineEvent:
                return "timeline"
            if args and args[0] is RootCauseCandidate:
                return "candidates"
        if schema is Postmortem:
            return "postmortem"
        return "unknown"


def _good_postmortem() -> Postmortem:
    return Postmortem(
        incident_id="INC-001",
        summary="Checkout failures after payment timeout drop.",
        impact="5.3% checkout error rate after the 14:02 deploy.",
        root_cause="config_timeout_drop",
        timeline=[
            TimelineEvent(ts="2026-08-20T14:02:00", description="deploy dropped timeout", evidence_refs=["E1"]),
            TimelineEvent(ts="2026-08-20T14:12:00", description="payment gateway timeout", evidence_refs=["E4"]),
        ],
        action_items=["Restore payment_timeout_ms to 2000."],
        claims=[
            Claim(statement="14:02 deploy dropped timeout causing failures", evidence_refs=["E1", "E4"]),
            Claim(statement="DB CPU spike at 14:25 was coincidental", evidence_refs=["E5"]),
        ],
        consulted_incidents=[],
    )


def _bad_postmortem() -> Postmortem:
    pm = _good_postmortem()
    # Inject a hallucinated evidence ref E99 to trigger the citation guard.
    return pm.model_copy(
        update={
            "claims": [
                Claim(statement="timeout drop caused failures", evidence_refs=["E1", "E99"]),
            ]
        }
    )
