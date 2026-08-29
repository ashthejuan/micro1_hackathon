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
class FakeLLMAdapter:
    def __init__(self, bad_postmortem_first: bool = False):
        self.bad_postmortem_first = bad_postmortem_first
        self._pm_calls = 0
        self.calls = []

    async def chat(self, system, messages, model=None, params=None, schema=None):
        self.calls.append((schema, system, messages))
        kind = self._schema_name(schema)
        if kind == "postmortem":
            self._pm_calls += 1
            if self.bad_postmortem_first and self._pm_calls == 1:
                return _bad_postmortem()
            return _good_postmortem()
        if kind == "timeline":
            return [
                TimelineEvent(ts="2026-08-20T14:02:00", description="deploy dropped timeout", evidence_refs=["E1"]),
                TimelineEvent(ts="2026-08-20T14:12:00", description="payment gateway timeout", evidence_refs=["E4"]),
            ]
        if kind == "candidates":
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
