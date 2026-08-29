"""Phase 5 — agents.py + tracer.py tests.

Drives the seven async nodes with a `FakeLLMAdapter` (no network, no key) and a
real in-memory SQLite store + in-memory Chroma collection. Covers schema returns,
the §6.2 citation-integrity retry (fires exactly once then passes), the
deterministic verifier math, the auto-approve gate persisting approver/time, the
memory writer embedding, and §B.4 trajectory files (prompts, tools, retries,
human checkpoint, verifier math + manifest).
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_incidents import generate_incidents  # noqa: E402
from rag import create_memory_collection  # noqa: E402
from schemas import (  # noqa: E402
    Claim,
    ClaimVerification,
    ConsultedIncident,
    Evidence,
    Postmortem,
    RootCauseCandidate,
    TimelineEvent,
    VerificationReport,
)
from store import (  # noqa: E402
    connect,
    init_db,
    insert_evidence,
    insert_incident,
    query_evidence,
    upsert_postmortem,
    valid_evidence_ids,
    verify_postmortem,
)
from tracer import Tracer  # noqa: E402
from agents import (  # noqa: E402
    analyze_node,
    human_gate_node,
    ingest_node,
    memory_writer_node,
    timeline_node,
    verifier_node,
    writer_node,
    _bad_refs,
)


# --------------------------------------------------------------------- fixtures
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


def base_state():
    return {
        "incident_id": INC["id"],
        "window_start": INC["window_start"],
        "window_end": INC["window_end"],
        "description": INC["description"],
        "evidence": EVIDENCE,
    }


def seed(conn):
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


def _hash_embed(texts):
    from llm_adapter import LLMAdapter

    return LLMAdapter()._hash_embed(texts)


def make_collection():
    return create_memory_collection(
        name="incident_memory_test", embed_fn=_hash_embed, backend="memory"
    )


async def seed_prior(collection):
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


def _good_postmortem():
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


def _bad_postmortem():
    pm = _good_postmortem()
    # Inject a hallucinated evidence ref E99 to trigger the citation guard.
    return pm.model_copy(
        update={
            "claims": [
                Claim(statement="timeout drop caused failures", evidence_refs=["E1", "E99"]),
            ]
        }
    )


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------- tests
def test_ingest_writes_store_and_fires_recall(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        coll = make_collection()
        await seed_prior(coll)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        res = await ingest_node(base_state(), llm=fake, tracer=tracer, conn=conn, collection=coll)
        assert res["recall_task"] is not None
        rows = query_evidence(conn, "INC-001")
        assert len(rows) == len(EVIDENCE)
        assert os.path.exists(os.path.join(str(tmp_path), "INC-001", "ingest.json"))

    run(main())


def test_timeline_returns_schema(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        seed(conn)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        res = await timeline_node(base_state(), llm=fake, tracer=tracer, conn=conn, collection=None)
        assert len(res["timeline_events"]) == 2
        assert all(isinstance(t, TimelineEvent) for t in res["timeline_events"])
        assert os.path.exists(os.path.join(str(tmp_path), "INC-001", "timeline.json"))

    run(main())


def test_analyze_returns_candidates_and_consulted(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        seed(conn)
        coll = make_collection()
        await seed_prior(coll)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        s = base_state()
        r = await ingest_node(s, llm=fake, tracer=tracer, conn=conn, collection=coll)
        s.update(r)
        res = await analyze_node(s, llm=fake, tracer=tracer, conn=conn, collection=coll)
        assert len(res["candidates"]) == 1
        assert isinstance(res["candidates"][0], RootCauseCandidate)
        assert res["candidates"][0].root_cause_label == "config_timeout_drop"
        assert len(res["consulted"]) == 1
        assert res["consulted"][0].incident_id == "INC-002"

    run(main())


def test_writer_citation_retry_fires_once(tmp_path):
    async def main():
        fake = FakeLLMAdapter(bad_postmortem_first=True)
        conn = connect(":memory:")
        seed(conn)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        s = {
            "incident_id": "INC-001",
            "description": INC["description"],
            "timeline_events": [
                TimelineEvent(ts="2026-08-20T14:02:00", description="deploy", evidence_refs=["E1"]),
            ],
            "candidates": [
                RootCauseCandidate(rank=1, confidence=0.9, hypothesis="x", root_cause_label="config_timeout_drop",
                                   supporting_evidence=["E1"], contradicting_evidence=["E5"]),
            ],
            "consulted": [],
        }
        res = await writer_node(s, llm=fake, tracer=tracer, conn=conn, collection=None)
        pm = res["postmortem"]
        bad = {r for r in _collect(pm) if r not in valid_evidence_ids(conn, "INC-001")}
        assert bad == set()  # corrected after retry
        wj = json.load(open(os.path.join(str(tmp_path), "INC-001", "writer.json")))
        assert len(wj["retries"]) == 1
        assert "E99" in wj["retries"][0]["reason"]

    run(main())


def test_bad_refs_flags_empty_claim_evidence():
    pm = _good_postmortem().model_copy(
        update={
            "claims": [
                Claim(statement="a claim with no citations", evidence_refs=[]),
            ]
        }
    )
    cands = [
        RootCauseCandidate(
            rank=1, confidence=0.9, hypothesis="x", root_cause_label="config_timeout_drop",
            supporting_evidence=["E1"], contradicting_evidence=["E5"],
        )
    ]
    violations = _bad_refs(pm, cands, {"E1", "E4", "E5"})
    assert any("empty evidence_refs" in v for v in violations)


def _collect(pm):
    refs = []
    for ev in pm.timeline:
        refs.extend(ev.evidence_refs)
    for c in pm.claims:
        refs.extend(c.evidence_refs)
    return refs


def test_verifier_computes_backed_deterministically(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        seed(conn)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))

        # valid refs -> all backed
        pm = _good_postmortem()
        s = {"incident_id": "INC-001", "postmortem": pm, "consulted": []}
        res = await verifier_node(s, llm=fake, tracer=tracer, conn=conn, collection=None)
        assert res["verification"].verification_score == 1.0

        # hallucinated ref -> not backed
        pm_bad = pm.model_copy(update={"claims": [Claim(statement="x", evidence_refs=["E99"])]})
        s2 = {"incident_id": "INC-001", "postmortem": pm_bad, "consulted": []}
        res2 = await verifier_node(s2, llm=fake, tracer=tracer, conn=conn, collection=None)
        assert res2["verification"].verification_score == 0.0

        # recalled ref cited as evidence -> not backed (consult-only ban)
        pm_rec = pm.model_copy(
            update={"claims": [Claim(statement="x", evidence_refs=["E1"], from_recalled_incident="INC-002")]}
        )
        s3 = {"incident_id": "INC-001", "postmortem": pm_rec, "consulted": [ConsultedIncident(incident_id="INC-002", similarity_score=0.9, note="x")]}
        res3 = await verifier_node(s3, llm=fake, tracer=tracer, conn=conn, collection=None)
        assert res3["verification"].verification_score == 0.0

        # verifier_math logged
        vj = json.load(open(os.path.join(str(tmp_path), "INC-001", "verifier.json")))
        assert "verifier_math" in vj

    run(main())


def test_human_gate_auto_approve_persists_approver(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        seed(conn)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        s = {
            "incident_id": "INC-001",
            "postmortem": _good_postmortem(),
            "verification": VerificationReport(incident_id="INC-001", claim_reports=[], verification_score=1.0),
            "consulted": [],
        }
        res = await human_gate_node(s, llm=fake, tracer=tracer, conn=conn, collection=None)
        assert res["human_decision"] == "approved"
        row = conn.execute(
            "SELECT status, approved_by, time_approved FROM postmortem WHERE incident_id=?",
            ("INC-001",),
        ).fetchone()
        assert row["status"] == "approved"
        assert row["approved_by"] == "auto-eval"
        assert row["time_approved"] is not None

    run(main())


def test_memory_writer_embeds_approved(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        coll = make_collection()
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        s = {"incident_id": "INC-001", "human_decision": "approved", "postmortem": _good_postmortem()}
        before = coll.count()
        await memory_writer_node(s, llm=fake, tracer=tracer, conn=conn, collection=coll)
        assert coll.count() == before + 1

    run(main())


def test_full_node_sequence_welds_pipeline_and_traces(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        seed(conn)
        coll = make_collection()
        await seed_prior(coll)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))

        s = base_state()
        for node in (ingest_node, timeline_node, analyze_node, writer_node, verifier_node, human_gate_node, memory_writer_node):
            res = await node(s, llm=fake, tracer=tracer, conn=conn, collection=coll)
            s.update(res)

        # pipeline outputs complete
        assert isinstance(s["postmortem"], Postmortem)
        assert isinstance(s["verification"], VerificationReport)
        assert s["human_decision"] == "approved"

        # consult-only: no recalled id leaked into evidence_refs
        recalled = {c.incident_id for c in s["consulted"]}
        for claim in s["postmortem"].claims:
            assert not (set(claim.evidence_refs) & recalled)

        # memory grew by the embedded approved postmortem (1 seeded + 1 embedded)
        assert coll.count() == 2

        # trajectory pack: every agent file present + manifest indexes them
        agents = ["ingest", "timeline", "analyze", "writer", "verifier", "human_gate", "memory_writer"]
        for a in agents:
            path = os.path.join(str(tmp_path), "INC-001", f"{a}.json")
            assert os.path.exists(path)
            ev = json.load(open(path))
            for field in ("system_prompt", "user_prompt", "input_state", "tool_calls", "output"):
                assert field in ev, f"{a} missing {field}"
        manifest = json.load(open(os.path.join(str(tmp_path), "INC-001", "manifest.json")))
        assert set(manifest["agents"]) == set(agents)

        # human checkpoint + verifier math captured
        gate = json.load(open(os.path.join(str(tmp_path), "INC-001", "human_gate.json")))
        assert gate["human_decision"] == "approved"
        ver = json.load(open(os.path.join(str(tmp_path), "INC-001", "verifier.json")))
        assert "verifier_math" in ver

    run(main())


def test_analyze_without_ingest_collection_none(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        init_db(conn)
        insert_incident(
            conn,
            {"id": "INC-001", "window_start": "w", "window_end": "x", "description": "d"},
        )
        insert_evidence(conn, "INC-001", [e.model_dump() for e in EVIDENCE])
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        # No ingest ran, so no recall_task in state and collection is None:
        # analyze must tolerate a missing recall_task and emit empty consulted.
        s = {"incident_id": "INC-001", "description": "x"}
        res = await analyze_node(s, llm=fake, tracer=tracer, conn=conn, collection=None)
        assert isinstance(res["candidates"][0], RootCauseCandidate)
        assert res["consulted"] == []

    run(main())


def test_ingest_idempotent_no_cascade(tmp_path):
    conn = connect(":memory:")
    seed(conn)
    # Simulate a postmortem already persisted for this incident.
    upsert_postmortem(
        conn, "INC-001", draft_json="{}", verification_json="{}",
        status="approved", approved_by="human", time_approved="t",
    )
    # Re-ingest the SAME incident (eval re-run) — must not cascade-delete prior data.
    insert_incident(
        conn,
        {"id": "INC-001", "window_start": "w", "window_end": "x", "description": "d"},
        status="running",
    )
    insert_evidence(conn, "INC-001", [e.model_dump() for e in EVIDENCE])

    row = conn.execute(
        "SELECT status FROM postmortem WHERE incident_id=?", ("INC-001",)
    ).fetchone()
    assert row is not None, "postmortem was cascade-deleted by re-ingest"
    assert row["status"] == "approved"
    assert len(query_evidence(conn, "INC-001")) == len(EVIDENCE)


def test_memory_writer_noop_branches(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        coll = make_collection()

        # collection=None -> no-op
        assert await memory_writer_node(
            {"incident_id": "INC-001"}, llm=fake, tracer=tracer, conn=conn, collection=None
        ) == {}

        # missing postmortem -> no-op
        s = {"incident_id": "INC-001", "human_decision": "approved"}
        assert await memory_writer_node(
            s, llm=fake, tracer=tracer, conn=conn, collection=coll
        ) == {}
        assert coll.count() == 0

    run(main())


def test_memory_writer_embeds_imperfect_non_leak_postmortem(tmp_path):
    """A rejected (failed-verification) but non-leaking postmortem is still embedded
    with its verification_score stamped into the metadata."""
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        coll = make_collection()
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        pm = _good_postmortem()
        report = VerificationReport(
            incident_id="INC-001",
            claim_reports=[
                ClaimVerification(claim_index=0, statement="x", evidence_refs=["E1", "E4"],
                                  from_recalled_incident=None, backed=True, missing_evidence=[]),
                ClaimVerification(claim_index=1, statement="y", evidence_refs=["E99"],
                                  from_recalled_incident=None, backed=False, missing_evidence=["E99"]),
            ],
            verification_score=0.5,
        )
        # decision is "rejected" for failed verification, but not a leak -> embed.
        s = {
            "incident_id": "INC-001",
            "human_decision": "rejected",
            "postmortem": pm,
            "verification": report,
            "consulted": [],
        }
        before = coll.count()
        await memory_writer_node(s, llm=fake, tracer=tracer, conn=conn, collection=coll)
        assert coll.count() == before + 1
        # The embedded doc carries the quality score in metadata.
        doc = next(d for d in coll._docs if d["id"] == "INC-001")
        assert doc["metadata"]["verification_score"] == 0.5

    run(main())


def test_memory_writer_never_embeds_consult_only_leak(tmp_path):
    """A consult-only leak must never be embedded, even if the gate let it through."""
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        coll = make_collection()
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        pm = _good_postmortem().model_copy(
            update={"claims": [Claim(statement="x", evidence_refs=["INC-002"])]}
        )
        s = {
            "incident_id": "INC-001",
            "human_decision": "rejected",
            "postmortem": pm,
            "verification": VerificationReport(incident_id="INC-001", claim_reports=[], verification_score=1.0),
            "consulted": [ConsultedIncident(incident_id="INC-002", similarity_score=0.9, note="x")],
        }
        before = coll.count()
        await memory_writer_node(s, llm=fake, tracer=tracer, conn=conn, collection=coll)
        assert coll.count() == before  # leak never embedded

    run(main())


def test_human_gate_applied_flip_returned(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        seed(conn)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        consulted = [ConsultedIncident(incident_id="INC-002", similarity_score=0.9, note="x", applied=False)]
        s = {
            "incident_id": "INC-001",
            "postmortem": _good_postmortem(),
            "verification": VerificationReport(incident_id="INC-001", claim_reports=[], verification_score=1.0),
            "consulted": consulted,
            "applied_incidents": ["INC-002"],
        }
        res = await human_gate_node(s, llm=fake, tracer=tracer, conn=conn, collection=None)
        returned = {c.incident_id: c.applied for c in res["consulted"]}
        assert returned["INC-002"] is True

    run(main())


def test_human_gate_rejects_failed_verification(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        seed(conn)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        report = VerificationReport(
            incident_id="INC-001",
            claim_reports=[
                ClaimVerification(
                    claim_index=0, statement="x", evidence_refs=["E99"],
                    from_recalled_incident=None, backed=False, missing_evidence=["E99"],
                )
            ],
            verification_score=0.0,
        )
        s = {
            "incident_id": "INC-001",
            "postmortem": _good_postmortem(),
            "verification": report,
            "consulted": [],
        }
        res = await human_gate_node(s, llm=fake, tracer=tracer, conn=conn, collection=None)
        assert res["human_decision"] == "rejected"
        row = conn.execute(
            "SELECT status, approved_by FROM postmortem WHERE incident_id=?", ("INC-001",)
        ).fetchone()
        assert row["status"] == "rejected"
        assert row["approved_by"] is None

    run(main())


def test_human_gate_rejects_consult_only_leakage(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        seed(conn)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        # Cited a recalled/consulted incident id (INC-002) as evidence -> consult-only leak.
        pm = _good_postmortem().model_copy(
            update={"claims": [Claim(statement="x", evidence_refs=["INC-002"])]}
        )
        consulted = [ConsultedIncident(incident_id="INC-002", similarity_score=0.9, note="x")]
        report = VerificationReport(incident_id="INC-001", claim_reports=[], verification_score=1.0)
        s = {
            "incident_id": "INC-001",
            "postmortem": pm,
            "verification": report,
            "consulted": consulted,
        }
        res = await human_gate_node(s, llm=fake, tracer=tracer, conn=conn, collection=None)
        assert res["human_decision"] == "rejected"

    run(main())
