"""Phase 6 — graph.py LangGraph orchestration integration tests.

Drives the full 7-node `StateGraph` from `graph.py` two ways:

* `test_graph_fake_adapter_e2e` — always runs offline via `FakeLLMAdapter`
  (the same reusable fake moved to `tests/helpers.py`).
* `test_graph_replay_e2e` — skippable; runs the *real* `LLMAdapter` in offline
  replay mode once `fixtures/llm_cache.jsonl` is recorded (Phase 8), else skips.

Both assert the end-to-end invariants: `Postmortem` + `VerificationReport`
produced, `human_decision == "approved"`, memory grew by one embedded prior, the
postmortem row transitioned `running -> approved`, every trajectory file +
`manifest.json` exists, and no recalled incident id leaked into `claims[].evidence_refs`.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import (  # noqa: E402
    FakeLLMAdapter,
    INCS,
    _good_postmortem,
    make_collection,
    seed_prior,
)
from schemas import (  # noqa: E402
    Claim,
    Postmortem,
    VerificationReport,
)
from store import connect, init_db  # noqa: E402
from tracer import Tracer  # noqa: E402
from graph import arun_incident, load_incident, run_cli  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "fixtures", "llm_cache.jsonl")
LLM_CACHE_EXISTS = os.path.exists(FIXTURE)

AGENTS = [
    "ingest",
    "timeline",
    "analyze",
    "writer",
    "verifier",
    "human_gate",
    "memory_writer",
]


def run(coro):
    return asyncio.run(coro)


def _assert_invariants(result, conn, coll, traj_dir, *, expect_approved=True):
    # Pipeline outputs complete.
    assert isinstance(result["postmortem"], Postmortem)
    assert isinstance(result["verification"], VerificationReport)
    pm = result["postmortem"]
    assert pm.summary and pm.impact and pm.root_cause
    assert pm.timeline and pm.claims and pm.action_items

    if expect_approved:
        assert result["human_decision"] == "approved"
    else:
        assert result["human_decision"] == "rejected"

    # Memory grew by exactly one embedded approved postmortem (1 seeded + 1 new).
    assert coll.count() == 2

    # Postmortem row transitioned running -> approved (or rejected).
    row = conn.execute(
        "SELECT status FROM postmortem WHERE incident_id=?", (pm.incident_id,)
    ).fetchone()
    assert row is not None
    assert row["status"] == ("approved" if expect_approved else "rejected")

    # Trajectory pack: every agent file present + manifest indexes them.
    for a in AGENTS:
        path = os.path.join(traj_dir, pm.incident_id, f"{a}.json")
        assert os.path.exists(path), f"missing trajectory {a}.json"
        with open(path) as fh:
            ev = __import__("json").load(fh)
        for field in ("system_prompt", "user_prompt", "input_state", "tool_calls", "output"):
            assert field in ev, f"{a} missing {field}"
    manifest = __import__("json").load(
        open(os.path.join(traj_dir, pm.incident_id, "manifest.json"))
    )
    assert set(manifest["agents"]) == set(AGENTS)

    # Consult-only: no recalled id leaked into any claim's evidence_refs.
    recalled = {c.incident_id for c in result.get("consulted", [])}
    for claim in pm.claims:
        assert not (set(claim.evidence_refs) & recalled), (
            f"recalled id leaked into claim: {claim.evidence_refs}"
        )
    return pm


def test_graph_fake_adapter_e2e(tmp_path):
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        coll = make_collection()
        await seed_prior(coll)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        result = await arun_incident(
            INCS["INC-001"], llm=fake, tracer=tracer, conn=conn, collection=coll
        )
        _assert_invariants(result, conn, coll, str(tmp_path), expect_approved=True)

    run(main())


def test_graph_fanout_runs_both_branches(tmp_path):
    """The `Send` fan-out must execute `timeline` and `analyze` independently."""
    async def main():
        fake = FakeLLMAdapter()
        conn = connect(":memory:")
        coll = make_collection()
        await seed_prior(coll)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        result = await arun_incident(
            INCS["INC-001"], llm=fake, tracer=tracer, conn=conn, collection=coll
        )
        assert result["timeline_events"], "timeline branch did not populate"
        assert result["candidates"], "analyze branch did not populate"
        assert result["consulted"], "analyze recall branch did not populate"

    run(main())


@pytest.mark.skipif(
    not LLM_CACHE_EXISTS,
    reason="fixtures/llm_cache.jsonl not recorded yet (Phase 8 --live)",
)
def test_graph_replay_e2e(tmp_path):
    async def main():
        from llm_adapter import LLMAdapter
        from rag import create_memory_collection, _default_embed

        llm = LLMAdapter(live=False)  # replay from cache; embeddings use fallback
        conn = connect(":memory:")
        init_db(conn)
        coll = create_memory_collection(
            name="incident_memory", embed_fn=_default_embed, backend="auto"
        )
        await seed_prior(coll)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        result = await arun_incident(
            INCS["INC-001"], llm=llm, tracer=tracer, conn=conn, collection=coll
        )
        _assert_invariants(result, conn, coll, str(tmp_path), expect_approved=True)

    run(main())


def test_graph_rejects_when_verification_fails(tmp_path):
    """When the writer emits an unciteable postmortem, verification < 1.0 and the
    gate rejects it (status `rejected`)."""

    class BadFakeLLMAdapter(FakeLLMAdapter):
        async def chat(self, system, messages, model=None, params=None, schema=None):
            kind = self._schema_name(schema)
            if kind == "postmortem":
                return self._always_bad()
            return await super().chat(system, messages, model, params, schema)

        def _always_bad(self):
            pm = _good_postmortem_for_test()
            return pm.model_copy(
                update={
                    "claims": [
                        Claim(statement="cited a hallucinated id", evidence_refs=["E99"])
                    ]
                }
            )

    from helpers import _good_postmortem as _good_postmortem_for_test
    from schemas import Claim

    async def main():
        fake = BadFakeLLMAdapter()
        conn = connect(":memory:")
        coll = make_collection()
        await seed_prior(coll)
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        result = await arun_incident(
            INCS["INC-001"], llm=fake, tracer=tracer, conn=conn, collection=coll
        )
        _assert_invariants(result, conn, coll, str(tmp_path), expect_approved=False)

    run(main())


# ------------------------------------------------------- load_incident / run_cli
def test_load_incident_returns_valid_payload():
    inc = load_incident("INC-001")
    assert inc["id"] == "INC-001"
    assert inc["window_start"] and inc["window_end"]
    assert isinstance(inc["evidence"], list) and inc["evidence"]


def test_load_incident_rejects_bad_id():
    with pytest.raises(ValueError):
        load_incident("../etc/x")


def test_load_incident_rejects_missing_fields(tmp_path):
    bad = os.path.join(str(tmp_path), "INC-BAD.json")
    with open(bad, "w") as fh:
        json.dump({"id": "INC-BAD"}, fh)
    with pytest.raises(ValueError):
        load_incident("INC-BAD", incidents_dir=str(tmp_path))


def test_load_incident_rejects_id_mismatch(tmp_path):
    p = os.path.join(str(tmp_path), "INC-X.json")
    with open(p, "w") as fh:
        json.dump(
            {
                "id": "INC-Y",
                "window_start": "a",
                "window_end": "b",
                "description": "d",
                "evidence": [],
            },
            fh,
        )
    with pytest.raises(ValueError):
        load_incident("INC-X", incidents_dir=str(tmp_path))


def test_run_cli_fake_runs_end_to_end(tmp_path):
    async def main():
        result = await run_cli("INC-001", fake=True, traj_dir=str(tmp_path))
        assert result["human_decision"] == "approved"
        assert isinstance(result["postmortem"], Postmortem)
        assert isinstance(result["verification"], VerificationReport)
        # Trajectory pack was written under the requested dir.
        assert os.path.exists(os.path.join(str(tmp_path), "INC-001", "manifest.json"))

    run(main())


# --------------------------------------- consult-only leak across the compiled graph
class LeakFakeLLMAdapter(FakeLLMAdapter):
    """Writer emits a claim that cites a *recalled* prior incident id via
    `from_recalled_incident`. The verifier's `assert_consult_only` only inspects
    `evidence_refs` (so it does NOT raise here); the consult-only leak must be
    caught downstream by `memory_writer_node` refusing to embed it."""

    async def chat(self, system, messages, model=None, params=None, schema=None):
        kind = self._schema_name(schema)
        if kind == "postmortem":
            pm = _good_postmortem()
            return pm.model_copy(
                update={
                    "claims": [
                        Claim(
                            statement="motivated by recalled prior incident",
                            evidence_refs=["E1"],
                            from_recalled_incident="INC-002",
                        )
                    ]
                }
            )
        return await super().chat(system, messages, model, params, schema)


def test_graph_memory_writer_refuses_consult_only_leak(tmp_path):
    async def main():
        fake = LeakFakeLLMAdapter()
        conn = connect(":memory:")
        coll = make_collection()
        before = coll.count()  # seeded prior only
        await seed_prior(coll)
        assert coll.count() == before + 1
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        result = await arun_incident(
            INCS["INC-001"], llm=fake, tracer=tracer, conn=conn, collection=coll
        )
        # Gate rejects the failed-verification (leaked claim is not backed).
        assert result["human_decision"] == "rejected"
        # memory_writer must NOT embed the leak: count stays at the seeded prior.
        assert coll.count() == before + 1
        mw = json.load(open(os.path.join(str(tmp_path), "INC-001", "memory_writer.json")))
        assert mw["output"].get("embedded") is False
        assert "consult-only leak" in mw["output"].get("reason", "")

    run(main())


class EvidenceRefLeakFakeLLMAdapter(FakeLLMAdapter):
    """Writer cites a *recalled* prior incident id as an evidence ref (the natural
    shape an LLM emits: citing a recalled incident as a citation). This hits
    `assert_consult_only` directly, which would have crashed the graph before the
    verifier_node guard was added. The run must survive cleanly: gate rejects,
    memory_writer refuses to embed, no exception escapes `arun_incident`."""

    async def chat(self, system, messages, model=None, params=None, schema=None):
        kind = self._schema_name(schema)
        if kind == "postmortem":
            pm = _good_postmortem()
            return pm.model_copy(
                update={
                    "claims": [
                        Claim(
                            statement="cited a recalled prior incident as a citation",
                            evidence_refs=["E1", "INC-002"],
                        )
                    ]
                }
            )
        return await super().chat(system, messages, model, params, schema)


def test_graph_survives_evidence_ref_consult_only_leak(tmp_path):
    """Regression guard for BUG 1: a leak via claims[].evidence_refs previously
    raised ValueError inside verifier_node and killed the whole batch. The graph
    must survive and reject cleanly."""
    async def main():
        fake = EvidenceRefLeakFakeLLMAdapter()
        conn = connect(":memory:")
        coll = make_collection()
        before = coll.count()
        await seed_prior(coll)
        assert coll.count() == before + 1
        tracer = Tracer("INC-001", out_dir=str(tmp_path))
        result = await arun_incident(
            INCS["INC-001"], llm=fake, tracer=tracer, conn=conn, collection=coll
        )
        # Graph survived; gate rejects the consult-only leak.
        assert result["human_decision"] == "rejected"
        # memory_writer must NOT embed the leak: count stays at the seeded prior.
        assert coll.count() == before + 1
        mw = json.load(open(os.path.join(str(tmp_path), "INC-001", "memory_writer.json")))
        assert mw["output"].get("embedded") is False
        assert "consult-only leak" in mw["output"].get("reason", "")

    run(main())
