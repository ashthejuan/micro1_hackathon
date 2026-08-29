"""Phase 4 — rag.py + memory.py (consult-only) tests.

Covers: metadata scalar-only (comma-joined keywords, "" not null); recall returns
scores in [0,1]; consult-only enforcement — injecting a recalled id into
`evidence_refs` is rejected.

These run against BOTH the in-memory cosine backend AND the Chroma backend, because
Chroma is the shipped `backend="auto"` (when `chromadb` is installed) and is exactly
the path that was previously untested. The in-memory fallback alone is not enough:
the metadata-scalar contract, duplicate-id enforcement, and distance→score conversion
are Chroma-specific behaviours that the memory backend does not fully replicate.

Chroma is required in `requirements.txt`, so CI must exercise it. If `chromadb` is
not installed the `chroma` parametrization is skipped (not silently dropped), and a
dedicated test fails loudly so the gap cannot hide behind a green memory-only run.
"""

import asyncio
import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import _HAS_CHROMA, create_memory_collection  # noqa: E402
from memory import (  # noqa: E402
    assert_consult_only,
    embed_postmortem,
    get_collection,
    recall_incidents,
    validate_consult_only,
)
from schemas import (  # noqa: E402
    Claim,
    ConsultedIncident,
    Postmortem,
    TimelineEvent,
)

# Backends that must be covered. Chroma is the shipped default; the in-memory store
# is only the offline fallback. Both are exercised whenever chromadb is available.
BACKENDS = ["memory"]
if _HAS_CHROMA:
    BACKENDS.append("chroma")


def pytest_generate_tests(metafunc):
    """Parametrize any test taking a `backend` arg over the available backends."""
    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", BACKENDS)


def _symptom_embed(texts):
    """Deterministic local embed so recall is reproducible offline."""
    from llm_adapter import LLMAdapter

    return LLMAdapter()._hash_embed(texts)


def _collection(backend="memory"):
    # Unique name per call so the (process-global, ephemeral) Chroma client does not
    # leak documents between tests and trip the duplicate-id guard.
    name = f"incident_memory_{uuid.uuid4().hex[:12]}"
    return create_memory_collection(
        name=name, embed_fn=_symptom_embed, backend=backend
    )


def _approved_postmortem(incident_id="INC-009", root_cause="feature_flag_wrong",
                         action_items=None):
    return Postmortem(
        incident_id=incident_id,
        summary="Feature rollout caused errors for a subset of users.",
        impact="Errors for 100% of users instead of 10%.",
        root_cause=root_cause,
        timeline=[
            TimelineEvent(ts="2026-08-28T10:01:00",
                          description="flag enabled for 100%",
                          evidence_refs=["E1"]),
        ],
        action_items=action_items if action_items is not None
        else ["Gate flag rollout behind percentage."],
        claims=[Claim(statement="flag rolled to everyone",
                      evidence_refs=["E1"])],
        consulted_incidents=[],
    )


# --------------------------------------------- the shipped backend must be covered
def test_chroma_is_available_in_this_environment():
    """Guard: Chroma is a shipped dependency (requirements.txt), so its code path
    must not be silently skipped. If this fails in CI, install chromadb.
    """
    assert _HAS_CHROMA, (
        "chromadb is not importable — the Chroma backend (the shipped default) is "
        "NOT being tested. Install chromadb>=0.5 (see requirements.txt)."
    )


# ---------------------------------------------------------------- metadata shape
def test_embed_postmortem_metadata_is_scalar_only():
    pm = _approved_postmortem()
    doc = embed_postmortem(pm, time_approved="2026-08-29T00:00:00", verification_score=0.5)
    meta = doc["metadata"]
    assert set(meta) == {
        "incident_id", "root_cause_label", "time_approved",
        "action_item_count", "symptom_keywords", "verification_score",
    }
    # Every value must be a scalar (str|int|float|bool), never list/dict/None.
    for k, v in meta.items():
        assert isinstance(v, (str, int, float, bool)), f"{k} is not scalar: {v!r}"
    # symptom_keywords is a comma-joined string and never null/empty-string-only.
    assert isinstance(meta["symptom_keywords"], str)
    assert meta["symptom_keywords"] == "feature,flag,wrong"
    assert meta["incident_id"] == "INC-009"
    assert meta["root_cause_label"] == "feature_flag_wrong"
    assert meta["action_item_count"] == 1
    assert meta["verification_score"] == 0.5
    assert meta["time_approved"] == "2026-08-29T00:00:00"


def test_embed_postmortem_action_item_count_is_faithful():
    pm = _approved_postmortem(action_items=["Fix the flag.", "Add a guardrail.", "Alert on rollout."])
    doc = embed_postmortem(pm, verification_score=1.0)
    assert doc["metadata"]["action_item_count"] == 3


def test_embed_postmortem_omits_score_when_unknown():
    pm = _approved_postmortem()
    doc = embed_postmortem(pm, time_approved="2026-08-29T00:00:00")
    assert "verification_score" not in doc["metadata"]


def test_embed_postmortem_matches_fixture_shape():
    """Live-embedded doc must share shape with pre-seeded memory_seed fixtures."""
    from generate_incidents import memory_seed

    fixture = memory_seed()[0]
    pm = _approved_postmortem(incident_id=fixture["incident_id"],
                              root_cause=fixture["metadata"]["root_cause_label"])
    doc = embed_postmortem(pm, time_approved=fixture["metadata"]["time_approved"],
                           verification_score=1.0)
    assert set(doc) == {"incident_id", "document", "metadata"}
    # Base fixture keys must all be present; live docs additionally carry quality.
    assert set(fixture["metadata"]).issubset(set(doc["metadata"]))
    assert "verification_score" in doc["metadata"]


# --------------------- Chroma-level boundary: non-scalar metadata is rejected -----
@pytest.mark.parametrize("bad_value", [
    ["a", "b"],            # list
    {"k": "v"},            # dict
    None,                  # null
    ("a", "b"),            # tuple
])
def test_add_rejects_nonscalar_metadata(backend, bad_value):
    """Both backends must reject non-scalar metadata at the boundary.

    Chroma rejects this itself, but the in-memory store does not — so the validation
    lives in `rag._validate_scalar_metadata` and is the real contract. This was
    previously only exercised by the memory backend.
    """
    coll = _collection(backend)
    with pytest.raises(TypeError):
        asyncio.run(coll.add(
            id="INC-BAD",
            document="anything at all",
            metadata={"incident_id": "INC-BAD", "symptom_keywords": bad_value},
        ))


# ----------------------------- Chroma-level boundary: duplicate ids are rejected --
def test_add_rejects_duplicate_id(backend):
    """Duplicate ids must raise on BOTH backends.

    The memory store appended silently while Chroma rejected — that divergence is the
    bug this guards against. Previously only the memory branch was tested.
    """
    coll = _collection(backend)
    meta = {"incident_id": "INC-DUP", "root_cause_label": "rc",
            "time_approved": "2026-08-20T00:00:00", "action_item_count": 1,
            "symptom_keywords": "a,b"}
    asyncio.run(coll.add(id="INC-DUP", document="first doc", metadata=meta))
    with pytest.raises(ValueError):
        asyncio.run(coll.add(id="INC-DUP", document="second doc", metadata=meta))


# ------------------------------------------------------------------- recall API
def test_recall_returns_scores_in_unit_interval(backend):
    coll = _collection(backend)
    # Seed two past incidents with distinct token signatures.
    asyncio.run(coll.add(
        id="INC-001",
        document="Checkout failures payment timeout dropped config 2000 200",
        metadata={"incident_id": "INC-001", "root_cause_label": "config_timeout_drop",
                  "time_approved": "2026-08-20T15:00:00", "action_item_count": 1,
                  "symptom_keywords": "config,timeout,drop,db,cpu,spike"},
    ))
    asyncio.run(coll.add(
        id="INC-005",
        document="Database connections exhausted pool max raised 200 6000",
        metadata={"incident_id": "INC-005", "root_cause_label": "connection_pool_exhaustion",
                  "time_approved": "2026-08-24T14:00:00", "action_item_count": 1,
                  "symptom_keywords": "connection,pool,exhaustion,cpu,throttling"},
    ))

    results = asyncio.run(coll.recall("payment timeout config dropped", n=5))
    assert len(results) == 2
    assert all(isinstance(r, ConsultedIncident) for r in results)
    # Chroma returns cosine *distance*; the wrapper converts to 1 - distance and
    # clamps to [0, 1]. This asserts the conversion/clamp actually happens.
    assert all(0.0 <= r.similarity_score <= 1.0 for r in results)
    assert all(isinstance(r.similarity_score, float) for r in results)
    # Most similar first.
    assert results[0].similarity_score >= results[1].similarity_score
    # The payment-timeout query should rank INC-001 above INC-005.
    assert results[0].incident_id == "INC-001"
    assert results[0].applied is False  # consult-only default


def test_recall_distance_conversion_is_one_for_exact_match(backend):
    """A query identical to a stored doc must score ~1.0.

    This directly proves the distance→similarity conversion (1 - distance) works on
    the Chroma backend rather than leaking raw distances or negatives.
    """
    doc_text = "redis cache eviction maxmemory lowered hit rate collapse"
    coll = _collection(backend)
    asyncio.run(coll.add(
        id="INC-002",
        document=doc_text,
        metadata={"incident_id": "INC-002", "root_cause_label": "redis_cache_eviction",
                  "time_approved": "2026-08-21T10:00:00", "action_item_count": 1,
                  "symptom_keywords": "redis,cache,eviction,network,latency"},
    ))
    results = asyncio.run(coll.recall(doc_text, n=1))
    assert len(results) == 1
    assert results[0].incident_id == "INC-002"
    # Exact match => cosine distance 0 => similarity 1.0 (within float tolerance).
    assert results[0].similarity_score == pytest.approx(1.0, abs=1e-4)


def test_recall_empty_collection_returns_nothing(backend):
    coll = _collection(backend)
    assert asyncio.run(coll.recall("anything", n=5)) == []


# ------------------------------------------------------------- consult-only ban
def test_consult_only_rejects_recalled_id_in_evidence_refs():
    recalled = ["INC-014"]
    pm = Postmortem(
        incident_id="INC-010",
        summary="Upstream gRPC timeouts.",
        impact="17% deadline_exceeded.",
        root_cause="grpc_upstream_timeout",
        timeline=[TimelineEvent(ts="2026-08-29T12:01:00",
                                     description="upstream upgraded", evidence_refs=["E1"])],
        action_items=["Loosen keepalive."],
        claims=[Claim(statement="prior incident INC-014 shows same pattern",
                      evidence_refs=["E1", "INC-014"])],  # ILLEGAL: recalled id cited
        consulted_incidents=[ConsultedIncident(incident_id="INC-014",
                                               similarity_score=0.9, note="similar")],
    )
    violations = validate_consult_only(pm, recalled)
    assert len(violations) == 1
    assert violations[0]["claim_index"] == 0
    assert violations[0]["leaked_refs"] == ["INC-014"]
    with pytest.raises(ValueError):
        assert_consult_only(pm, recalled)


def test_consult_only_passes_when_recalled_id_only_in_consulted():
    recalled = ["INC-014"]
    pm = Postmortem(
        incident_id="INC-010",
        summary="Upstream gRPC timeouts.",
        impact="17% deadline_exceeded.",
        root_cause="grpc_upstream_timeout",
        timeline=[TimelineEvent(ts="2026-08-29T12:01:00",
                                     description="upstream upgraded", evidence_refs=["E1"])],
        action_items=["Loosen keepalive."],
        claims=[Claim(statement="upstream keepalive killing streams",
                      evidence_refs=["E1"])],  # only real evidence
        consulted_incidents=[ConsultedIncident(incident_id="INC-014",
                                               similarity_score=0.9,
                                               applied=True,
                                               note="consulted only")],
    )
    assert validate_consult_only(pm, recalled) == []
    assert_consult_only(pm, recalled)  # does not raise


def test_recall_incidents_returns_consulted_only_objects(backend):
    coll = _collection(backend)
    asyncio.run(coll.add(
        id="INC-002",
        document="API latency redis cache eviction maxmemory lowered",
        metadata={"incident_id": "INC-002", "root_cause_label": "redis_cache_eviction",
                  "time_approved": "2026-08-21T10:00:00", "action_item_count": 1,
                  "symptom_keywords": "redis,cache,eviction,network,latency"},
    ))
    results = asyncio.run(
        recall_incidents("redis eviction cache hit rate collapse", coll, n=3)
    )
    assert len(results) == 1
    assert results[0].incident_id == "INC-002"
    assert results[0].applied is False
