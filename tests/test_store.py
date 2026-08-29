"""Phase 2 — store + deterministic verifier tests.

The verifier is a pure set-check against the evidence table: a claim is backed iff
every ref is a real supplied id, at least one ref is present, and no recalled
incident is cited as evidence.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import Claim, Postmortem, VerificationReport  # noqa: E402
from store import (  # noqa: E402
    connect,
    init_db,
    insert_evidence,
    insert_incident,
    insert_verification_rows,
    query_evidence,
    set_check,
    valid_evidence_ids,
    verify_postmortem,
)


@pytest.fixture
def db():
    c = connect(":memory:")
    init_db(c)
    return c


def _seed(db):
    inc = {
        "id": "INC-001",
        "window_start": "2026-08-20T14:00:00",
        "window_end": "2026-08-20T15:00:00",
        "description": "checkout failures",
    }
    insert_incident(db, inc)
    insert_evidence(db, "INC-001", [
        {"id": "E1", "ts": "2026-08-20T14:02:00", "source": "deploys",
         "source_url": "u", "content": "timeout 2000->200"},
        {"id": "E2", "ts": "2026-08-20T14:03:30", "source": "metrics",
         "source_url": "u", "content": "latency up"},
        {"id": "E5", "ts": "2026-08-20T14:25:00", "source": "metrics",
         "source_url": "u", "content": "db cpu spike"},
    ])


def test_init_db_creates_tables(db):
    cur = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name IN ('incident','evidence','postmortem','verification','evaluation_run')"
    )
    names = {r["name"] for r in cur.fetchall()}
    assert {"incident", "evidence", "postmortem", "verification", "evaluation_run"} <= names


def test_query_evidence_ordering_and_window(db):
    _seed(db)
    rows = query_evidence(db, "INC-001")
    assert [r["id"] for r in rows] == ["E1", "E2", "E5"]  # chronological (ISO text)
    windowed = query_evidence(db, "INC-001", start="2026-08-20T14:03:00",
                              end="2026-08-20T14:26:00")
    assert [r["id"] for r in windowed] == ["E2", "E5"]


def test_valid_evidence_ids(db):
    _seed(db)
    assert valid_evidence_ids(db, "INC-001") == {"E1", "E2", "E5"}


def test_set_check_backed_true(db):
    _seed(db)
    claim = Claim(statement="timeout dropped", evidence_refs=["E1"])
    backed, missing = set_check(db, "INC-001", claim)
    assert backed is True
    assert missing == []


def test_set_check_hallucinated_ref_fails(db):
    _seed(db)
    claim = Claim(statement="made up", evidence_refs=["E99"])
    backed, missing = set_check(db, "INC-001", claim)
    assert backed is False
    assert missing == ["E99"]


def test_set_check_recalled_incident_cited_fails(db):
    _seed(db)
    # Even though E1 is a valid id, citing a recalled incident is forbidden.
    claim = Claim(statement="per prior incident", evidence_refs=["E1"],
                  from_recalled_incident="INC-014")
    backed, missing = set_check(db, "INC-001", claim)
    assert backed is False
    assert missing == []  # ref is valid; failure is the recalled-incident rule


def test_set_check_empty_refs_fails(db):
    _seed(db)
    claim = Claim(statement="no citation", evidence_refs=[])
    backed, _ = set_check(db, "INC-001", claim)
    assert backed is False


def test_verify_postmortem_score(db):
    _seed(db)
    pm = Postmortem(
        incident_id="INC-001", summary="s", impact="i", root_cause="config_timeout_drop",
        timeline=[], action_items=["a"],
        claims=[
            Claim(statement="timeout dropped", evidence_refs=["E1"]),
            Claim(statement="hallucinated", evidence_refs=["E99"]),
        ],
        consulted_incidents=[],
    )
    report = verify_postmortem(db, "INC-001", pm)
    assert isinstance(report, VerificationReport)
    assert report.verification_score == 0.5
    assert report.claim_reports[0].backed is True
    assert report.claim_reports[1].backed is False
    assert report.claim_reports[1].missing_evidence == ["E99"]


def test_insert_verification_rows_persists(db):
    _seed(db)
    pm = Postmortem(
        incident_id="INC-001", summary="s", impact="i", root_cause="r",
        timeline=[], action_items=["a"],
        claims=[Claim(statement="x", evidence_refs=["E1"])], consulted_incidents=[],
    )
    report = verify_postmortem(db, "INC-001", pm)
    insert_verification_rows(db, "INC-001", report)
    rows = db.execute(
        "SELECT claim_index, backed, missing_evidence FROM verification WHERE incident_id='INC-001'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["backed"] == 1


def test_on_delete_cascade_removes_evidence(db):
    _seed(db)
    db.execute("DELETE FROM incident WHERE id='INC-001'")
    db.commit()
    assert query_evidence(db, "INC-001") == []


def test_e2e_generate_incidents_load_into_store(db):
    """Full Phase 1->2 wiring: synthetic incidents load and verify cleanly."""
    from generate_incidents import generate_incidents

    incidents = generate_incidents()
    assert len(incidents) >= 10
    for inc in incidents:
        assert {"id", "window_start", "window_end", "true_root_cause",
                "red_herring", "evidence"} <= set(inc)
        insert_incident(db, inc)
        insert_evidence(db, inc["id"], inc["evidence"])
        ids = valid_evidence_ids(db, inc["id"])
        assert ids == {e["id"] for e in inc["evidence"]}
        assert all(eid.startswith("E") for eid in ids)

    showcase = next(i for i in incidents if i["id"] == "INC-001")
    assert showcase["true_root_cause"] == "config_timeout_drop"
    assert showcase["red_herring"] == "db_cpu_spike"
    assert showcase["red_herring"] not in showcase["true_root_cause"]
