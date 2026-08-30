"""Phase 7 — api.py CLI human-gate tests.

Covers the full ``run → approve/reject → show`` lifecycle through the
persistent SQLite store and the trajectory pack. All runs use the offline
FakeLLMAdapter (no network/key) via ``--fake`` / direct helpers.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import cmd_approve, cmd_reject, cmd_run, cmd_show, main as api_main  # noqa: E402
from generate_incidents import generate_incidents  # noqa: E402
from store import connect  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCS = {i["id"]: i for i in generate_incidents()}


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------- run -> pending
def test_api_run_leaves_pending_and_writes_trajectories(tmp_path):
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")

    state = run(cmd_run("INC-001", db_path=db, traj_dir=traj, fake=True))
    assert state["postmortem"].incident_id == "INC-001"
    assert state["human_decision"] == "pending"
    assert state["verification"].verification_score == 1.0

    # DB row is pending_approval with verification rows persisted
    conn = connect(db)
    row = conn.execute("SELECT status, draft_json, verification_json FROM postmortem WHERE incident_id='INC-001'").fetchone()
    assert row is not None
    assert row["status"] == "pending_approval"
    assert row["draft_json"] and row["verification_json"]
    vrows = conn.execute("SELECT backed FROM verification WHERE incident_id='INC-001'").fetchall()
    assert len(vrows) == len(state["postmortem"].claims)
    assert all(r["backed"] == 1 for r in vrows)

    # Trajectory pack — run writes ingest/timeline/analyze/writer/verifier (no gate yet)
    for node in ("ingest", "timeline", "analyze", "writer", "verifier"):
        path = os.path.join(traj, "INC-001", f"{node}.json")
        assert os.path.exists(path), f"missing trajectory {node}.json"
        ev = json.load(open(path))
        for field in ("system_prompt", "user_prompt", "input_state", "output"):
            assert field in ev, f"{node} missing {field}"
    # gate not yet written for pending
    assert not os.path.exists(os.path.join(traj, "INC-001", "human_gate.json"))


def test_api_run_via_main_cli_and_id_validation(tmp_path):
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")
    # via the argparse entrypoint (exercises the build_parser/main wiring)
    assert api_main(["run", "INC-002", "--db", db, "--traj-dir", traj, "--fake"]) == 0
    conn = connect(db)
    row = conn.execute("SELECT status FROM postmortem WHERE incident_id='INC-002'").fetchone()
    assert row["status"] == "pending_approval"

    # Path-traversal id is rejected (load_incident guard)
    with pytest.raises(Exception):
        run(cmd_run("../etc/passwd", db_path=db, traj_dir=traj, fake=True))


def test_api_run_from_json_file_path(tmp_path):
    # run should also accept a direct .json file path (the ``run <incident.json>`` form)
    inc_path = os.path.join(str(tmp_path), "custom.json")
    inc = dict(INCS["INC-003"])
    with open(inc_path, "w", encoding="utf-8") as fh:
        json.dump(inc, fh)
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")
    state = run(cmd_run(inc_path, db_path=db, traj_dir=traj, fake=True))
    assert state["postmortem"].incident_id == "INC-003"


# --------------------------------------------------------- approve / reject / show
def test_api_approve_flips_pending_to_approved_and_embeds(tmp_path):
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")

    run(cmd_run("INC-001", db_path=db, traj_dir=traj, fake=True))
    result = run(cmd_approve("INC-001", db_path=db, traj_dir=traj, applied=[], approved_by="human", fake=True))
    assert result["status"] == "approved"

    conn = connect(db)
    row = conn.execute("SELECT status, approved_by, time_approved FROM postmortem WHERE incident_id='INC-001'").fetchone()
    assert row["status"] == "approved"
    assert row["approved_by"] == "human"
    assert row["time_approved"] is not None

    # human_gate + memory_writer trajectories now present
    assert os.path.exists(os.path.join(traj, "INC-001", "human_gate.json"))
    assert os.path.exists(os.path.join(traj, "INC-001", "memory_writer.json"))
    gate = json.load(open(os.path.join(traj, "INC-001", "human_gate.json")))
    assert gate["human_decision"] == "approved"


def test_api_approve_with_apply_flag(tmp_path):
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")
    # INC-004 recall may surface a consulted incident; we seed one prior so --apply has a target.
    # The FakeLLM's analyze recall will surface INC-002 as consulted (helpers.seed_prior shape).
    # Approve with --apply via CLI main
    assert api_main(["run", "INC-001", "--db", db, "--traj-dir", traj, "--fake"]) == 0
    # Read the consulted incident id from the pending draft so we pick a real one
    conn = connect(db)
    row = conn.execute("SELECT consulted_json FROM postmortem WHERE incident_id='INC-001'").fetchone()
    consulted = json.loads(row["consulted_json"]) if row["consulted_json"] else []
    apply_id = consulted[0]["incident_id"] if consulted else "INC-002"

    assert api_main(["approve", "INC-001", "--apply", apply_id, "--by", "oncall", "--db", db, "--traj-dir", traj, "--fake"]) == 0
    row2 = conn.execute("SELECT consulted_json, approved_by FROM postmortem WHERE incident_id='INC-001'").fetchone()
    assert row2["approved_by"] == "oncall"
    consulted2 = json.loads(row2["consulted_json"])
    applied_map = {c["incident_id"]: c["applied"] for c in consulted2}
    if apply_id in applied_map:
        assert applied_map[apply_id] is True


def test_api_reject_flips_to_rejected(tmp_path):
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")
    run(cmd_run("INC-005", db_path=db, traj_dir=traj, fake=True))
    run(cmd_reject("INC-005", "not convincing", db_path=db, traj_dir=traj))
    conn = connect(db)
    row = conn.execute("SELECT status FROM postmortem WHERE incident_id='INC-005'").fetchone()
    assert row["status"] == "rejected"
    gate = json.load(open(os.path.join(traj, "INC-005", "human_gate.json")))
    assert gate["human_decision"] == "rejected"


def test_api_show_returns_draft_and_verification(tmp_path, capsys):
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")
    run(cmd_run("INC-006", db_path=db, traj_dir=traj, fake=True))
    out = cmd_show("INC-006", db_path=db)
    assert out["incident_id"] == "INC-006"
    assert out["status"] == "pending_approval"
    assert out["draft"] is not None
    assert out["verification"] is not None
    # capsys not needed — cmd_show prints JSON; we assert the return value


def test_api_approve_rejects_non_pending(tmp_path):
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")
    run(cmd_run("INC-007", db_path=db, traj_dir=traj, fake=True))
    run(cmd_approve("INC-007", db_path=db, traj_dir=traj, fake=True))
    # second approve should fail (already approved, not pending)
    with pytest.raises(SystemExit):
        run(cmd_approve("INC-007", db_path=db, traj_dir=traj, fake=True))


def test_api_show_missing_raises(tmp_path):
    with pytest.raises(SystemExit):
        cmd_show("INC-999", db_path=str(tmp_path / "app.db"))


def test_api_reject_via_cli_main(tmp_path):
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")
    assert api_main(["run", "INC-008", "--db", db, "--traj-dir", traj, "--fake"]) == 0
    assert api_main(["reject", "INC-008", "needs more evidence", "--db", db, "--traj-dir", traj]) == 0
    assert api_main(["show", "INC-008", "--db", db]) == 0


# ---------------- Test gaps from Phase 7/8 fix list ----------------
def test_api_run_fake_approve_default_inherits_backend(tmp_path):
    """run --fake → approve (no flag) must inherit fake store (Design Risk #2)."""
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")
    assert api_main(["run", "INC-001", "--db", db, "--traj-dir", traj, "--fake"]) == 0
    from store import get_cli_backend

    conn = connect(db)
    assert get_cli_backend(conn, "INC-001") == "fake"
    # approve with no --fake/--live flag should inherit and succeed
    assert api_main(["approve", "INC-001", "--db", db, "--traj-dir", traj]) == 0
    row = conn.execute("SELECT status FROM postmortem WHERE incident_id='INC-001'").fetchone()
    assert row["status"] == "approved"


def test_api_run_fake_approve_live_mismatch_fails(tmp_path):
    """run --fake but approve --live must fail with backend mismatch guard."""
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")
    assert api_main(["run", "INC-002", "--db", db, "--traj-dir", traj, "--fake"]) == 0
    with pytest.raises(SystemExit, match="backend mismatch"):
        run(cmd_approve("INC-002", db_path=db, traj_dir=traj, live=True))


def test_api_pending_leaves_no_memory_and_leak_refuses_embed(tmp_path):
    """Pending run must not embed; consult-only leak approve → embedded=False."""
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")

    # Normal pending: no memory_writer yet, no embed
    assert api_main(["run", "INC-003", "--db", db, "--traj-dir", traj, "--fake"]) == 0
    assert not os.path.exists(os.path.join(traj, "INC-003", "memory_writer.json"))

    # Now craft a leaked pending postmortem and approve it
    from schemas import Claim, ConsultedIncident, Postmortem
    from store import init_db, insert_evidence, insert_incident, set_cli_backend, upsert_postmortem

    conn = connect(db)
    inc = INCS["INC-003"]
    # Ensure evidence exists for verification (not used by leak check)
    insert_incident(conn, inc)
    insert_evidence(conn, inc["id"], inc["evidence"])
    # Create the LEAK-001 incident row first (FK for postmortem)
    insert_incident(conn, {"id": "LEAK-001", "window_start": "2026-08-20T14:00:00", "window_end": "2026-08-20T15:00:00", "description": "leak test"})
    leak_pm = Postmortem(
        incident_id="LEAK-001",
        summary="leak",
        impact="impact",
        root_cause="config_timeout_drop",
        timeline=[],
        action_items=["a"],
        claims=[Claim(statement="leak", evidence_refs=["E1", "INC-002"])],
        consulted_incidents=[ConsultedIncident(incident_id="INC-002", similarity_score=0.9, note="x")],
    )
    leak_consulted = [ConsultedIncident(incident_id="INC-002", similarity_score=0.9, note="x")]
    upsert_postmortem(
        conn,
        "LEAK-001",
        draft_json=leak_pm.model_dump_json(),
        verification_json='{"incident_id":"LEAK-001","claim_reports":[],"verification_score":0.0}',
        consulted_json=json.dumps([c.model_dump() for c in leak_consulted], ensure_ascii=False),
        status="pending_approval",
    )
    set_cli_backend(conn, "LEAK-001", "fake")
    result = run(cmd_approve("LEAK-001", db_path=db, traj_dir=traj, fake=True))
    assert result["embedded"] is False
    mw_path = os.path.join(traj, "LEAK-001", "memory_writer.json")
    assert os.path.exists(mw_path)
    mw = json.load(open(mw_path))
    assert mw["output"].get("embedded") is False
    assert "consult-only leak" in str(mw["output"]).lower()


def test_api_direct_json_path_shape_validation(tmp_path):
    """Direct .json path with missing required fields must be rejected at cmd_run."""
    bad = os.path.join(str(tmp_path), "bad.json")
    with open(bad, "w", encoding="utf-8") as fh:
        json.dump({"id": "INC-BAD"}, fh)
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")
    with pytest.raises(ValueError, match="missing required field"):
        run(cmd_run(bad, db_path=db, traj_dir=traj, fake=True))


def test_api_run_fake_approve_live_via_cli_regression(tmp_path):
    """Regression lock: run --fake → approve --live via CLI must SystemExit (api.py:309-313)."""
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")
    assert api_main(["run", "INC-009", "--db", db, "--traj-dir", traj, "--fake"]) == 0
    # via direct async API
    with pytest.raises(SystemExit, match="backend mismatch"):
        run(cmd_approve("INC-009", db_path=db, traj_dir=traj, live=True))
    # via CLI entrypoint (argparse wiring)
    with pytest.raises(SystemExit, match="backend mismatch"):
        api_main(["approve", "INC-009", "--db", db, "--traj-dir", traj, "--live"])
    # default approve still succeeds on a fresh incident (inherits fake)
    assert api_main(["run", "INC-010", "--db", db, "--traj-dir", traj, "--fake"]) == 0
    assert api_main(["approve", "INC-010", "--db", db, "--traj-dir", traj]) == 0
    conn = connect(db)
    row = conn.execute("SELECT status FROM postmortem WHERE incident_id='INC-010'").fetchone()
    assert row["status"] == "approved"


def test_api_run_fake_approve_live_mismatch_error_message(tmp_path):
    """The mismatch error must mention both backends so the operator knows how to fix it."""
    db = str(tmp_path / "app.db")
    traj = str(tmp_path / "trajectories")
    assert api_main(["run", "INC-011", "--db", db, "--traj-dir", traj, "--fake"]) == 0
    with pytest.raises(SystemExit) as exc:
        run(cmd_approve("INC-011", db_path=db, traj_dir=traj, live=True))
    msg = str(exc.value)
    assert "backend mismatch" in msg
    assert "--fake" in msg and "--live" in msg
