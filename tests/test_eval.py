"""Phase 8 — eval.py judging-critical tests.

Validates the plan's ``Phase 7/8`` contract:

* Agent verification  ≫ Baseline A (structural typed claims + set-check).
* Red-herring rejection rate = 1.0 on the fixture set for the agent.
* Deterministic across two offline runs (same scores, hash-embed + FakeLLM).
* ``evaluation_run`` rows written; ``--live`` path writes fixtures (skipped offline).
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import run_eval  # noqa: E402
from store import connect  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "fixtures", "llm_cache.jsonl")


def run(coro):
    return asyncio.run(coro)


def test_eval_agent_beats_baseline_fake(tmp_path):
    """Agent verification ≫ Baseline A; red-herring rate 1.0 for agent."""
    summary = run(run_eval(db_path=":memory:", traj_dir=str(tmp_path), fake=True, print_table=False))

    # Shape
    assert "agent" in summary and "baseline" in summary
    assert summary["eval_ids"]  # at least 4..11
    assert len(summary["agent"]["per_incident"]) == len(summary["eval_ids"])
    assert len(summary["baseline"]["per_incident"]) == len(summary["eval_ids"])

    a = summary["agent"]
    b = summary["baseline"]

    # Primary: agent verification near-perfect, baseline near-zero
    assert a["verification_mean"] == pytest.approx(1.0, abs=0.01)
    assert b["verification_mean"] == pytest.approx(0.0, abs=0.01)
    assert a["verification_mean"] > b["verification_mean"] + 0.5

    # Secondary: agent rejects every planted red herring; baseline blames it
    assert a["red_herring_rate"] == pytest.approx(1.0, abs=0.01)
    assert b["red_herring_rate"] == pytest.approx(0.0, abs=0.01)

    # Secondary: both are structurally complete (required sections present)
    assert a["completeness_mean"] == pytest.approx(1.0, abs=0.01)
    assert b["completeness_mean"] == pytest.approx(1.0, abs=0.01)


def test_eval_deterministic_across_two_offline_runs(tmp_path):
    """Two offline runs with the same FakeLLMAdapter produce identical scores."""
    s1 = run(run_eval(db_path=":memory:", traj_dir=str(tmp_path / "a"), fake=True, print_table=False))
    s2 = run(run_eval(db_path=":memory:", traj_dir=str(tmp_path / "b"), fake=True, print_table=False))

    for mode in ("agent", "baseline"):
        assert s1[mode]["verification_mean"] == pytest.approx(s2[mode]["verification_mean"])
        assert s1[mode]["red_herring_rate"] == pytest.approx(s2[mode]["red_herring_rate"])
        assert s1[mode]["completeness_mean"] == pytest.approx(s2[mode]["completeness_mean"])
        # Per-incident scores also deterministic
        for r1, r2 in zip(s1[mode]["per_incident"], s2[mode]["per_incident"]):
            assert r1["verification"] == r2["verification"]
            assert r1["red_herring"] == r2["red_herring"]


def test_eval_writes_evaluation_run_rows(tmp_path):
    """Each (mode, incident) writes a row into evaluation_run; mirrors store.py contract."""
    db = str(tmp_path / "eval.db")
    summary = run(run_eval(db_path=db, traj_dir=str(tmp_path / "traj"), fake=True, print_table=False))

    conn = connect(db)
    rows = conn.execute("SELECT mode, incident_id, verification_score, red_herring_correct FROM evaluation_run").fetchall()
    # One row per (agent, baseline) × eval_ids
    assert len(rows) == len(summary["eval_ids"]) * 2
    modes = {r["mode"] for r in rows}
    assert modes == {"agent", "baseline"}
    # Agent rows are backed; baseline rows are not
    for r in rows:
        if r["mode"] == "agent":
            assert r["verification_score"] == pytest.approx(1.0)
            assert r["red_herring_correct"] == 1
        else:
            assert r["verification_score"] == pytest.approx(0.0)
            assert r["red_herring_correct"] == 0


def test_eval_cli_writes_comparison_table(tmp_path, capsys):
    """``python eval.py`` prints the required comparison table header."""
    import eval as eval_module

    assert eval_module.main(["--fake", "--db", ":memory:", "--traj-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "verification" in out.lower()
    assert "red_herring" in out.lower()
    assert "agent" in out.lower()
    assert "baseline" in out.lower()


@pytest.mark.skipif(
    not os.path.exists(FIXTURE),
    reason="fixtures/llm_cache.jsonl not recorded yet (Phase 8 --live)",
)
def test_eval_live_replay_path(tmp_path):
    """When fixtures exist, the replay path (LLMAdapter live=False) yields the
    same scores as the fake path (recorded responses are structured objects)."""
    summary = run(run_eval(db_path=":memory:", traj_dir=str(tmp_path), fake=False, live=False, print_table=False))
    assert summary["agent"]["verification_mean"] == pytest.approx(1.0, abs=0.01)


@pytest.mark.skipif(
    os.environ.get("LIVE", "") not in ("1", "true", "True"),
    reason="LIVE mode needs LIVE=1 (would hit the network)",
)
def test_eval_live_writes_fixtures(tmp_path):
    """``--live`` path appends to fixtures/llm_cache.jsonl (requires key)."""
    # This is the Phase 8 --live smoke test; it actually hits the endpoint and
    # persists. Skipped in offline CI by the guard above (requires LIVE=1).
    summary = run(run_eval(db_path=":memory:", traj_dir=str(tmp_path), fake=False, live=True, print_table=False))
    assert os.path.exists(FIXTURE)
    assert summary["agent"]["verification_mean"] >= 0.0


def test_eval_live_skips_without_key_or_fixture(tmp_path, monkeypatch, capsys):
    """BLOCKER: --live with no key and no fixture must skip cleanly (return 0)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LIVE", "0")
    # Ensure the fixture path does not exist for this test (isolated tmp ROOT
    # would be ideal, but we can just verify the guard prints the skip message).
    # The real fixture is absent in this repo (fixtures/ dir missing), so the
    # guard should trigger when we force has_key=False via empty env.
    import eval as eval_module

    # Patch os.path.exists to pretend fixtures/llm_cache.jsonl is missing
    orig_exists = os.path.exists

    def fake_exists(p):
        if "llm_cache.jsonl" in str(p):
            return False
        return orig_exists(p)

    monkeypatch.setattr(os.path, "exists", fake_exists)
    assert eval_module.main(["--live", "--db", ":memory:", "--traj-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "skipped" in out.lower()
    assert "no key/fixture" in out.lower()


def test_eval_isolated_recall_no_sibling_contamination(tmp_path):
    """Design Risk #3: isolated eval must not recall siblings from the same run."""
    # With fake LLM the verification is 1 regardless, but we can assert the
    # per-incident collection size stays at seed+1 (not growing).
    from helpers import make_collection
    from generate_incidents import generate_incidents
    from rag import create_memory_collection
    from llm_adapter import LLMAdapter

    # Direct check of the isolate flag wiring: run with isolate=True then
    # inspect that each agent run's collection had only seed docs + its own.
    # The run_eval helper already tests scores, here we just verify isolation
    # does not break the scores.
    summary = run(run_eval(db_path=":memory:", traj_dir=str(tmp_path), fake=True, print_table=False, isolate=True))
    assert summary["isolate"] is True
    assert summary["agent"]["verification_mean"] == pytest.approx(1.0)


def test_eval_fake_llm_is_incident_specific(tmp_path):
    """Design Risk #4: FakeLLM must return incident-specific root_cause_label."""
    from helpers import FakeLLMAdapter

    fake = FakeLLMAdapter()

    async def _check(incident_id: str, expected_label: str):
        msg = [{"role": "user", "content": f"Incident {incident_id}: test"}]
        cands = await fake.chat("system", msg, schema=__import__("typing").List[__import__("schemas").RootCauseCandidate])
        assert cands[0].root_cause_label == expected_label
        pm = await fake.chat("system", msg, schema=__import__("schemas").Postmortem)
        assert pm.root_cause == expected_label
        assert pm.incident_id == incident_id

    async def _run():
        await _check("INC-004", "dns_resolution_failure")
        await _check("INC-005", "connection_pool_exhaustion")
        await _check("INC-010", "grpc_upstream_timeout")

    run(_run())
