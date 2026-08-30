"""Evaluation harness — Phase 7/8 (judging-critical).

Implements the plan's ``eval.py`` contract (pytest, offline replay by default):

* Pre-seed ``INC-001..003`` into ``incident_memory`` (the consulted-hypotheses
  collection) so recall has priors.
* Run ``INC-004..011`` (plan says 4–10; we cover 4–11, tolerating extra fixtures)
  through **two** paths on the *same* incidents:

  - **Agent** — the full 7-node ``graph.arun_incident`` with ``FakeLLMAdapter``
    (offline, deterministic) or ``LLMAdapter(live=…)`` when ``--live``.
  - **Baseline A** — a single-prompt stub that outputs the *same* ``Postmortem``
    schema but deliberately blames the planted ``red_herring`` and cites a
    hallucinated ``E99``. This is the honest structural delta the PRD argues
    for (typed claims + set-check, not "being smarter").

* Score each run on:

  - **Primary:** verification pass rate — deterministic SQLite set-check
    ``backed / total`` (``VerificationReport.verification_score``).
  - **Secondary:** red-herring rejection — ``candidates[0].root_cause_label !=
    incident["red_herring"]`` (label-vs-truth, not judgement).
  - **Secondary:** structural completeness — required sections present.

* Write one row per (mode, incident) into the ``evaluation_run`` table and
  print a comparison table. Deterministic across two offline runs (same scores)
  — enforced by the hash-embed fallback and the FakeLLM stub.

* ``--live`` guard (BLOCKER fix): when ``--live`` is requested but neither
  ``OPENAI_API_KEY`` nor ``fixtures/llm_cache.jsonl`` exists, the CLI prints
  ``skipped: no key/fixture`` and exits 0 instead of crashing with a 401.
  Offline judges run ``python eval.py`` (or ``--fake``) and never hit the network.

Usage::

    python eval.py                  # offline, no key, deterministic
    python eval.py --live           # hits OPENAI_BASE_URL and refreshes fixtures
                                    # (no-op with message if no key/fixture)
    pytest tests/test_eval.py -q    # the same harness, asserted
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from generate_incidents import generate_incidents  # noqa: E402
from helpers import FakeLLMAdapter, make_collection  # noqa: E402
from schemas import Claim, ConsultedIncident, Postmortem, TimelineEvent, VerificationReport  # noqa: E402
from store import connect, init_db, insert_evaluation_run, verify_postmortem  # noqa: E402
from tracer import Tracer  # noqa: E402

# ---------------------------------------------------------------- metrics
REQUIRED_SECTIONS = ("summary", "impact", "root_cause", "timeline", "action_items", "claims")


def completeness_score(pm: Postmortem) -> float:
    """Structural completeness — 1 iff every required section is non-empty."""
    checks = [
        bool(pm.summary and pm.summary.strip()),
        bool(pm.impact and pm.impact.strip()),
        bool(pm.root_cause and pm.root_cause.strip()),
        bool(pm.timeline),
        bool(pm.action_items),
        bool(pm.claims),
    ]
    return 1.0 if all(checks) else 0.0


def _top_label(candidates: List[Any], pm: Postmortem | None) -> str | None:
    if candidates:
        c0 = candidates[0]
        # Candidates may be dicts or RootCauseCandidate objects
        if isinstance(c0, dict):
            return c0.get("root_cause_label")
        return getattr(c0, "root_cause_label", None)
    if pm is not None:
        return getattr(pm, "root_cause", None)
    return None


def red_herring_correct(candidates: List[Any], pm: Postmortem | None, incident: Dict[str, Any]) -> int:
    """Label-vs-truth red-herring check — 1 if top label != red_herring."""
    herring = incident.get("red_herring")
    label = _top_label(candidates, pm)
    if herring is None or label is None:
        return 0
    return 1 if label != herring else 0


# -------------------------------------------------------------- baselines
def _baseline_postmortem(incident: Dict[str, Any]) -> Postmortem:
    """Deterministic Baseline A — same schema, but *wrongs* the two metrics.

    * ``root_cause`` = ``red_herring`` (so red-herring check fails).
    * ``claims[0].evidence_refs = ["E99"]`` (hallucinated, so verification fails).
    * All required sections are still present so completeness stays 1 (the
      honest structural delta: verification fails even though structure is fine).
    """
    return Postmortem(
        incident_id=incident["id"],
        summary=f"Baseline summary for {incident['id']}: {incident['description']}",
        impact="Baseline impact (generic, not evidence-backed).",
        root_cause=incident.get("red_herring") or "unknown",
        timeline=[
            TimelineEvent(
                ts=incident["window_start"],
                description="Baseline timeline (hallucinated refs)",
                evidence_refs=["E99"],
            )
        ],
        action_items=["Baseline action: review incident."],
        claims=[
            Claim(
                statement="Baseline claim cites hallucinated evidence",
                evidence_refs=["E99"],
            )
        ],
        consulted_incidents=[],
    )


def _verify_baseline(conn, incident: Dict[str, Any], pm: Postmortem) -> VerificationReport:
    """Verify the baseline postmortem against the evidence store.

    Seeds the evidence table for this incident (so verification is comparable
    to the agent path) and runs the deterministic set-check.
    """
    from store import insert_evidence, insert_incident

    # Ensure evidence is present for this incident (eval may run baseline
    # without having run the agent's ingest).
    try:
        insert_incident(conn, incident, status="running")
    except Exception:
        pass
    try:
        insert_evidence(conn, incident["id"], incident["evidence"])
    except Exception:
        pass
    return verify_postmortem(conn, incident["id"], pm, recalled_incident_ids=[])


# -------------------------------------------------------------- agent runner
async def _run_agent_on_incident(
    incident: Dict[str, Any],
    *,
    llm: Any,
    conn: Any,
    collection: Any,
    traj_dir: str | None = None,
) -> Dict[str, Any]:
    """Run the full 7-node graph (ingest → ... → memory_writer) for one incident."""
    from graph import arun_incident

    tracer = Tracer(incident["id"], out_dir=traj_dir or os.path.join(ROOT, "trajectories"))
    result = await arun_incident(incident, llm=llm, tracer=tracer, conn=conn, collection=collection)
    return result


# -------------------------------------------------------------- eval core
async def run_eval(
    *,
    db_path: str = ":memory:",
    traj_dir: str | None = None,
    fake: bool = True,
    live: bool = False,
    incidents_dir: str | None = None,
    seed_ids: List[str] | None = None,
    eval_ids: List[str] | None = None,
    print_table: bool = True,
    isolate: bool = True,
) -> Dict[str, Any]:
    """Execute the full eval harness and return a scores dict.

    Returns ``{"agent": {...}, "baseline": {...}, "rows": [...]}`` where each
    mode's dict holds ``verification_mean``, ``red_herring_rate``, ``completeness_mean``.

    Recall-contamination fix (Design Risk #3): when ``isolate=True`` (default)
    each eval incident starts from a fresh collection seeded only with the
    original priors (1–3). Without isolation the shared ``incident_memory``
    grows via ``memory_writer`` inside the loop, so later siblings recall
    earlier siblings from the *same* run — a feedback loop that biases live
    results. Isolation eliminates that loop at the cost of not measuring the
    cumulative-recall benefit. The note is printed in the table footer.
    """
    incs = {i["id"]: i for i in generate_incidents()}
    # Default split: 1–3 seed, 4–end eval (plan says 4–10; we handle 4–11)
    all_ids = sorted(incs.keys())
    if seed_ids is None:
        seed_ids = [i for i in all_ids if i in ("INC-001", "INC-002", "INC-003")]
    if eval_ids is None:
        eval_ids = [i for i in all_ids if i not in set(seed_ids)]

    conn = connect(db_path)
    init_db(conn)

    from generate_incidents import memory_doc as _memory_doc

    seed_docs = [_memory_doc(incs[sid]) for sid in seed_ids]

    async def _fresh_collection():
        if fake:
            coll = make_collection()
        else:
            from rag import create_memory_collection, _default_embed

            # Use an ephemeral in-memory collection when isolated so the
            # persistent chroma_db is not polluted across incident iterations.
            # Non-isolated mode reuses the shared auto collection (original
            # feedback-loop evaluation).
            if isolate:
                coll = create_memory_collection(name="incident_memory", embed_fn=_default_embed, backend="memory")
            else:
                coll = create_memory_collection(name="incident_memory", embed_fn=_default_embed, backend="auto")
        for doc in seed_docs:
            try:
                await coll.add(id=doc["incident_id"], document=doc["document"], metadata=doc["metadata"])
            except Exception:
                pass
        return coll

    # Non-isolated (legacy) shared collection — created once and mutated by
    # memory_writer across the loop (feedback-loop evaluation). Isolated mode
    # creates a fresh clone per incident below.
    shared_collection = None if isolate else await _fresh_collection()

    # LLM for agent
    if fake:
        llm = FakeLLMAdapter()
    else:
        from llm_adapter import LLMAdapter

        llm = LLMAdapter(live=live)

    agent_scores: List[Dict[str, Any]] = []
    baseline_scores: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    run_id_prefix = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    model_name = getattr(llm, "model", "fake") if not fake else "fake"

    for iid in eval_ids:
        inc = incs[iid]
        # Isolated per-incident collection (fix #3) — fresh seed, no sibling contamination.
        collection = await _fresh_collection() if isolate else shared_collection

        # ---- Agent
        result = await _run_agent_on_incident(inc, llm=llm, conn=conn, collection=collection, traj_dir=traj_dir or os.path.join(ROOT, "trajectories"))
        pm: Postmortem = result["postmortem"]
        ver: VerificationReport = result["verification"]
        cands = result.get("candidates", [])
        a_ver = float(ver.verification_score) if ver is not None else 0.0
        a_rh = red_herring_correct(cands, pm, inc)
        a_comp = completeness_score(pm)
        agent_scores.append({"incident_id": iid, "verification": a_ver, "red_herring": a_rh, "completeness": a_comp})
        insert_evaluation_run(conn, f"{run_id_prefix}-agent-{iid}", "agent", model_name, iid, a_ver, a_rh, now)
        rows.append({"mode": "agent", "incident_id": iid, "verification_score": a_ver, "red_herring_correct": a_rh, "completeness": a_comp})

        # ---- Baseline A (same incident, deterministic stub)
        b_pm = _baseline_postmortem(inc)
        b_ver = _verify_baseline(conn, inc, b_pm)
        b_ver_score = float(b_ver.verification_score)
        b_rh = red_herring_correct([], b_pm, inc)
        b_comp = completeness_score(b_pm)
        baseline_scores.append({"incident_id": iid, "verification": b_ver_score, "red_herring": b_rh, "completeness": b_comp})
        insert_evaluation_run(conn, f"{run_id_prefix}-baseline-{iid}", "baseline", model_name, iid, b_ver_score, b_rh, now)
        rows.append({"mode": "baseline", "incident_id": iid, "verification_score": b_ver_score, "red_herring_correct": b_rh, "completeness": b_comp})

    def _mean(key: str, lst: List[Dict[str, Any]]) -> float:
        return sum(d[key] for d in lst) / len(lst) if lst else 0.0

    summary = {
        "agent": {
            "verification_mean": _mean("verification", agent_scores),
            "red_herring_rate": _mean("red_herring", agent_scores),
            "completeness_mean": _mean("completeness", agent_scores),
            "per_incident": agent_scores,
        },
        "baseline": {
            "verification_mean": _mean("verification", baseline_scores),
            "red_herring_rate": _mean("red_herring", baseline_scores),
            "completeness_mean": _mean("completeness", baseline_scores),
            "per_incident": baseline_scores,
        },
        "rows": rows,
        "seed_ids": seed_ids,
        "eval_ids": eval_ids,
        "model": model_name,
        "isolate": isolate,
    }

    if print_table:
        _print_table(summary)

    return summary


def _print_table(summary: Dict[str, Any]) -> None:
    a = summary["agent"]
    b = summary["baseline"]
    print("\n=== Agentic Postmortem — eval comparison (offline replay) ===")
    print(f"Model: {summary['model']}")
    print(f"Seeds: {summary['seed_ids']}  Eval: {summary['eval_ids']}")
    print(f"{'mode':<10} {'verification':>13} {'red_herring':>12} {'completeness':>13}")
    print("-" * 52)
    print(f"{'agent':<10} {a['verification_mean']:>13.3f} {a['red_herring_rate']:>12.3f} {a['completeness_mean']:>13.3f}")
    print(f"{'baseline':<10} {b['verification_mean']:>13.3f} {b['red_herring_rate']:>12.3f} {b['completeness_mean']:>13.3f}")
    print("-" * 52)
    # Per-incident breakdown (useful for audit)
    for iid in summary["eval_ids"]:
        ar = next((x for x in a["per_incident"] if x["incident_id"] == iid), {})
        br = next((x for x in b["per_incident"] if x["incident_id"] == iid), {})
        print(f"  {iid}: agent v={ar.get('verification',0):.1f} rh={ar.get('red_herring',0)} | base v={br.get('verification',0):.1f} rh={br.get('red_herring',0)}")
    isolated = summary.get("isolate", True)
    print("Note: eval simulates human checkpoint (auto-approve); real gate is CLI `api.py approve` in live use.")
    print("Verifier measures citation integrity not semantic truth (paired with red-herring + completeness metrics).")
    if isolated:
        print("Recall is isolated per incident (seed priors only, no sibling feedback loop).")
    else:
        print("Recall is cumulative (feedback loop): later incidents recall earlier siblings from this run.")
    print()


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the Agentic Postmortem eval harness (Phase 8, offline replay by default).")
    p.add_argument("--live", action="store_true", help="Use LIVE LLM (needs API key; records fixtures).")
    p.add_argument("--fake", action="store_true", help="Force offline FakeLLMAdapter (default).")
    p.add_argument("--db", default=":memory:", help="SQLite db path (default :memory:).")
    p.add_argument("--traj-dir", default=None, help="Trajectory dir (default ./trajectories).")
    p.add_argument("--isolate", action="store_true", default=True, help="Isolate recall per incident (seed priors only, no sibling feedback loop) [default on].")
    p.add_argument("--no-isolate", dest="isolate", action="store_false", help="Disable isolation — cumulative feedback-loop evaluation.")
    args = p.parse_args(argv)

    # --live guard (BLOCKER): fail-soft when no key and no fixture exists.
    if args.live:
        has_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        fixture_path = os.path.join(ROOT, "fixtures", "llm_cache.jsonl")
        has_fixture = os.path.exists(fixture_path)
        if not has_key and not has_fixture:
            print("skipped: no key/fixture — no OPENAI_API_KEY and no fixtures/llm_cache.jsonl; run without --live for offline demo")
            return 0

    # Default is fake (offline, no key). --live overrides.
    live = bool(args.live)
    fake = True if not live else bool(args.fake)
    if live and not fake:
        fake = False

    asyncio.run(run_eval(db_path=args.db, traj_dir=args.traj_dir, fake=fake, live=live, print_table=True, isolate=bool(args.isolate)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
