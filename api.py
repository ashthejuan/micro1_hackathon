"""CLI human gate for the Agentic Incident Postmortem Synthesizer — Phase 7.

Covers the plan's `api.py` contract:

  python api.py run <incident_id|.json> [--db PATH] [--traj-dir DIR] [--fake|--live]
  python api.py approve <id> [--apply INC-014] [--apply INC-015 ...] [--by NAME] [--db PATH]
  python api.py reject <id> <reason> [--db PATH]
  python api.py show <id> [--db PATH]

* `run`  — ingest → timeline/analyze (fan-out) → writer → verifier, then
  persists the draft as `pending_approval` (no auto-approve; the human must call
  `approve`). Returns the id + draft + report. Mirrors the truncated graph the
  PRD describes for the CLI gate (human_gate is simulated in eval, real here).

* `approve` / `reject` — the mandatory human checkpoint (§8). `approve` flips
  `pending_approval → approved`, stamps `approved_by`/`time_approved`, honours
  `--apply` (flips consulted incidents to applied), writes the human_gate
  trajectory, and embeds the approved postmortem into `incident_memory` (consult-only
  leak is refused). `reject` flips to `rejected`.

* `show` — prints the draft + verification report + consulted incidents for an id.

All commands honour `--db` (default ``./app.db`` persistent, ``:memory:`` in
tests) and share the sqlite + chroma bootstrap so `run → approve → show`
round-trips through one DB file. Offline `--fake` uses the deterministic
FakeLLMAdapter + hash embed (no network/key).

Inspired by the Phase 6 graph.py orchestrator but intentionally *does not* reuse
its compiled 7-node graph (which ends in an auto-approve human_gate for the eval
harness). The CLI owns its own pending gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Ensure root is importable when invoked as `python api.py ...`
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents import ingest_node, timeline_node, analyze_node, writer_node, verifier_node  # noqa: E402
from memory import is_consult_only_leak, store_memory  # noqa: E402
from schemas import ConsultedIncident, Postmortem, VerificationReport  # noqa: E402
from store import (  # noqa: E402
    connect,
    get_cli_backend,
    init_db,
    insert_verification_rows,
    set_cli_backend,
    upsert_postmortem,
)
from tracer import Tracer  # noqa: E402

DEFAULT_DB = os.path.join(ROOT, "app.db")
DEFAULT_TRAJ_DIR = os.path.join(ROOT, "trajectories")
DEFAULT_INCIDENTS_DIR = os.path.join(ROOT, "incidents")
_ID_RE = re.compile(r"[A-Za-z0-9._-]+")

# ------------------------------------------------------------------ helpers

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _validate_id(incident_id: str) -> None:
    if not isinstance(incident_id, str) or not _ID_RE.fullmatch(incident_id):
        raise ValueError(f"invalid incident id: {incident_id!r}")


def _load_incident_arg(arg: str, incidents_dir: str | None = None) -> Dict[str, Any]:
    """Load an incident by id *or* by file path.

    * If ``arg`` is an existing file (or ends with .json and lives under
      ``incidents_dir``) it is read directly.
    * Otherwise it is treated as an incident id and loaded via ``graph.load_incident``
      (which validates the id and the payload shape).
    """
    # Direct file path (absolute or relative)
    if os.path.exists(arg) and arg.endswith(".json"):
        with open(arg, "r", encoding="utf-8") as fh:
            inc = json.load(fh)
        _validate_id(inc.get("id", ""))
        return inc
    # incidents_dir/<arg> if arg looks like a file
    incidents_dir = incidents_dir or DEFAULT_INCIDENTS_DIR
    candidate = os.path.join(incidents_dir, arg)
    if os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as fh:
            inc = json.load(fh)
        return inc
    # Fallback: treat as incident id
    from graph import load_incident as _load

    return _load(arg, incidents_dir)


def _get_conn(db_path: str):
    conn = connect(db_path)
    init_db(conn)
    return conn


def _get_collection(*, db_path: str = DEFAULT_DB, fake: bool = False, live: bool = False):
    """Return an incident_memory collection.

    * ``fake`` → in-memory hash-embed collection (offline, deterministic).
    * otherwise → Chroma ``incident_memory`` (auto backend) via rag.py.
      In CLI persistent use this may be a PersistentClient under ./chroma_db
      when chromadb is available; otherwise in-memory.
    """
    if fake:
        from helpers import make_collection

        return make_collection()
    # Live / replay path — route through rag
    try:
        from rag import create_memory_collection, _default_embed

        # Use a file-backed chroma dir when the default persistent db is in use
        # so successive CLI invocations share the memory. Tests pass :memory: db
        # and therefore get an ephemeral client (no persist_dir).
        persist_dir = None
        if db_path != ":memory:" and os.path.isdir(ROOT):
            persist_dir = os.path.join(ROOT, "chroma_db")
        return create_memory_collection(
            name="incident_memory",
            embed_fn=_default_embed,
            backend="auto",
            persist_dir=persist_dir,
        )
    except Exception:
        # No chromadb — fall back to in-memory so CLI still works
        from helpers import make_collection

        return make_collection()


def _get_llm(*, fake: bool = False, live: bool = False):
    if fake:
        from helpers import FakeLLMAdapter

        return FakeLLMAdapter()
    from llm_adapter import LLMAdapter

    return LLMAdapter(live=live)


# --------------------------------------------------------- run pipeline core
async def _run_to_pending(
    incident: Dict[str, Any],
    *,
    llm: Any,
    tracer: Tracer,
    conn: Any,
    collection: Any,
) -> Dict[str, Any]:
    """Execute ingest → timeline/analyze (parallel) → writer → verifier, then
    persist the draft as ``pending_approval``. No human_gate / memory_writer."""
    from schemas import Evidence

    state: Dict[str, Any] = {
        "incident_id": incident["id"],
        "window_start": incident["window_start"],
        "window_end": incident["window_end"],
        "description": incident["description"],
        "evidence": [
            Evidence(**{**e, "incident_id": incident["id"]}) for e in incident["evidence"]
        ],
    }

    # ingest (fires async recall)
    res = await ingest_node(state, llm=llm, tracer=tracer, conn=conn, collection=collection)
    state.update(res)

    # timeline + analyze fan-out
    t1 = asyncio.create_task(timeline_node(state, llm=llm, tracer=tracer, conn=conn, collection=collection))
    t2 = asyncio.create_task(analyze_node(state, llm=llm, tracer=tracer, conn=conn, collection=collection))
    r1, r2 = await asyncio.gather(t1, t2)
    state.update(r1)
    state.update(r2)

    # writer (with citation-integrity retry)
    res = await writer_node(state, llm=llm, tracer=tracer, conn=conn, collection=collection)
    state.update(res)

    # verifier (deterministic set-check; never raises on consult-only leak)
    res = await verifier_node(state, llm=llm, tracer=tracer, conn=conn, collection=collection)
    state.update(res)

    pm: Postmortem = state["postmortem"]
    ver: VerificationReport = state["verification"]
    consulted: List[ConsultedIncident] = state.get("consulted", [])

    # Persist as pending_approval (the human must call `approve`)
    import json as _json

    upsert_postmortem(
        conn,
        pm.incident_id,
        draft_json=pm.model_dump_json(),
        verification_json=ver.model_dump_json(),
        consulted_json=_json.dumps([c.model_dump() for c in consulted], ensure_ascii=False),
        status="pending_approval",
        approved_by=None,
        time_approved=None,
    )
    # Also mirror per-claim rows for queryable eval — fail loudly so CLI
    # does not report pending while per-claim rows are missing (Design Risk #5).
    try:
        insert_verification_rows(conn, pm.incident_id, ver)
    except Exception as exc:
        print(f"error: failed to persist verification rows: {exc}", file=sys.stderr)
        raise
    # Assert rows exist before returning (secondary guard for the same risk).
    try:
        row_count = conn.execute(
            "SELECT COUNT(*) AS c FROM verification WHERE incident_id=?", (pm.incident_id,)
        ).fetchone()["c"]
        if row_count != len(ver.claim_reports):
            raise RuntimeError(
                f"verification row count mismatch: expected {len(ver.claim_reports)}, got {row_count}"
            )
    except Exception as exc:
        # Verification table mismatch is a hard error — do not leave a half-written draft.
        print(f"error: verification row assertion failed: {exc}", file=sys.stderr)
        raise

    state["human_decision"] = "pending"
    return state


# ------------------------------------------------------------------- CLI fns
async def cmd_run(
    incident_arg: str,
    *,
    db_path: str = DEFAULT_DB,
    traj_dir: str = DEFAULT_TRAJ_DIR,
    incidents_dir: str | None = None,
    fake: bool = False,
    live: bool = False,
) -> Dict[str, Any]:
    incident = _load_incident_arg(incident_arg, incidents_dir)
    _validate_id(incident["id"])
    # Validate shape — keep helper contract weak but enforce here (NIT).
    # `_load_incident_arg` ID-validates for direct paths; missing fields / id
    # mismatch are still caught here so the error surfaces before any LLM call.
    for k in ("window_start", "window_end", "description", "evidence"):
        if k not in incident:
            raise ValueError(f"incident {incident['id']!r} missing required field: {k}")
    if incident["id"] != incident.get("id"):
        raise ValueError(f"incident id mismatch: {incident['id']!r}")
    tracer = Tracer(incident["id"], out_dir=traj_dir)
    conn = _get_conn(db_path)
    collection = _get_collection(db_path=db_path, fake=fake, live=live)
    llm = _get_llm(fake=fake, live=live)

    state = await _run_to_pending(incident, llm=llm, tracer=tracer, conn=conn, collection=collection)
    pm = state["postmortem"]
    ver = state["verification"]
    # Persist backend choice so approve can enforce the same store (Design Risk #2).
    backend = "fake" if fake else ("live" if live else "replay")
    try:
        set_cli_backend(conn, pm.incident_id, backend, time_created=_now_iso())
    except Exception as exc:
        print(f"warning: failed to persist backend meta: {exc}", file=sys.stderr)
    print(
        f"incident={pm.incident_id} status=pending_approval "
        f"verification_score={ver.verification_score:.2f} "
        f"claims={len(pm.claims)} trajectory={tracer.trace_dir()}"
    )
    return state


async def cmd_approve(
    incident_id: str,
    *,
    db_path: str = DEFAULT_DB,
    traj_dir: str = DEFAULT_TRAJ_DIR,
    applied: Optional[List[str]] = None,
    approved_by: str = "human",
    fake: bool = False,
    live: bool = False,
) -> Dict[str, Any]:
    _validate_id(incident_id)
    conn = _get_conn(db_path)
    row = conn.execute(
        "SELECT draft_json, verification_json, consulted_json, status FROM postmortem WHERE incident_id=?",
        (incident_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"no postmortem found for {incident_id!r}; run `api.py run {incident_id}` first")
    if row["status"] not in ("pending_approval", "pending"):
        raise SystemExit(f"postmortem {incident_id} is {row['status']!r}, not pending_approval")

    # Enforce collection-store consistency (Design Risk #2):
    # `run --fake` uses an in-memory hash-embed store; default `approve`
    # uses a persistent Chroma store. Without this guard the approved
    # postmortem would embed into a different store than recall used.
    stored_backend = get_cli_backend(conn, incident_id)
    requested_backend = "fake" if fake else ("live" if live else None)
    if stored_backend is not None and requested_backend is not None and requested_backend != stored_backend:
        raise SystemExit(
            f"backend mismatch: run used --{stored_backend} but approve was called with --{requested_backend}. "
            f"Re-run approve with --{stored_backend} to match the run's store."
        )
    # If caller did not specify a backend, inherit the run's choice.
    effective_fake = fake
    effective_live = live
    if requested_backend is None and stored_backend is not None:
        effective_fake = stored_backend == "fake"
        effective_live = stored_backend == "live"

    applied = list(applied or [])
    for aid in applied:
        _validate_id(aid)

    pm = Postmortem.model_validate_json(row["draft_json"])
    consulted: List[ConsultedIncident] = []
    if row["consulted_json"]:
        try:
            consulted = [ConsultedIncident.model_validate(c) for c in json.loads(row["consulted_json"])]
        except Exception:
            consulted = list(pm.consulted_incidents)

    # Flip applied flags
    if applied:
        consulted = [
            c.model_copy(update={"applied": True}) if c.incident_id in applied else c
            for c in consulted
        ]
        pm = pm.model_copy(update={"consulted_incidents": consulted})

    ver: Optional[VerificationReport] = None
    if row["verification_json"]:
        try:
            ver = VerificationReport.model_validate_json(row["verification_json"])
        except Exception:
            ver = None

    now = _now_iso()
    upsert_postmortem(
        conn,
        incident_id,
        draft_json=pm.model_dump_json(),
        verification_json=row["verification_json"],
        consulted_json=json.dumps([c.model_dump() for c in consulted], ensure_ascii=False),
        status="approved",
        approved_by=approved_by,
        time_approved=now,
    )

    tracer = Tracer(incident_id, out_dir=traj_dir)
    tracer.record(
        "human_gate",
        system_prompt="human gate (CLI approve)",
        user_prompt="",
        input_state={"incident_id": incident_id, "applied_incidents": applied},
        human_decision="approved",
        output={"status": "approved", "approved_by": approved_by, "applied": applied},
    )

    # Embed into incident_memory unless it is a consult-only leak
    collection = _get_collection(db_path=db_path, fake=effective_fake, live=effective_live)
    recalled = {c.incident_id for c in consulted}
    embedded = False
    if not is_consult_only_leak(pm, list(recalled)):
        score = ver.verification_score if ver is not None else None
        try:
            await store_memory(pm, collection, verification_score=score)
            embedded = True
        except Exception as exc:
            print(f"warning: embed failed: {exc}", file=sys.stderr)
        tracer.record(
            "memory_writer",
            system_prompt="memory_writer (CLI approve → embed)",
            user_prompt="",
            input_state={"incident_id": incident_id, "verification_score": score},
            output={"embedded": embedded},
        )
    else:
        tracer.record(
            "memory_writer",
            system_prompt="memory_writer (CLI approve → embed)",
            user_prompt="",
            input_state={"incident_id": incident_id, "skipped": "consult-only leak"},
            output={"embedded": False, "reason": "consult-only leak"},
        )

    print(f"incident={incident_id} status=approved approved_by={approved_by} embedded={embedded}")
    return {"incident_id": incident_id, "status": "approved", "embedded": embedded}


async def cmd_reject(
    incident_id: str,
    reason: str,
    *,
    db_path: str = DEFAULT_DB,
    traj_dir: str = DEFAULT_TRAJ_DIR,
) -> Dict[str, Any]:
    _validate_id(incident_id)
    conn = _get_conn(db_path)
    row = conn.execute(
        "SELECT draft_json, verification_json, consulted_json, status FROM postmortem WHERE incident_id=?",
        (incident_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"no postmortem found for {incident_id!r}")
    if row["status"] not in ("pending_approval", "pending"):
        raise SystemExit(f"postmortem {incident_id} is {row['status']!r}, not pending_approval")

    now = _now_iso()
    upsert_postmortem(
        conn,
        incident_id,
        draft_json=row["draft_json"],
        verification_json=row["verification_json"],
        consulted_json=row["consulted_json"],
        status="rejected",
        approved_by=None,
        time_approved=None,
    )

    tracer = Tracer(incident_id, out_dir=traj_dir)
    tracer.record(
        "human_gate",
        system_prompt="human gate (CLI reject)",
        user_prompt="",
        input_state={"incident_id": incident_id, "reason": reason},
        human_decision="rejected",
        output={"status": "rejected", "reason": reason},
    )
    print(f"incident={incident_id} status=rejected reason={reason!r}")
    return {"incident_id": incident_id, "status": "rejected"}


def cmd_show(incident_id: str, *, db_path: str = DEFAULT_DB) -> Dict[str, Any]:
    _validate_id(incident_id)
    conn = _get_conn(db_path)
    row = conn.execute(
        "SELECT incident_id, draft_json, verification_json, consulted_json, status, approved_by, time_approved "
        "FROM postmortem WHERE incident_id=?",
        (incident_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"no postmortem found for {incident_id!r}")
    draft = json.loads(row["draft_json"]) if row["draft_json"] else None
    ver = json.loads(row["verification_json"]) if row["verification_json"] else None
    consulted = json.loads(row["consulted_json"]) if row["consulted_json"] else []
    out = {
        "incident_id": row["incident_id"],
        "status": row["status"],
        "approved_by": row["approved_by"],
        "time_approved": row["time_approved"],
        "draft": draft,
        "verification": ver,
        "consulted_incidents": consulted,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ------------------------------------------------------------------ argparser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="api.py",
        description="Agentic Incident Postmortem — CLI human gate (Phase 7).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Ingest → verifier and leave pending_approval for human gate.")
    pr.add_argument("incident", help="Incident id (e.g. INC-001) or path to an incident .json file.")
    pr.add_argument("--db", default=DEFAULT_DB, help=f"SQLite db path (default {DEFAULT_DB})")
    pr.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR, help="Trajectory output dir.")
    pr.add_argument("--incidents-dir", default=None, help="Incidents dir (default ./incidents).")
    g = pr.add_mutually_exclusive_group()
    g.add_argument("--live", action="store_true", help="Use live LLM (needs API key; records fixtures).")
    g.add_argument("--fake", action="store_true", help="Use offline FakeLLMAdapter (no network).")

    pa = sub.add_parser("approve", help="Approve a pending postmortem (human checkpoint).")
    pa.add_argument("incident_id", help="Incident id to approve (e.g. INC-001).")
    pa.add_argument("--apply", action="append", dest="apply", default=[], help="Consulted incident to mark applied (repeatable).")
    pa.add_argument("--by", dest="approved_by", default="human", help="Approver name (default human).")
    pa.add_argument("--db", default=DEFAULT_DB, help=f"SQLite db path (default {DEFAULT_DB})")
    pa.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR, help="Trajectory output dir.")
    ga = pa.add_mutually_exclusive_group()
    ga.add_argument("--live", action="store_true", help="Use live embeddings when embedding to memory.")
    ga.add_argument("--fake", action="store_true", help="Use offline hash embed.")

    prj = sub.add_parser("reject", help="Reject a pending postmortem.")
    prj.add_argument("incident_id", help="Incident id to reject.")
    prj.add_argument("reason", help="Reason for rejection.")
    prj.add_argument("--db", default=DEFAULT_DB, help=f"SQLite db path (default {DEFAULT_DB})")
    prj.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR, help="Trajectory output dir.")

    ps = sub.add_parser("show", help="Show a postmortem draft + verification report.")
    ps.add_argument("incident_id", help="Incident id.")
    ps.add_argument("--db", default=DEFAULT_DB, help=f"SQLite db path (default {DEFAULT_DB})")

    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "run":
            asyncio.run(
                cmd_run(
                    args.incident,
                    db_path=args.db,
                    traj_dir=args.traj_dir,
                    incidents_dir=args.incidents_dir,
                    fake=bool(args.fake),
                    live=bool(args.live),
                )
            )
        elif args.cmd == "approve":
            asyncio.run(
                cmd_approve(
                    args.incident_id,
                    db_path=args.db,
                    traj_dir=args.traj_dir,
                    applied=args.apply or [],
                    approved_by=args.approved_by,
                    fake=bool(getattr(args, "fake", False)),
                    live=bool(getattr(args, "live", False)),
                )
            )
        elif args.cmd == "reject":
            asyncio.run(cmd_reject(args.incident_id, args.reason, db_path=args.db, traj_dir=args.traj_dir))
        elif args.cmd == "show":
            cmd_show(args.incident_id, db_path=args.db)
        else:
            build_parser().print_help()
            return 2
    except SystemExit as e:
        # Re-raise intentional exits (argparse or our guard) with their code/msg
        raise
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
