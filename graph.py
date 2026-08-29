"""LangGraph orchestration + CLI entrypoint for the Agentic Incident Postmortem
Synthesizer — Phase 6.

Binds the seven async nodes from `agents.py` into a `StateGraph(AgentState)` with
an async `timeline`/`analyze` fan-out (via `langgraph.types.Send`). Each node is
`async def node(state, *, llm, tracer, conn, collection)`, so it is wrapped in a
closure that swallows LangGraph's `(state, config)` call and injects the bound
config — keeping `AgentState` pure data (except `recall_task`).

The CLI entrypoint (`python graph.py --incident INC-001 [--live] [--fake]`) builds
an `LLMAdapter`, a sqlite conn, an `incident_memory` collection and a `Tracer`, then
calls `arun_incident`. It is reused by Phase 7 (`api.py`) and Phase 8 (`eval.py`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agents import (
    AgentState,
    ingest_node,
    timeline_node,
    analyze_node,
    writer_node,
    verifier_node,
    human_gate_node,
    memory_writer_node,
)
from schemas import Evidence

ROOT = os.path.dirname(os.path.abspath(__file__))


def _bind(node, *, llm, tracer, conn, collection):
    async def _runner(state, config=None):
        return await node(state, llm=llm, tracer=tracer, conn=conn, collection=collection)

    return _runner


def build_graph(llm, tracer, conn, collection):
    b = StateGraph(AgentState)
    for name, fn in [
        ("ingest", ingest_node),
        ("timeline", timeline_node),
        ("analyze", analyze_node),
        ("writer", writer_node),
        ("verifier", verifier_node),
        ("human_gate", human_gate_node),
        ("memory_writer", memory_writer_node),
    ]:
        b.add_node(name, _bind(fn, llm=llm, tracer=tracer, conn=conn, collection=collection))
    b.add_edge(START, "ingest")
    b.add_conditional_edges(
        "ingest", lambda s: [Send("timeline", s), Send("analyze", s)]
    )
    b.add_edge(["timeline", "analyze"], "writer")  # fan-in merge
    b.add_edge("writer", "verifier")
    b.add_edge("verifier", "human_gate")
    b.add_edge("human_gate", "memory_writer")
    b.add_edge("memory_writer", END)
    return b.compile()


async def arun_incident(incident: Dict[str, Any], *, llm, tracer, conn, collection):
    app = build_graph(llm, tracer, conn, collection)
    init = {
        "incident_id": incident["id"],
        "window_start": incident["window_start"],
        "window_end": incident["window_end"],
        "description": incident["description"],
        "evidence": [
            Evidence(**{**e, "incident_id": incident["id"]})
            for e in incident["evidence"]
        ],
    }
    return await app.ainvoke(init)


def load_incident(incident_id: str, incidents_dir: str | None = None) -> Dict[str, Any]:
    # Guard the id up front so Phase 7/8 reuse can't slip a path-traversal or
    # malformed id into `os.path.join` below.
    if not isinstance(incident_id, str) or not __import__("re").fullmatch(
        r"[A-Za-z0-9._-]+", incident_id
    ):
        raise ValueError(f"invalid incident id: {incident_id!r}")
    incidents_dir = incidents_dir or os.path.join(ROOT, "incidents")
    path = os.path.join(incidents_dir, f"{incident_id}.json")
    with open(path, "r", encoding="utf-8") as fh:
        incident = json.load(fh)
    # Guard the payload shape so downstream nodes fail fast on a bad fixture.
    missing = [k for k in ("id", "window_start", "window_end", "description", "evidence") if k not in incident]
    if missing:
        raise ValueError(f"incident {incident_id!r} missing required fields: {missing}")
    if incident["id"] != incident_id:
        raise ValueError(
            f"incident id mismatch: file is {incident['id']!r} but requested {incident_id!r}"
        )
    return incident


async def run_cli(
    incident_id: str,
    *,
    live: bool = False,
    fake: bool = False,
    db_path: str = ":memory:",
    traj_dir: str | None = None,
    incidents_dir: str | None = None,
    seed_memory: bool = True,
):
    from llm_adapter import LLMAdapter
    from rag import create_memory_collection, _default_embed
    from store import connect
    from tracer import Tracer

    incident = load_incident(incident_id, incidents_dir)
    tracer = Tracer(incident["id"], out_dir=traj_dir or os.path.join(ROOT, "trajectories"))
    conn = connect(db_path)

    if fake:
        # Offline placeholder run: reuse the reproducible FakeLLMAdapter + a seeded
        # in-memory collection so trajectories/ can be committed without fixtures.
        from helpers import FakeLLMAdapter, make_collection, seed_prior

        llm = FakeLLMAdapter()
        collection = make_collection()
        if seed_memory:
            await seed_prior(collection)
    else:
        llm = LLMAdapter(live=live)
        collection = create_memory_collection(
            name="incident_memory", embed_fn=_default_embed, backend="auto"
        )

    return await arun_incident(
        incident, llm=llm, tracer=tracer, conn=conn, collection=collection
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run the incident postmortem graph end-to-end."
    )
    p.add_argument("--incident", required=True, help="Incident id, e.g. INC-001")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true", help="Use the live LLM (needs API key)")
    group.add_argument(
        "--fake",
        action="store_true",
        help="Use the offline FakeLLMAdapter (placeholder sample, no network/fixtures)",
    )
    p.add_argument("--db", default=":memory:", help="SQLite db path (default in-memory)")
    p.add_argument(
        "--traj-dir",
        default=None,
        help="Trajectory output dir (default ./trajectories)",
    )
    args = p.parse_args(argv)

    result = asyncio.run(
        run_cli(
            args.incident,
            live=args.live,
            fake=args.fake,
            db_path=args.db,
            traj_dir=args.traj_dir,
        )
    )

    verified = result.get("verification")
    score = verified.verification_score if verified is not None else None
    print(
        f"incident={result.get('incident_id')} "
        f"decision={result.get('human_decision')} "
        f"verification_score={score}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
