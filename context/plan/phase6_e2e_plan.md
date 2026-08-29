# Phase 6 — E2E Orchestration Plan

**Status:** implemented (72 passed, 1 skipped offline; trajectories/INC-001 committed sample generated via `--fake`)
**Depends on:** Phases 0–5 (all built + tested; `test_agents.py` manually sequences the 7 nodes)
**Missing after Phase 5:** `graph.py`, `tests/test_graph_integration.py`, committed `trajectories/` sample, `fixtures/llm_cache.jsonl` (deferred to Phase 8).

**Environment confirmed:** `langgraph` importable; `Send` is in `langgraph.types` (NOT `langgraph.graph`);
`StateGraph.add_edge(start_key: str | list[str], end_key)` supports fan-in; pytest 8.4.2 works.
No `fixtures/llm_cache.jsonl` exists yet, so the real `LLMAdapter` chat path cannot replay offline without live recording.

**Scope decisions (user):** (a) run e2e BOTH ways — offline fake-adapter test + skippable real-replay test;
(b) implement `graph.py` + a reusable entrypoint + commit a sample `trajectories/` artifact.

---

## Deliverable 1 — `graph.py` (LangGraph orchestration + entrypoint)

Bind the 7 async nodes from `agents.py` into a `StateGraph(AgentState)` with an async
`timeline`/`analyze` fan-out. Nodes are `async def node(state, *, llm, tracer, conn, collection)`,
so each is wrapped in a closure that swallows LangGraph's `(state, config)` call and injects the
bound config — this keeps `AgentState` pure data (except `recall_task`).

```python
from functools import partial
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from agents import (AgentState, ingest_node, timeline_node, analyze_node,
                    writer_node, verifier_node, human_gate_node, memory_writer_node)
from schemas import Evidence

def _bind(node, *, llm, tracer, conn, collection):
    async def _runner(state, config=None):
        return await node(state, llm=llm, tracer=tracer, conn=conn, collection=collection)
    return _runner

def build_graph(llm, tracer, conn, collection):
    b = StateGraph(AgentState)
    for name, fn in [("ingest", ingest_node), ("timeline", timeline_node),
                     ("analyze", analyze_node), ("writer", writer_node),
                     ("verifier", verifier_node), ("human_gate", human_gate_node),
                     ("memory_writer", memory_writer_node)]:
        b.add_node(name, _bind(fn, llm=llm, tracer=tracer, conn=conn, collection=collection))
    b.add_edge(START, "ingest")
    b.add_conditional_edges("ingest", lambda s: [Send("timeline", s), Send("analyze", s)])
    b.add_edge(["timeline", "analyze"], "writer")   # fan-in merge
    b.add_edge("writer", "verifier")
    b.add_edge("verifier", "human_gate")
    b.add_edge("human_gate", "memory_writer")
    b.add_edge("memory_writer", END)
    return b.compile()

async def arun_incident(incident, *, llm, tracer, conn, collection):
    app = build_graph(llm, tracer, conn, collection)
    init = {"incident_id": incident["id"], "window_start": incident["window_start"],
            "window_end": incident["window_end"], "description": incident["description"],
            "evidence": [Evidence(**e) for e in incident["evidence"]]}
    return await app.ainvoke(init)
```

Plus a CLI entrypoint (`if __name__ == "__main__": --incident INC-XXX [--live]`) that builds an
`LLMAdapter`, a sqlite conn (`:memory:` or file), an `incident_memory` collection, a `Tracer`, and
calls `arun_incident` — reused by Phase 7 (`api.py`) and Phase 8 (`eval.py`).

---

## Deliverable 2 — `tests/test_graph_integration.py`

- **`test_graph_fake_adapter_e2e`** (always runs, offline): build graph with a `FakeLLMAdapter`
  (reuse the one in `test_agents.py` — see helpers note below), in-memory sqlite, in-memory
  `incident_memory` with `INC-002` seeded, `Tracer(tmp_path)`. `await arun_incident(INC-001)`.
  Assert: `postmortem` is a `Postmortem` with all sections; `verification` is a `VerificationReport`;
  `human_decision == "approved"`; `coll.count() == 2`; `postmortem` row status
  `running → approved`; `trajectories/INC-001/` has all 7 `{agent}.json` + `manifest.json`;
  no recalled id appears in any `claims[].evidence_refs`.
- **`test_graph_fanout_runs_both_branches`**: assert both `timeline_events` and `candidates` are
  populated (proves `Send` fan-out executed independently).
- **`test_graph_replay_e2e`** (`@pytest.mark.skipif(not os.path.exists("fixtures/llm_cache.jsonl"), ...)`):
  same invariants via the **real** `LLMAdapter` in replay mode (embeddings use offline fallback).
  Skipped until fixtures are recorded.
- **`test_graph_rejects_when_verification_fails`** (optional): scenario where verification score < 1.0
  asserts `human_decision == "rejected"` + status `rejected`.

### Recommended refactor: `tests/helpers.py`
Both `test_agents.py` and the new integration test need `FakeLLMAdapter` — move it to
`tests/helpers.py` to avoid duplication.

---

## Deliverable 3 — committed `trajectories/` sample

Run `python graph.py --incident INC-001` once. If `fixtures/llm_cache.jsonl` exists, it produces a
real-LLM evidence pack; if not, generate a structural sample via the fake adapter (clearly a
placeholder, refreshed during Phase 8 `--live`) and commit `trajectories/INC-001/`.

---

## Verification

- `python -m pytest tests/test_graph_integration.py -q` → green (fake test runs; replay test skips cleanly offline).
- `python graph.py --incident INC-001` runs end-to-end and writes `trajectories/INC-001/`.
- Full suite `python -m pytest` stays green (no regressions; ~69 existing tests unaffected).

---

## Risks / notes

- If the installed LangGraph rejects `add_edge(["timeline","analyze"], "writer")` despite the list
  signature, insert a no-op `merge` node (`timeline→merge`, `analyze→merge`, `merge→writer`) —
  verify at implementation since `add_edge` already declares `list[str]` support.
- `recall_task` (an `asyncio.Task`) lives in `AgentState` between `ingest` and `analyze`; LangGraph
  passes it through untouched (no persister configured).
- Real-replay test stays skipped until Phase 8 records fixtures — expected, keeps Phase 6 fully offline/reproducible.
