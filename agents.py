"""Async agent nodes for the Agentic Incident Postmortem Synthesizer — Phase 5.

Seven node functions, each `async def node(state, *, llm, tracer, conn, collection)
-> dict`. Config objects (LLMAdapter, Tracer, sqlite conn, Chroma collection) are
passed explicitly (bound via functools.partial in Phase 6's graph.py) so AgentState
stays pure data. The only non-data field on AgentState is `recall_task`, an
asyncio.Task used to overlap the memory recall with the timeline fan-out (PRD §7).

Each node returns a dict of state updates that the orchestrator merges. The verifier
reuses `store.verify_postmortem` (deterministic set-check + consult-only ban); the
memory writer reuses `memory.store_memory` (embed + add in one call) and refuses to
embed any consult-only leak via `memory.is_consult_only_leak`.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from memory import is_consult_only_leak, recall_incidents, store_memory
from schemas import (
    Claim,
    ConsultedIncident,
    Evidence,
    Postmortem,
    RootCauseCandidate,
    TimelineEvent,
    VerificationReport,
)
from store import (
    init_db,
    insert_evidence,
    insert_incident,
    query_evidence,
    upsert_postmortem,
    valid_evidence_ids,
    verify_postmortem,
)


class AgentState(TypedDict, total=False):
    incident_id: str
    window_start: str
    window_end: str
    description: str
    evidence: List[Evidence]
    recall_task: "asyncio.Task"
    timeline_events: List[TimelineEvent]
    candidates: List[RootCauseCandidate]
    consulted: List[ConsultedIncident]
    postmortem: Optional[Postmortem]
    verification: Optional[VerificationReport]
    human_decision: Optional[str]
    applied_incidents: List[str]


# ----------------------------------------------------------- prompt construction
_SYSTEM_INGEST = "ingest node (no LLM; writes evidence and fires memory recall)"
_SYSTEM_TIMELINE = (
    "You are the timeline agent. Reconstruct a chronological timeline strictly from "
    "the supplied evidence. Only reference evidence ids that appear in the evidence "
    "list; never invent ids."
)
_SYSTEM_ANALYZE = (
    "You are the root-cause analysis agent. Propose ranked root-cause candidates. "
    "Use `contradicting_evidence` to explicitly reject red herrings. You MAY be "
    "motivated by consulted prior incidents but MUST cite the CURRENT incident's "
    "evidence for every conclusion. Only reference current evidence ids; never invent ids."
)
_SYSTEM_WRITER = (
    "You are the postmortem writer. Produce a publishable postmortem. Every Claim "
    "must have non-empty `evidence_refs` that are REAL ids from the supplied "
    "evidence list only. Never invent ids. Set `from_recalled_incident` only when a "
    "claim is motivated by a consulted prior incident (it is then excluded from verification)."
)


def _evidence_block(evidence: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{e['id']} [{e['source']} {e['ts']}] {e['content']}" for e in evidence
    )


def _collect_refs(pm: Postmortem, candidates: List[RootCauseCandidate]) -> List[str]:
    refs: List[str] = []
    for ev in pm.timeline:
        refs.extend(ev.evidence_refs)
    for c in pm.claims:
        refs.extend(c.evidence_refs)
    for cand in candidates:
        refs.extend(cand.supporting_evidence)
        refs.extend(cand.contradicting_evidence)
    return refs


def _bad_refs(
    pm: Postmortem, candidates: List[RootCauseCandidate], valid_ids: set
) -> set:
    violations: set = {r for r in _collect_refs(pm, candidates) if r not in valid_ids}
    for i, c in enumerate(pm.claims):
        if not c.evidence_refs:
            violations.add(f"claim[{i}] has empty evidence_refs")
    return violations


# ------------------------------------------------------------------- node: ingest
async def ingest_node(
    state: AgentState,
    *,
    llm: Any,
    tracer: Any,
    conn: Any,
    collection: Any,
) -> Dict[str, Any]:
    incident_id = state["incident_id"]
    init_db(conn)
    insert_incident(
        conn,
        {
            "id": incident_id,
            "window_start": state["window_start"],
            "window_end": state["window_end"],
            "description": state["description"],
        },
        status="running",
    )
    insert_evidence(conn, incident_id, [e.model_dump() for e in state["evidence"]])

    recall_task = None
    response = "<skipped: no memory collection>"
    if collection is not None:
        recall_task = asyncio.create_task(
            recall_incidents(state["description"], collection, n=5)
        )
        response = "<async recall task fired>"

    tracer.record(
        "ingest",
        system_prompt=_SYSTEM_INGEST,
        user_prompt="",
        input_state={
            "incident_id": incident_id,
            "evidence_ids": [e.id for e in state["evidence"]],
        },
        tool_calls=[
            {
                "tool": "recall_incidents",
                "args": {"symptom_text": state["description"], "n": 5},
                "response": response,
            }
        ],
        output={"incident_id": incident_id, "recall_fired": recall_task is not None},
    )
    return {"recall_task": recall_task}


# ----------------------------------------------------------------- node: timeline
async def timeline_node(
    state: AgentState,
    *,
    llm: Any,
    tracer: Any,
    conn: Any,
    collection: Any,
) -> Dict[str, Any]:
    incident_id = state["incident_id"]
    evidence = query_evidence(conn, incident_id)
    user = (
        f"Incident {incident_id}: {state['description']}\n\n"
        f"Evidence:\n{_evidence_block(evidence)}"
    )
    result = await llm.chat(
        _SYSTEM_TIMELINE,
        [{"role": "user", "content": user}],
        schema=List[TimelineEvent],
    )
    tracer.record(
        "timeline",
        system_prompt=_SYSTEM_TIMELINE,
        user_prompt=user,
        input_state={"evidence_ids": [e["id"] for e in evidence]},
        output=result,
    )
    return {"timeline_events": result}


# ----------------------------------------------------------------- node: analyze
async def analyze_node(
    state: AgentState,
    *,
    llm: Any,
    tracer: Any,
    conn: Any,
    collection: Any,
) -> Dict[str, Any]:
    incident_id = state["incident_id"]
    evidence = query_evidence(conn, incident_id)
    consulted: List[ConsultedIncident] = []
    if state.get("recall_task") is not None:
        consulted = await state["recall_task"]

    consulted_text = "\n".join(
        f"- {c.incident_id} (score {c.similarity_score}): {c.note}" for c in consulted
    ) or "(none)"
    user = (
        f"Incident {incident_id}: {state['description']}\n\n"
        f"Evidence:\n{_evidence_block(evidence)}\n\n"
        f"Consulted prior incidents (hypotheses ONLY, never cite as evidence):\n"
        f"{consulted_text}"
    )
    result = await llm.chat(
        _SYSTEM_ANALYZE,
        [{"role": "user", "content": user}],
        schema=List[RootCauseCandidate],
    )
    tracer.record(
        "analyze",
        system_prompt=_SYSTEM_ANALYZE,
        user_prompt=user,
        input_state={
            "evidence_ids": [e["id"] for e in evidence],
            "consulted": [c.incident_id for c in consulted],
        },
        output=result,
    )
    return {"candidates": result, "consulted": consulted}


# ------------------------------------------------------------------ node: writer
async def writer_node(
    state: AgentState,
    *,
    llm: Any,
    tracer: Any,
    conn: Any,
    collection: Any,
) -> Dict[str, Any]:
    incident_id = state["incident_id"]
    timeline = state["timeline_events"]
    candidates = state["candidates"]
    consulted = state.get("consulted", [])
    valid_ids = valid_evidence_ids(conn, incident_id)

    timeline_text = "\n".join(
        f"- {t.ts}: {t.description} (refs {t.evidence_refs})" for t in timeline
    )
    user = (
        f"Incident {incident_id}: {state['description']}\n\n"
        f"Timeline:\n{timeline_text}\n\n"
        f"Root-cause candidates:\n"
        + "\n".join(
            f"- {c.rank}. {c.hypothesis} (label={c.root_cause_label}, "
            f"support={c.supporting_evidence}, contradict={c.contradicting_evidence})"
            for c in candidates
        )
        + f"\n\nConsulted prior incidents (hypotheses ONLY): "
        f"{[c.incident_id for c in consulted] or '(none)'}"
    )

    pm = await llm.chat(
        _SYSTEM_WRITER,
        [{"role": "user", "content": user}],
        schema=Postmortem,
    )

    retries: List[Dict[str, Any]] = []
    bad = _bad_refs(pm, candidates, valid_ids)
    if bad:
        re_prompt = (
            user + "\n\nREJECTED — citation violations detected: "
            f"{sorted(bad)}. Every Claim.evidence_refs (and candidate evidence) "
            f"must be non-empty and use only valid ids from {sorted(valid_ids)}. "
            "Revise so each claim and candidate cites at least one valid evidence id."
        )
        pm = await llm.chat(
            _SYSTEM_WRITER,
            [{"role": "user", "content": re_prompt}],
            schema=Postmortem,
        )
        retries.append(
            {
                "reason": f"citation violations {sorted(bad)}",
                "re_prompt": re_prompt,
                "result": "ok",
            }
        )

    # Ensure consulted incidents + incident_id are attached regardless of LLM output.
    pm = pm.model_copy(
        update={"consulted_incidents": consulted, "incident_id": incident_id}
    )

    tracer.record(
        "writer",
        system_prompt=_SYSTEM_WRITER,
        user_prompt=user,
        input_state={
            "valid_ids": sorted(valid_ids),
            "consulted": [c.incident_id for c in consulted],
        },
        retries=retries,
        output=pm,
    )
    return {"postmortem": pm}


# ----------------------------------------------------------------- node: verifier
async def verifier_node(
    state: AgentState,
    *,
    llm: Any,
    tracer: Any,
    conn: Any,
    collection: Any,
) -> Dict[str, Any]:
    incident_id = state["incident_id"]
    pm = state["postmortem"]
    consulted = state.get("consulted", [])
    recalled_ids = [c.incident_id for c in consulted]

    report = verify_postmortem(
        conn, incident_id, pm, recalled_incident_ids=recalled_ids
    )

    valid_ids = valid_evidence_ids(conn, incident_id)
    verifier_math = [
        {
            "claim_index": r.claim_index,
            "claimed_refs": r.evidence_refs,
            "valid_ids": sorted(valid_ids),
            "from_recalled_incident": r.from_recalled_incident,
            "backed": r.backed,
            "missing_evidence": r.missing_evidence,
        }
        for r in report.claim_reports
    ]

    tracer.record(
        "verifier",
        system_prompt="verifier node (deterministic SQLite set-check, no LLM)",
        user_prompt="",
        input_state={"incident_id": incident_id, "recalled_ids": recalled_ids},
        verifier_math=verifier_math,
        output=report,
    )
    return {"verification": report}


# --------------------------------------------------------------- node: human_gate
async def human_gate_node(
    state: AgentState,
    *,
    llm: Any,
    tracer: Any,
    conn: Any,
    collection: Any,
) -> Dict[str, Any]:
    incident_id = state["incident_id"]
    pm = state["postmortem"]
    report = state["verification"]
    consulted = state.get("consulted", [])
    applied = state.get("applied_incidents", [])

    # Apply the human's applied/dismissed flips to consulted incidents.
    if applied:
        consulted = [
            c.model_copy(update={"applied": True})
            if c.incident_id in applied
            else c
            for c in consulted
        ]

    # Safety: do NOT auto-approve a postmortem whose verification failed or that
    # leaks a recalled incident id into claim evidence (consult-only ban).
    recalled = {c.incident_id for c in consulted}
    leak = any(
        c.from_recalled_incident is not None or (set(c.evidence_refs) & recalled)
        for c in pm.claims
    )
    approved = report.verification_score >= 1.0 and not leak
    decision = "approved" if approved else "rejected"
    status = "approved" if approved else "rejected"
    approved_by = "auto-eval" if approved else None
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    upsert_postmortem(
        conn,
        incident_id,
        draft_json=pm.model_dump_json(),
        verification_json=report.model_dump_json(),
        consulted_json=json.dumps(
            [c.model_dump() for c in consulted], ensure_ascii=False
        ),
        status=status,
        approved_by=approved_by,
        time_approved=now if approved else None,
    )

    tracer.record(
        "human_gate",
        system_prompt="human gate (simulated auto-approve in eval; real CLI in live use)",
        user_prompt="",
        input_state={"incident_id": incident_id, "applied_incidents": applied},
        human_decision=decision,
        output={"status": status, "approved_by": approved_by},
    )
    return {
        "human_decision": decision,
        "applied_incidents": applied,
        "consulted": consulted,
    }


# ----------------------------------------------------------- node: memory_writer
async def memory_writer_node(
    state: AgentState,
    *,
    llm: Any,
    tracer: Any,
    conn: Any,
    collection: Any,
) -> Dict[str, Any]:
    if collection is None:
        return {}
    pm = state.get("postmortem")
    if pm is None:
        return {}

    # Hard consult-only ban (PRD §5.2): leaks must NEVER be embedded, even though
    # the gate already rejected them. Everything else is embedded so recall can
    # surface a mix of good and imperfect priors.
    recalled = {c.incident_id for c in state.get("consulted", [])}
    if is_consult_only_leak(pm, recalled):
        tracer.record(
            "memory_writer",
            system_prompt="memory_writer node (embed postmortems into incident_memory)",
            user_prompt="",
            input_state={"incident_id": pm.incident_id, "skipped": "consult-only leak"},
            output={"embedded": False, "reason": "consult-only leak"},
        )
        return {}

    verification = state.get("verification")
    score = verification.verification_score if verification is not None else None
    await store_memory(pm, collection, verification_score=score)

    tracer.record(
        "memory_writer",
        system_prompt="memory_writer node (embed postmortems into incident_memory)",
        user_prompt="",
        input_state={
            "incident_id": pm.incident_id,
            "verification_score": score,
        },
        tool_calls=[
            {
                "tool": "store_memory",
                "args": {"incident_id": pm.incident_id, "verification_score": score},
                "response": "embedded into incident_memory",
            }
        ],
        output={"embedded": True},
    )
    return {}
