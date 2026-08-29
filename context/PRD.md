# PRD — Agentic Incident Postmortem Synthesizer

**Project:** micro1 hackathon ("Agentic Workflows")
**Author:** Akshit Dasgupta
**Date:** 2026-08-29
**Status:** Draft v2 (review revisions 1,2,4,5,6 applied)

---

## 0. TL;DR

An agentic system that, given an incident (time window + short description), autonomously ingests
evidence from an observability stack, reconstructs a verified timeline, proposes root cause(s), drafts a
publishable postmortem, **recalls similar past incidents as consulted hypotheses**, and routes a
human-approval gate before anything is published. The centerpiece is a **deterministic claim verifier**
that demands ≥1 evidence reference per factual assertion via a set-check against SQLite (no flaky LLM
judgement); the differentiator vs a plain prompt is that *every claim is signed by evidence* and the agent
*learns from prior incidents* without treating them as fact.

**Stack:** FastAPI (thin demo layer; CLI gate also valid) + SQLite (evidence store + deterministic
verifier) + ChromaDB (single `incident_memory` collection) + LangGraph (async agent orchestration) +
OpenAI (LLM, structured outputs) + Pydantic (all agent/IO schemas) + pytest (eval, **runs offline** via a
recorded-LLM replay adapter).

---

## 1. Goals

1. Produce a **citable, publishable** postmortem — not an obvious AI draft.
2. Verify **every factual claim** via a deterministic set-check against raw evidence; kill unverifiable
   claims before human review.
3. Consult **past incidents as hypotheses** (memory), surfaced with similarity scores, never auto-injected
   as fact.
4. Run agents with **async orchestration** (task-level fan-out + async I/O) — valued for *structured,
   independently-auditable causal reasoning*, not wall-clock speed.
5. Carry a **fair baseline** (single-prompt) and an **improvement changelog** with measured deltas.
6. Be **fully reproducible from a clean environment with no API key** (recorded-LLM replay), using
   synthetic incident data (no private data).

## 2. Non-Goals

- Real integration with production Prometheus/Datadog/Slack (synthetic adapters only; adapters are
  interface-shaped so real ones drop in later).
- Auto-publishing or any write/rollback action against live systems (violates hackathon ground rules).
- Multi-tenant auth, billing, persistence beyond the eval harness.

## 2.1 Hackathon deliverables compliance (traceability)

Maps each spec requirement to a PRD section so nothing is missed:

| Spec requirement | PRD coverage |
|---|---|
| Agentic workflow solving a real problem | §0–§3 (incident postmortems) |
| Purposeful agent design (context/tools/memory/verification/orchestration) | §6 agents, §5 memory, §4 verifier, §7 orchestration |
| Fair baseline + improvement changelog | §10 baseline, §11 changelog |
| Measured improvement + reproducibility | §9.1 replay adapter, §10 metrics |
| **§B.4 — representative trajectories per agent (instructions→result, tool responses, retries, human checkpoints)** | **§6.1 (BLOCKER — implemented via `tracer.py` + `trajectories/`)** |
| Human approval checkpoint | §8 human_gate (simulated in eval, real in live use) |
| Ground rules (sandbox, no private data, consult-only memory) | §2 Non-Goals, §5.2 consult-only |

---

## 3. Architecture Overview

```
                         ┌────────────────────────────────────────────┐
  Incident trigger  ───► │  FastAPI (thin; POST /incident, /approve)   │
  (or CLI)               └───────────────┬────────────────────────────┘
                                         │ invokes (async)
                                         ▼
                         ┌────────────────────────────────────────────┐
                         │  LangGraph StateGraph  (async runtime)      │
                         │                                              │
                         │   ingest ──► evidence store (SQLite)         │
                         │      └─► memory recall (Chroma, async)       │
                         │                                              │
                         │   ┌── timeline agent  ┐  (FAN-OUT, parallel)│
                         │   └── analyze agent   ┘                     │
                         │            │ merge (state)                   │
                         │            ▼                                 │
                         │        writer agent   ◄── recalled incidents│
                         │            │  (consulted hypotheses)         │
                         │            ▼                                 │
                         │       verifier agent  (DETERMINISTIC       │
                         │                      SQLite set-check)       │
                         │            │                                 │
                         │            ▼                                 │
                         │       human_gate  (approval required)       │
                         │            │                                 │
                         │            ▼                                 │
                         │       memory_writer (embeds approved pm)     │
                         └────────────────────────────────────────────┘
                                         │
                  ┌──────────────────────┼───────────────────────┐
                  ▼                      ▼                       ▼
              SQLite (evidence +       ChromaDB (incident_memory   pytest eval harness
              verification tables)     collection)                (offline via replay)
```

**Async rule (requirement):** agents use a real async runtime (fan-out + `await` I/O), but the win we
*sell* is **structured, independently-auditable causal reasoning**, not speed. `timeline` and `analyze`
reason from the same read-only evidence base as two separate agents and merge — that separation is what
makes the causality auditable and lets the analyzer explicitly reject red herrings. The wall-clock gain
from running two LLM calls in parallel is small and is **not** a changelog claim.
- **Task-level parallelism:** `timeline` and `analyze` run as a **parallel fan-out** after ingestion
  (both consume the evidence store, neither needs the other's output). Merged before `writer`.
- **Async I/O:** every LLM call uses `await llm.ainvoke(...)` / `await llm.with_structured_output(...).ainvoke(...)`
  so multiple agents and the memory recall overlap on the event loop rather than blocking serially.
- The **incident-memory recall** is fired asynchronously *during* ingestion so it is ready before `writer`
  without extending the critical path.

---

## 4. Data Model (SQLite — canonical store)

Tables (all evidence is canonical in SQLite; Chroma holds only the `incident_memory` embeddings):

```sql
CREATE TABLE evidence (
    id          TEXT PRIMARY KEY,        -- E1, E2, ...
    incident_id TEXT NOT NULL,
    ts          TEXT NOT NULL,           -- ISO-8601
    source      TEXT NOT NULL,           -- deploys | metrics | logs | chat
    source_url  TEXT,                    -- provenance link
    content     TEXT NOT NULL
);
CREATE INDEX idx_ev_ts ON evidence(incident_id, ts);

CREATE TABLE incident (
    id          TEXT PRIMARY KEY,
    window_start TEXT,
    window_end   TEXT,
    description  TEXT,
    status       TEXT,                    -- running | pending_approval | approved | rejected
    time_created TEXT
);

CREATE TABLE postmortem (
    incident_id TEXT PRIMARY KEY,
    draft_json  TEXT,                     -- serialized Pydantic Postmortem
    verification_json TEXT,               -- serialized VerificationReport
    recalled_incidents TEXT,              -- serialized list of consulted hypotheses + scores
    approved_by  TEXT,
    time_approved TEXT
);
```

`query_evidence(start, end)` ⇒ `SELECT * FROM evidence WHERE incident_id=? AND ts BETWEEN ? AND ? ORDER BY ts`.

### 4.1 SQLite schema detail

```sql
-- 1. Incident lifecycle
CREATE TABLE incident (
    id           TEXT PRIMARY KEY,                 -- INC-001
    window_start TEXT NOT NULL,                     -- ISO-8601 (lexicographically sortable)
    window_end   TEXT NOT NULL,
    description  TEXT,
    status       TEXT NOT NULL DEFAULT 'running',  -- running|pending_approval|approved|rejected
    time_created TEXT NOT NULL
);

-- 2. Canonical evidence (source of truth for the verifier's set-check)
CREATE TABLE evidence (
    incident_id  TEXT NOT NULL REFERENCES incident(id) ON DELETE CASCADE,
    id           TEXT NOT NULL,                     -- E1, E2, ...
    ts           TEXT NOT NULL,                     -- ISO-8601
    source       TEXT NOT NULL,                     -- deploys|metrics|logs|chat
    source_url   TEXT,                              -- provenance link
    content      TEXT NOT NULL,
    PRIMARY KEY (incident_id, id)
);
CREATE INDEX idx_ev_ts     ON evidence(incident_id, ts);
CREATE INDEX idx_ev_source ON evidence(incident_id, source);

-- 3. Postmortem: draft is canonical (writer owns it)
CREATE TABLE postmortem (
    incident_id      TEXT PRIMARY KEY REFERENCES incident(id) ON DELETE CASCADE,
    draft_json       TEXT NOT NULL,                 -- serialized Pydantic Postmortem
    verification_json TEXT,                         -- serialized VerificationReport
    consulted_json   TEXT,                           -- serialized List[ConsultedIncident]
    status           TEXT NOT NULL DEFAULT 'pending_approval',
    approved_by      TEXT,
    time_approved    TEXT,
    time_created     TEXT NOT NULL
);

-- 4. Verification: verifier owns this (single writer -> no dual-source conflict)
--    The verifier is a PURE set-check; no LLM "contradiction" call.
CREATE TABLE verification (
    incident_id      TEXT NOT NULL REFERENCES incident(id) ON DELETE CASCADE,
    claim_index      INTEGER NOT NULL,
    statement        TEXT NOT NULL,
    evidence_refs    TEXT NOT NULL,                 -- JSON array of Evidence.id
    from_recalled    TEXT,                           -- recalled incident_id OR NULL
    backed           INTEGER NOT NULL,              -- 0/1  (set-check result)
    missing_evidence TEXT,                           -- JSON array
    PRIMARY KEY (incident_id, claim_index)
);
```

**Design notes:**
- `ts` is ISO-8601 **text** on purpose — string compare = chronological order, so `BETWEEN` works and
  `query_evidence(start,end)` is a plain indexed scan. No datetime parsing needed.
- Draft (writer-owned) is split from verification (verifier-owned) so there is **one writer per table** —
  no dual-source-of-truth problem, and the verifier's exact set-check (`evidence_refs ⊆ evidence ids`) is
  queryable for the eval instead of buried in JSON.
- Booleans are `INTEGER 0/1` (SQLite has no real bool). JSON columns are `TEXT`.
- **The verifier is deterministic:** `backed = 1` iff every id in `evidence_refs` exists in `evidence`
  for that incident AND `from_recalled IS NULL`. No LLM call → no flakiness, fully reproducible.

**Optional — reproducibility:** an eval results table so baseline vs agent scores persist and the
changelog can be regenerated from a clean environment:
```sql
CREATE TABLE evaluation_run (
    run_id       TEXT PRIMARY KEY,
    mode         TEXT NOT NULL,   -- baseline|agent
    model        TEXT,
    incident_id  TEXT,
    verification_score REAL,
    red_herring_correct INTEGER,  -- 0/1: did it avoid the planted cause?
    time_created TEXT
);
```

---

## 5. RAG Design (ChromaDB — single collection: `incident_memory`)

> **Revision (v2):** the earlier `evidence_chunks` Chroma collection was dropped. Claim verification is
> now a deterministic SQLite set-check (§4.1), so semantic evidence retrieval is no longer needed for
> scoring. Chroma is kept **only** for the genuine RAG need: recalling similar past incidents as consulted
> hypotheses. This removes a dependency and all evidence-embedding cost.

### 5.1 Collection `incident_memory` (Incident-memory RAG — consulted hypotheses only)
- **What:** one embedded doc per *approved* postmortem (summary + symptoms + root_cause + action_items).
- **Used by:** `analyze`/`writer` as **consulted hypotheses**.
- **Rule (hard):** recalled incidents are attached with their **similarity score** and labeled
  `CONSULT_ONLY`. The agent MAY use them to *motivate a hypothesis to check* but MUST cite the **current**
  incident's own evidence for any conclusion. Recalled incidents are **forbidden** from appearing as
  evidence references in `Claim.evidence_refs`.
- **Surfaced in output:** the final postmortem includes a `Similar past incidents consulted` section
  listing each recalled incident id, its score, and whether the on-call **applied** or **dismissed** it
  (default: dismissed unless explicitly applied with current-evidence backing).

### 5.2 Vector metadata detail (ChromaDB)

**Collection — `incident_memory`** (consulted hypotheses; recall across all incidents)

| Field | Value |
|---|---|
| **ID** | `incident_id` → `INC-014` |
| **Document** | `"{summary}\nSymptoms: {symptoms}\nRoot cause: {root_cause}\nAction items: {action_items}"` |
| **Metadata** | `incident_id`, `root_cause_label` (slug), `time_approved` (ISO), `action_item_count` (int), `symptom_keywords` (comma-joined string) |
| **Embedding** | OpenAI `text-embedding-3-small` over the document text |
| **Collection metadata** | `{ "embedding_model": "text-embedding-3-small", "purpose": "incident_recall_consult_only" }` |

Query: `collection.query(query_texts=[current_symptom_description], n_results=5)` → ids + cosine distance.
Distance → similarity = `1 - distance`. Each becomes `ConsultedIncident(incident_id, similarity_score,
applied=False)` — **`applied` stays false unless the human flips it in the gate**, and a recalled id can
never enter `evidence_refs`.

**Chroma gotchas to respect:**
1. **Metadata values must be scalar** — `str | int | float | bool`. No `None`, no nested dicts, no lists.
   So `symptom_keywords` is a **comma-joined string**, and missing `source_url` is `""` not `null`.
2. **IDs must be unique strings.** `incident_memory` uses `INC-014` (global per incident).
3. **Embedding model must be fixed & recorded** in collection metadata (`text-embedding-3-small`, matches
   the OpenAI key). If the index is rebuilt with a different model, old vectors are incompatible — the
   stored `embedding_model` lets you detect that. All embedding calls route through `llm_adapter` (§9.1),
   which records/replays against `fixtures/llm_cache.jsonl` and falls back to a local deterministic
   embedding when `LIVE` is unset — so the eval loop never needs a key (see §9.1).

Both the Chroma persistence and the recorded-LLM fixtures live under the repo so the eval is reproducible
from a clean clone.

---

## 6. Agent Design (Pydantic-typed; memory consulted)

All agent I/O are Pydantic models → `llm.with_structured_output(Model)`. The model cannot return prose;
it returns the schema or fails.

```python
from pydantic import BaseModel
from typing import List, Optional

class Evidence(BaseModel):
    id: str; ts: str; source: str; source_url: Optional[str]; content: str

class TimelineEvent(BaseModel):
    ts: str; description: str; evidence_refs: List[str]

class RootCauseCandidate(BaseModel):
    rank: int; confidence: float
    hypothesis: str
    root_cause_label: str              # slug vs fixture true_root_cause / red_herring (scoring key)
    supporting_evidence: List[str]
    contradicting_evidence: List[str]   # explicit red-herring rejection
    from_prior_incident: Optional[str] = None   # id if motivated by memory (NOT as fact)

class Postmortem(BaseModel):
    summary: str; impact: str; root_cause: str
    timeline: List[TimelineEvent]
    action_items: List[str]
    claims: List[Claim]                  # writer commits its assertions explicitly
    consulted_incidents: List[ConsultedIncident]

class Claim(BaseModel):                 # <-- the key move
    statement: str
    evidence_refs: List[str]            # MUST be non-empty & valid Evidence ids
    from_recalled_incident: Optional[str] = None  # must be None to count as verified

class ConsultedIncident(BaseModel):
    incident_id: str; similarity_score: float
    applied: bool; note: str            # on-call/human can flip applied; default dismissed

class VerificationReport(BaseModel):
    claim: str; backed: bool; missing_evidence: List[str]
    verification_score: float           # backed / total  (DETERMINISTIC set-check)
```

### Node responsibilities (async)
| Node | Input (async) | Output | Concurrency |
|---|---|---|---|
| `ingest` | incident window | evidence in SQLite + memory recall fired | sequential first; fires memory recall async |
| `timeline` | `query_evidence` | `List[TimelineEvent]` | **parallel** with analyze |
| `analyze` | `query_evidence` + consulted incidents | `List[RootCauseCandidate]` | **parallel** with timeline |
| `writer` | timeline + analyze + consulted incidents | `Postmortem` | after merge |
| `verifier` | `Postmortem` vs SQLite evidence set | `VerificationReport` | after writer; **pure set-check, no RAG** |
| `human_gate` | draft + report + consulted_incidents | approve / edit / reject | after verifier |
| `memory_writer` | approved `Postmortem` | embed into `incident_memory` | after approval |

### 6.1 Trajectory tracing (required deliverable — submission blocker, maps to hackathon §B.4 / Final Deliverables item 04)

The hackathon **Final Deliverables (item 04, spec §B.4)** require *"representative trajectories for every
agent … instructions → result, tool responses, feedback/retries, human checkpoints."* This is a **submission
blocker** — the tracing layer below is mandatory, not optional.

**Per-agent trajectory file:** `trajectories/{incident_id}/{agent}.json` (one file per agent per incident,
so judges can read each agent's path independently). Each event object records every §B.4 element:

| §B.4 required element | Captured as |
|---|---|
| Instructions | `system_prompt` + `user_prompt` (the exact prompt sent to the LLM) |
| → result | `output` (the typed Pydantic object returned) |
| Tool responses | `tool_calls: [{tool, args, response}]` for every tool the agent invoked |
| Feedback / retries | `retries: [{reason, re_prompt, result}]` — incl. citation-integrity re-prompts (§6.2) and any verify→fix loops |
| Human checkpoints | `human_decision` on `human_gate` (auto-approved in eval; human in live use) + `applied_incidents` |

- `verifier` also logs the explicit set-check math per claim (`claimed_refs` vs `valid_ids`, `backed`).
- A top-level `trajectories/{incident_id}/manifest.json` lists which agents ran and the file for each, so
  the judge has an index.
- These files are the **evidence pack** for judges — they can read exactly what each agent saw, called,
  and concluded, including retries and the human checkpoint. **Ship `trajectories/` in the repo (or a
  representative sample subset with a note).** This is mandatory.

**Schema (one event, per agent):**
```json
{
  "node": "analyze",
  "system_prompt": "...",
  "user_prompt": "...",
  "input_state": { "evidence_ids": ["E1","E2"], "consulted": ["INC-014"] },
  "tool_calls": [{ "tool": "query_evidence", "args": {"start":"14:00","end":"15:30"}, "response": "..." }],
  "retries": [{ "reason": "hallucinated ref E99", "re_prompt": "...", "result": "ok" }],
  "output": { "candidates": [ { "root_cause_label": "config_timeout_drop", "...": "..." } ] },
  "human_decision": null
}
```
(human_gate carries `"human_decision": "approved", "applied_incidents": []`.)

### 6.2 Citation integrity (make-or-break for scoring)

The verification score is only meaningful if `evidence_refs` are **real supplied ids** (E1…En), never
hallucinated ones (E9). If the model drifts and emits `E99`, the set-check fails those claims and the
agent collapses toward Baseline A. Hard guards:
- **Instruction:** every agent that emits refs is told *"only reference ids from the supplied evidence
  list; never invent ids."*
- **Validation + retry:** before verification, a step rejects any `Claim`/`TimelineEvent`/
  `RootCauseCandidate` whose refs are not a subset of the incident's valid ids; the writer is re-prompted
  once with the offending refs listed. This keeps the set-check honest and the agent reliably above
  Baseline A.

---

## 7. Async Orchestration (LangGraph)

State is a `TypedDict` shared across nodes. Fan-out via the **`Send` API**, `await`-based LLM calls
throughout. The async design is justified by *structured causal reasoning* (two independent agents merge),
not by latency.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

async def after_ingest(state):
    # timeline and analyze are independent → schedule both in parallel
    return [Send("timeline", state), Send("analyze", state)]

builder = StateGraph(AgentState)
builder.add_node("ingest", ingest_node)
builder.add_node("timeline", timeline_node)
builder.add_node("analyze", analyze_node)
builder.add_node("writer", writer_node)
builder.add_node("verifier", verifier_node)
builder.add_node("human_gate", human_gate_node)
builder.add_node("memory_writer", memory_writer_node)

builder.add_edge(START, "ingest")
builder.add_conditional_edges("ingest", after_ingest)   # parallel fan-out
builder.add_edge(["timeline", "analyze"], "writer")     # merge
builder.add_edge("writer", "verifier")
builder.add_edge("verifier", "human_gate")
builder.add_edge("human_gate", "memory_writer")
builder.add_edge("memory_writer", END)
```

- LLM calls: `await model.with_structured_output(Postmortem).ainvoke(messages)`.
- Memory recall: `asyncio.create_task(recall_incidents(...))` inside `ingest`; `await` before `writer`.
- Verifier: **no RAG, no LLM** — `backed = set(claim.evidence_refs) ⊆ set(evidence_ids_for_incident)
  and claim.from_recalled_incident is None`. Computed in Python against SQLite.

---

## 8. API (FastAPI — thin demo layer)

> **Revision (v2):** FastAPI is a *thin* surface for live-demo polish. A **CLI `approve` command is
> equally valid** and requires no server — it satisfies the mandatory human-checkpoint ground rule just as
> well. Keep FastAPI only if you want the clickable demo (e.g. for the video).

| Method | Path | Body | Behavior |
|---|---|---|---|
| POST | `/incident` | `{window_start, window_end, description, synthetic_data?}` | runs ingest→...→verifier, returns `postmortem_id` + draft + report (pending approval) |
| GET | `/postmortem/{id}` | — | draft + verification report + consulted incidents w/ scores |
| POST | `/approve/{id}` | `{approved_by, edits?, applied_incidents?}` | human checkpoint; writes `postmortem` row; embeds into memory; **nothing publishes without this** |
| POST | `/reject/{id}` | `{reason}` | marks rejected |

Human gate UI (optional): a single static HTML page served by FastAPI showing draft + clickable evidence
links (E1…En) + verification report + consulted-incidents panel with apply/dismiss toggles. The CLI
alternative: `python api.py approve <id> [--apply INC-014]` reads the same draft from SQLite.

---

## 9. Synthetic Data & Reproducibility

- `generate_incidents.py` emits 10+ incidents as JSON across `deploys/metrics/logs/chat`, each with a
  **known root cause** and at least one **red herring** (e.g., the checkout-timeout case: 14:02 deploy
  dropped `payment_timeout_ms` 2000→200; 14:25 DB CPU spike is coincidental). **Each fixture carries
  ground-truth fields** `true_root_cause` (slug, e.g. `config_timeout_drop`) and `red_herring` (slug, e.g.
  `db_cpu_spike`) — these drive the reproducible red-herring rejection score (§10) and must be present on
  every incident so the eval is label-based, not judgement-based.
- `requirements.txt` + `README` setup → clean-env run:
  `pip install -r requirements.txt && python eval.py` — **runs offline, no API key needed.**
- No private data; all inputs synthetic; all actions sandboxed read-only.

### 9.1 Recorded / Replay LLM adapter (reproducibility fix — v2)

To satisfy the **Reproducibility (15)** criterion, the eval must run without an API key. An `LLMAdapter`
wraps the OpenAI client:

- On each call, compute `request_hash = sha256(system + messages + model + params)`.
- Look up `fixtures/llm_cache.jsonl` for a matching request. If found → **replay** the stored response
  (no network). This makes the eval **deterministic** (identical scores every run).
- If not found and `LIVE=1` (or `eval.py --live`), call the API and **append** `request→response` to the
  fixture.

**Embeddings — critical for keyless reproduction (the only real blocker):** the adapter wraps the OpenAI
**embeddings** endpoint too, not just chat completions. So `memory_writer`'s embed call during the eval
loop, and any in-loop `incident_memory` write, route through the same record/replay — they hit
`fixtures/llm_cache.jsonl`, not the network. **Fallback:** when `LIVE` is unset AND there is no fixture
hit, the adapter uses a **local deterministic embedding** (`sentence-transformers` `all-MiniLM-L6-v2` if
importable, else a hash-bucket bag-of-words vector) so the loop never touches the network. This removes the
last key dependency and the eval is keyless end-to-end.

Usage:
- `python eval.py` → default **replay/fallback** mode (offline, no key). Required for judge reproduction.
- `python eval.py --live` → refreshes both chat and embedding fixtures (requires `OPENAI_API_KEY`); commit
  `fixtures/llm_cache.jsonl` to the repo.

Caveats: fixtures must be committed; if a prompt or embedding input changes, the hash misses and you
re-run `--live` once. `incident_memory` is pre-seeded for incidents 1–3 and appended during the loop via
replay/fallback embeddings, so recall works fully offline.

---

## 10. Baseline & Evaluation (judging-critical)

- **Baseline A (single prompt):** feed all raw JSON to one prompt → "write a postmortem." This is the
  **automated fair comparison** and the primary demonstrator of the agent's delta.
  **Fairness note (must be stated to judges):** a single prose prompt cannot emit `Claim.evidence_refs`,
  so its *verification score is 0 by construction* — the delta is **structural** (typed claim objects +
  SQLite set-check), not the agent "being smarter." To make the comparison apples-to-apples, Baseline A is
  additionally prompted to output the **same `Postmortem` schema** (claims with `evidence_refs`); it still
  scores near 0 because it invents refs / blames the red herring, which is the honest point.
- **Baseline B (manual) — OPTIONAL, single case only:** one human-written reference postmortem for the
  showcase incident (the checkout-timeout case), used for a one-off "agent vs human" completeness
  narrative in the video. **Not** a 10-case labour item and **not** part of the automated eval loop —
  authoring 10 human write-ups is rejected as too much work and not reproducible by a third party.
- **Primary metric:** *Verification pass rate* = fraction of `Claim`s where `evidence_refs ⊆ evidence ids`
  and `from_recalled_incident IS NULL`. This is a **deterministic set-check** — no LLM judgement, so scores
  are stable. Expected: agent ≫ Baseline A (which invents numbers / blames red herrings).
- **Secondary — red-herring rejection rate (precisely defined, reproducible):** every fixture in
  `generate_incidents.py` carries ground-truth fields `true_root_cause` (a `root_cause_label` slug, e.g.
  `config_timeout_drop`) and `red_herring` (a slug, e.g. `db_cpu_spike`). `eval.py` reads the agent's
  **top `RootCauseCandidate`** (`candidates[0].hypothesis` mapped to its `root_cause_label`) and scores
  `red_herring_correct = 1` iff `candidates[0].root_cause_label != red_herring`. This is a string/label
  comparison against fixture truth — **not** a manual judgement — so it is reproducible. The agent wins
  because timeline/analyze reason from timestamps and explicitly reject the red herring via
  `contradicting_evidence`.
- **Secondary — structural completeness:** summary/impact/timeline/root_cause/action_items sections
  present (no human reference needed).
- `eval.py` (pytest) runs both on the same 10 incidents **in replay/fallback mode by default** (offline,
  no key), prints the comparison table, and writes rows to `evaluation_run`. The secondary completeness
  score is computed structurally (required sections present), not by matching a human reference, so no
  human write-up is needed for the 10-case loop.
- **Hot take:** "A deterministic verifier (Pydantic-typed outputs + a SQLite set-check) mattered more than
  model choice or LLM-based contradiction detection for trustworthy, reproducible output — with the honest
  caveat that verification measures *citation integrity*, not semantic truth; it is paired with the
  red-herring (label-vs-truth) and completeness metrics to cover the gap."

---

## 11. Improvement Changelog Plan (each tied to an eval delta)

**Memory warm-up sequencing (honest, required for §11.4):** the "+ incident-memory RAG" iteration can
only score if prior approved postmortems are already embedded. Define the eval run order explicitly:
1. **Pre-seed** incidents 1–3 into `incident_memory` (their embeddings recorded/replayed via `llm_adapter`
   and committed in `fixtures/`). These are the "prior incidents" the recall consults.
2. Run incidents 4–10 with recall **on**; for each, `human_gate` is **auto-approved by a simulated reviewer**
   (the eval loop calls `approve` with `applied_incidents=[]` by default) so `memory_writer` embeds the
   accepted postmortem and the next incident's recall has more context. No human is in the loop during eval.
3. Scoring (§10) runs on the 4–10 outputs (and 1–3 if also evaluated), populating `evaluation_run`.

This must be stated plainly in the README: **"eval.py simulates the human-approval checkpoint (auto-
approve) so the loop can populate incident_memory and produce comparable scores; the real gate is a human
in live use."** That is honest and does not violate the ground rule (the rule applies to production use;
the eval is a measurement harness).

1. **Baseline** — single prompt. (low verification, blames red herring)
2. **+ deterministic verifier** — SQLite set-check (`evidence_refs ⊆ evidence ids`) kills unverifiable
   claims; no LLM contradiction flakiness. *(verification ↑, fully reproducible)*
3. **+ timeline + analyze agents (async fan-out)** — independent causal reasoning from the same evidence
   base; explicit red-herring rejection; merge before writer. Sold as **structured causality**, not speed.
   *(red-herring rate ↑)*
4. **+ incident-memory RAG (consulted)** — recall on pre-seeded 1–3, run 4–10 with simulated auto-approve
   populating memory; recurrence detected, durable action items; shown as hypotheses. *(recurrence caught;
   still evidence-backed)*
5. **− writer few-shot** — if measurement shows no lift, drop it and document why. *(honest subtraction)*

---

## 12. Risks / Honest Failure Modes

- **Verifier checks citation integrity, NOT semantic truth (be explicit).** The set-check confirms a claim
  references *an existing supplied evidence id* — it does **not** confirm the cited evidence actually
  supports the claim semantically. A claim "DB CPU spike [E5] caused it" passes the set-check because E5
  exists, even though E5 is the wrong cause. The **red-herring rejection metric** (§10, label-vs-truth)
  catches this gross case; **structural completeness** catches missing sections. Verification alone is
  necessary but not sufficient — the three metrics are complementary. State this candor in the write-up;
  it strengthens the Hot Take rather than weakening the "trustworthy" claim.
- **Sparse memory space:** at 10–20 incidents, similarity can mislead. Mitigated by *consult-only* rule +
  scores shown + human can dismiss. Recalled incident ids are **banned** from `evidence_refs`.
- **Async race on shared state:** LangGraph state is immutable-per-node; fan-out reads evidence store
  (read-only) so no write race. Verifier/writer merge is explicit.
- **Structured-output reliability:** OpenAI `gpt-4o-mini`/`gpt-4o` chosen specifically for reliable
  tool-calling + JSON mode (this is why we dropped NIM).
- **Reproducibility without a key:** guaranteed by the recorded-LLM replay adapter (§9.1) + committed
  `incident_memory` embeddings. A judge runs `python eval.py` offline and gets identical scores.
- **Chroma footprint now minimal:** single `incident_memory` collection; evidence-embedding cost removed.

---

## 13. Proposed File Structure

```
micro1 hackathon/
├─ context/PRD.md                 (this file)
├─ generate_incidents.py          (synthetic incidents, incl. checkout-timeout)
├─ schemas.py                     (Pydantic: Evidence, Claim, Postmortem, ...)
├─ store.py                       (SQLite evidence store + query_evidence + set-check)
├─ rag.py                         (Chroma: incident_memory collection only)
├─ llm_adapter.py                 (recorded/replay LLM adapter; fixtures/llm_cache.jsonl)
├─ tracer.py                      (per-incident trajectory JSON: prompts, tools, retries, human checkpoint)
├─ agents.py                      (timeline/analyze/writer/verifier node functions, async)
├─ graph.py                       (LangGraph StateGraph w/ async fan-out)
├─ memory.py                      (recall_incidents + embed_approved, consult-only)
├─ api.py                         (FastAPI thin routes OR CLI approve; static gate HTML optional)
├─ eval.py                        (pytest baseline vs agent; OFFLINE replay by default)
├─ fixtures/llm_cache.jsonl      (committed; enables keyless eval)
├─ trajectories/                  (per-incident evidence pack for judges; REQUIRED deliverable)
├─ requirements.txt
└─ README.md
```

---

## 14. Open Questions (resolve before scaffold)

1. Model: `gpt-4o-mini` (cheap, fine for structured) vs `gpt-4o` (better reasoning on red herrings)?
   Recommend `gpt-4o-mini` for the pipeline, `gpt-4o` only if verification scores disappoint.
2. Human gate: thin FastAPI (recommended for demo polish) vs CLI `approve` (zero-dep, equally valid)?
3. Do we need real adapters stubbed now, or synthetic-only for the demo?

*(Baseline B resolved: single optional human postmortem for the showcase incident only — not a 10-case
labour item, not in the automated eval loop.)*
```
