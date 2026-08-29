# Agentic Incident Postmortem Synthesizer — E2E Build Plan

## A. Decisions applied to the PRD
- **`llm_adapter.py`** is endpoint-agnostic: reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`
  (default `https://api.openai.com/v1`), `MODEL` (default `gpt-4o-mini`) from env/CLI.
  Works with Ollama, vLLM, LiteLLM, OpenRouter, etc. Replay/fallback logic unchanged.
- **Human gate = CLI only** (`python api.py approve <id> [--apply INC-014]` / `reject <id>`).
  No FastAPI server. A tiny static HTML is optional, not in scope.
- **Adapters** are synthetic, interface-shaped stubs fed by `generate_incidents.py`. No real
  Prometheus/Datadog/Slack code.

## B. File structure (scaffold order)
```
context/PRD.md                (given)
generate_incidents.py         (synthetic incidents + ground-truth labels)
requirements.txt              (fastapi NOT required now; langgraph, openai, chromadb, pydantic, pytest)
schemas.py                    (Pydantic models)
store.py                      (SQLite: incident/evidence/postmortem/verification + query_evidence + set_check)
rag.py                        (Chroma incident_memory)
llm_adapter.py                (OpenAI-compatible client + record/replay + embedding fallback)
tracer.py                     (per-agent trajectory JSON)
memory.py                     (recall_incidents + embed_approved; consult-only enforcement)
agents.py                     (async timeline/analyze/writer/verifier/human_gate/memory_writer nodes)
graph.py                      (LangGraph StateGraph w/ Send fan-out)
api.py                        (CLI: run, approve, reject, show)
eval.py                       (pytest: baseline vs agent, offline replay)
fixtures/llm_cache.jsonl      (committed; keyless eval)
trajectories/                 (judge evidence pack; committed sample)
README.md
tests/
  test_store.py
  test_schemas.py
  test_rag.py
  test_llm_adapter.py
  test_agents.py
  test_graph_integration.py
  test_eval.py
```

## C. Build phases (each ends with green tests)

### Phase 0 — Project setup
- `requirements.txt`, `pytest.ini` (offline mode default), `.env.example`
  (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL`, `LIVE`).
- Test: `pytest --co` collects empty suite; env loads.

### Phase 1 — `generate_incidents.py` + `schemas.py`
- Emit 10+ incidents across `deploys/metrics/logs/chat`; each with `true_root_cause` and
  `red_herring` slugs. Include the showcase checkout-timeout case (14:02 deploy drops
  `payment_timeout_ms` 2000→200; 14:25 DB CPU spike is the red herring). Also emit the
  pre-seed incidents 1–3 as `incident_memory` docs.
- `schemas.py`: `Evidence, TimelineEvent, RootCauseCandidate, Claim, ConsultedIncident,
  Postmortem, VerificationReport` exactly per §6.
- Tests (`test_schemas.py`): every model rejects missing/extra fields; `Claim.evidence_refs`
  typed; `from_recalled_incident` optional; invalid JSON fails structured-output contract.

### Phase 2 — `store.py` (deterministic core)
- `init_db`, `insert_incident`, `insert_evidence`, `query_evidence(start,end)`,
  `upsert_postmortem`, `set_check(claim, incident_id)` →
  `backed = set(evidence_refs) ⊆ valid_ids AND from_recalled_incident is None`.
- `evaluation_run` table.
- Tests (`test_store.py`): ISO-ts `BETWEEN` ordering; `set_check` true/false cases incl.
  hallucinated `E99` → `backed=0`; recalled-ref claim → `backed=0`; ON DELETE CASCADE.

### Phase 3 — `llm_adapter.py` (OpenAI-compatible + replay)
- `LLMAdapter.chat(system, messages, model, params, schema)` and `.embed(texts)`.
- `request_hash = sha256(system+messages+model+params)`. Replay from `fixtures/llm_cache.jsonl`;
  if `LIVE` and miss → call configured endpoint, append. Embedding miss in replay → local
  fallback (`all-MiniLM-L6-v2` else hash bag-of-words).
- Tests (`test_llm_adapter.py`): hash determinism; replay returns stored response with no
  network (monkeypatch socket); fallback embedding shape/determinism; `OPENAI_BASE_URL` override
  honored; chat-miss-in-replay raises (forces `--live`).

### Phase 4 — `rag.py` + `memory.py` (consult-only)
- `rag.py`: single `incident_memory` collection; `embed_approved(postmortem)`;
  `recall(symptom_text, n=5)` → `ConsultedIncident` with `similarity=1-distance`.
- `memory.py`: `recall_incidents` async fired during ingest; `embed_approved` after approval;
  **hard rule**: recalled ids may never enter `evidence_refs` (enforced in `verifier` +
  pre-verify validation).
- Tests (`test_rag.py`): metadata scalar-only (comma-joined keywords, `""` not null); recall
  returns scores ∈ [0,1]; consult-only enforcement test — injecting a recalled id into
  `evidence_refs` is rejected.

### Phase 5 — `agents.py` + `tracer.py`
- Async nodes: `ingest` (writes SQLite, fires `recall_incidents` via `asyncio.create_task`),
  `timeline`, `analyze` (parallel via `Send`), `writer` (merge), `verifier` (pure set-check,
  logs math), `human_gate` (auto-approve in eval, real in CLI), `memory_writer`.
- **Citation-integrity guard (§6.2):** pre-verify step rejects any output whose refs ∉ valid ids;
  `writer` re-prompted once. Tested explicitly.
- `tracer.py`: writes `trajectories/{incident_id}/{agent}.json` with `system_prompt, user_prompt,
  input_state, tool_calls, retries, output, human_decision` + `manifest.json`. §B.4 compliant.
- Tests (`test_agents.py`): with a fake `LLMAdapter` returning fixed structured objects, each node
  returns correct schema; citation-integrity retry fires once on bad refs then passes; verifier
  computes `backed` deterministically; trajectory file written with all §B.4 fields.

### Phase 6 — `graph.py` (orchestration)
- `StateGraph(AgentState)`; `after_ingest` returns `[Send("timeline"), Send("analyze")]`;
  merge → writer → verifier → human_gate → memory_writer → END. All `await` LLM calls.
- Tests (`test_graph_integration.py`): run full graph on 1 incident in replay mode → assert
  `Postmortem` complete, `VerificationReport` present, `trajectories/` populated for all agents,
  no recalled id in `evidence_refs`, `postmortem` row status transitions correctly.

### Phase 7 — `api.py` (CLI) + Phase 8 — `eval.py` (judging-critical)
- `api.py`: `run <incident.json>` (ingest→verifier, returns id+draft+report pending),
  `approve <id> [--apply INC-014]`, `reject <id> <reason>`, `show <id>`.
- `eval.py` (pytest): pre-seed 1–3 to `incident_memory`; run 4–10 with **simulated auto-approve**;
  compare **Baseline A** (single prompt, same schema) vs **Agent** on:
  - Primary: verification pass rate (deterministic set-check).
  - Secondary: red-herring rejection (`candidates[0].root_cause_label != red_herring`, label-vs-truth).
  - Secondary: structural completeness (required sections present).
  - Writes `evaluation_run` rows; prints comparison table.
- Tests (`test_eval.py`): agent verification ≫ Baseline A; red-herring rate = 1.0 on fixture set;
  deterministic across two offline runs (same scores); `--live` path writes fixtures (skipped offline).

## D. Test matrix (summary)
| Area | Test file | Key assertions |
|---|---|---|
| Schemas | test_schemas | typed IO, rejects prose |
| Store/verifier | test_store | set-check math, hallucinated ref fails, recalled ref fails |
| Adapter | test_llm_adapter | hash, replay no-net, embedding fallback, base_url override |
| RAG/memory | test_rag | scalar metadata, consult-only ban |
| Agents | test_agents | node schemas, citation retry, trajectory §B.4 |
| Graph | test_graph_integration | full pipeline, trajectory pack, status flow |
| Eval | test_eval | agent > baseline, red-herring=1.0, deterministic |

## E. Reproducibility & submission checklist
- `pip install -r requirements.txt && python eval.py` → offline, no key, identical scores.
- `fixtures/llm_cache.jsonl` + `trajectories/` committed.
- README states: eval simulates the human checkpoint (auto-approve); real gate is CLI in live use;
  verifier measures citation integrity not semantic truth (pair with red-herring + completeness metrics).
- §11 changelog iterations buildable by toggling the verifier/agents/memory features and re-running `eval.py`.
