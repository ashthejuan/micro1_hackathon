# Reproduction Guide — Agentic Incident Postmortem Synthesizer

> For someone starting from a **clean environment** (no API key, no prior data). Every command below is exact; expected outputs are shown. Tested on Windows 11 + PowerShell 5.1 with Python 3.12, but portable to macOS/Linux (use `python3`/`pip3` there).

## 1. Prerequisites

| Requirement | Version used (verified) | Notes |
|---|---|---|
| Python | `3.12.3` (`python --version`) | 3.10+ works |
| pip | 24+ | |
| OS | Windows 11 / macOS / Linux | `Graph` async + SQLite + in-memory Chroma work everywhere |
| API key | **None required** | Offline harness uses `helpers.py:145` `FakeLLMAdapter` + `llm_adapter.py:325` hash-embed. Live mode (`--live`) is optional |
| Native deps | None | `chromadb` is optional — without it the in-memory cosine store `rag.py:97` is used automatically |

Tool versions (from `pip show` at submission):

```
pydantic==2.12.5  pytest==8.4.2  langgraph==1.0.1  langgraph-checkpoint==3.0.1
openai==2.43.0  chromadb==1.5.9  sentence-transformers==5.6.0  fastapi==0.115.14
```

`requirements.txt:1` pins `pydantic>=2.0, pytest>=8.0, langgraph>=0.2, openai>=1.0, chromadb>=0.5, sentence-transformers>=2.7`.

---

## 2. Setup (clean clone)

```powershell
git clone <your-repo-url> .
# or: unzip submission.zip ; cd "micro1 hackathon"

python -m venv .venv
.\.venv\Scripts\Activate.ps1     # on macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

# optional sanity:
python -c "import pydantic, langgraph, openai, chromadb; print('deps ok')"
```

**Data generation (idempotent):** fixtures are already committed (`incidents/*.json` + `incidents/memory_seed.json`). To regenerate:

```powershell
python generate_incidents.py --out incidents
# → Wrote 11 incidents + memory_seed to incidents
```

Each incident `incidents/INC-00n.json:22` has `id, window_start/end, description, true_root_cause, red_herring, evidence[E1..En]`. Seed `INC-001..003` are the `incident_memory` priors `generate_incidents.py:313`.

---

## 3. Run the solution (agent)

### 3.1 One incident via the full 7-node graph (eval-style, auto-gate)

The graph `graph.py:49` runs all 7 nodes (`ingest → timeline‖analyze → writer → verifier → human_gate → memory_writer`) and writes `trajectories/{id}/{agent}.json`.

```powershell
python graph.py --incident INC-001 --fake
# → incident=INC-001 decision=approved verification_score=1.0
# trajectories: trajectories/INC-001/{ingest,timeline,analyze,writer,verifier,human_gate,memory_writer}.json + manifest.json
```

Flags:

- `--fake` — `FakeLLMAdapter` (`helpers.py:145`) + hash-embed, **no network, deterministic** (recommended for judges)
- `--live` — `LLMAdapter(live=True)` (`llm_adapter.py:44`), needs `OPENAI_API_KEY` + `OPENAI_BASE_URL` (default `https://api.openai.com/v1`, `MODEL=gpt-4o-mini`)
- `--db :memory:` — ephemeral SQLite (default); `--db ./app.db` — persistent
- `--traj-dir trajectories` — trajectory output dir (default)

Examples:

```powershell
# another incident (isolated, seed INC-001..003 consulted as (none) on first run)
python graph.py --incident INC-004 --fake

# persistent DB + visible file
python graph.py --incident INC-001 --fake --db app.db
python -c "import sqlite3; print(list(sqlite3.connect('app.db').execute('select id,status from incident')))"
```

### 3.2 CLI human gate (real gate — pending_approval, no auto-approve)

`api.py:157` deliberately does **not** reuse the graph's auto-approve gate. `run` leaves `pending_approval`; `approve`/`reject` is the mandatory checkpoint (ground rules §04/§05).

```powershell
# 1) run → pending_approval (writes trajectories/…/ingest,timeline,analyze,writer,verifier)
python api.py run INC-001 --fake --db app.db
# → incident=INC-001 status=pending_approval verification_score=1.00 claims=2 trajectory=trajectories/INC-001

# 2) inspect
python api.py show INC-001 --db app.db
# → JSON { incident_id, status:"pending_approval", draft:{summary,claims,...}, verification:{verification_score,claim_reports}, consulted_incidents:[] }

# 3) approve (embeds to incident_memory unless consult-only leak; writes human_gate + memory_writer trajectories)
python api.py approve INC-001 --by "human" --fake --db app.db
# → incident=INC-001 status=approved approved_by=human embedded=True

# optional: mark a consulted prior as applied (only if cited with current evidence)
python api.py approve INC-004 --apply INC-002 --by "oncall@example.com" --fake --db app.db

# reject path
python api.py run INC-005 --fake --db app.db
python api.py reject INC-005 "too speculative" --db app.db
# → incident=INC-005 status=rejected

# idempotency guard
python api.py approve INC-001 --fake --db app.db
# → SystemExit: postmortem INC-001 is 'approved', not pending_approval
```

Backend consistency (Design Risk #2, `store.py:242`): `run --fake` uses hash-embed in-memory; `approve --fake` must match. If mismatched, CLI exits with `backend mismatch: run used --fake but approve was called with --live …`. When `approve` is called without a flag it inherits `run`'s backend.

Sanity: `api.py run` persists verification rows and asserts `count == len(claim_reports)` `api.py:223` before returning — a half-written draft is never reported as pending.

### 3.3 Inspect trajectories (deliverable §B.4)

```powershell
Get-ChildItem trajectories/INC-001 | ForEach-Object Name
# → ingest.json  timeline.json  analyze.json  writer.json  verifier.json  human_gate.json  memory_writer.json  manifest.json

Get-Content trajectories/INC-001/manifest.json
# → {"incident_id":"INC-001","agents":["ingest","timeline","analyze","writer","verifier","human_gate","memory_writer"]}

Get-Content trajectories/INC-001/verifier.json | ConvertFrom-Json | ForEach-Object verifier_math
# → per-claim {claimed_refs, valid_ids, from_recalled_incident, backed, missing_evidence}
```

See `deliverables/agent_trajectories.md` for the annotated walk-through.

---

## 4. Run the baseline (same incidents, same harness)

Baseline A `eval.py:100` is a deterministic stub that outputs the **same `Postmortem` schema** but blames `red_herring` and cites `E99`. It is **not** a separate script — the eval harness runs it alongside the agent on every incident so the comparison is apples-to-apples:

```powershell
python -c "from eval import _baseline_postmortem; from generate_incidents import generate_incidents; inc={i['id']:i for i in generate_incidents()}['INC-004']; pm=_baseline_postmortem(inc); print(pm.root_cause, pm.claims[0].evidence_refs)"
# → dns_resolution_failure  ['E99']  (actually load_balancer_flap is the red herring — baseline blames it)
```

To see baseline scores without the agent, run the harness and read the `baseline` row:

```powershell
python eval.py --fake 2>&1 | Select-String "baseline"
# → baseline           0.000        0.000         1.000
```

Baseline alone: `verification 0.0` (every claim misses `E99`), `red_herring 0.0` (top label == herring), `completeness 1.0` (sections present — the honest structural delta is verifiability, not completeness).

---

## 5. Evaluation (agent vs baseline — the comparison table)

### 5.1 Offline (no key) — **required for judges**

```powershell
python eval.py --fake
# or:  python eval.py              # default is --fake
# or:  pytest tests/test_eval.py -q
```

Output you should see (byte-identical across runs, `tests/test_eval.py:57`):

```
=== Agentic Postmortem — eval comparison (offline replay) ===
Model: fake
Seeds: ['INC-001', 'INC-002', 'INC-003']  Eval: ['INC-004', 'INC-005', 'INC-006', 'INC-007', 'INC-008', 'INC-009', 'INC-010', 'INC-011']
mode        verification  red_herring  completeness
----------------------------------------------------
agent              1.000        1.000         1.000
baseline           0.000        0.000         1.000
----------------------------------------------------
  INC-004: agent v=1.0 rh=1 | base v=0.0 rh=0
  INC-005: agent v=1.0 rh=1 | base v=0.0 rh=0
  INC-006: agent v=1.0 rh=1 | base v=0.0 rh=0
  INC-007: agent v=1.0 rh=1 | base v=0.0 rh=0
  INC-008: agent v=1.0 rh=1 | base v=0.0 rh=0
  INC-009: agent v=1.0 rh=1 | base v=0.0 rh=0
  INC-010: agent v=1.0 rh=1 | base v=0.0 rh=0
  INC-011: agent v=1.0 rh=1 | base v=0.0 rh=0
Note: eval simulates human checkpoint (auto-approve); real gate is CLI `api.py approve` in live use.
Verifier measures citation integrity not semantic truth (paired with red-herring + completeness metrics).
Recall is isolated per incident (seed priors only, no sibling feedback loop).
```

The harness also writes `evaluation_run` rows (16 rows = 8 × 2 modes) when `db_path != :memory:`:

```powershell
python eval.py --fake --db app.db
python -c "import sqlite3; c=sqlite3.connect('app.db'); print(list(c.execute('select mode,incident_id,verification_score,red_herring_correct from evaluation_run order by mode,incident_id')))"
```

Isolation flag:

```powershell
python eval.py --fake                   # isolate=True (default) — each eval incident gets a fresh clone of seeds 1-3
python eval.py --fake --no-isolate     # cumulative feedback-loop mode (siblings recall each other) — for ablation only
```

### 5.2 Live (optional — refreshes fixtures)

```powershell
# needs OPENAI_API_KEY (and optionally OPENAI_BASE_URL, MODEL, EMBEDDINGS_MODEL)
$env:OPENAI_API_KEY="sk-..."
$env:MODEL="gpt-4o-mini"
python eval.py --live --db app.db
# → hits OPENAI_BASE_URL, appends chat+embeddings to fixtures/llm_cache.jsonl, prints same table shape
# Without key/fixture, eval.py --live exits 0 with:
#   skipped: no key/fixture — no OPENAI_API_KEY and no fixtures/llm_cache.jsonl; run without --live for offline demo
# (guard eval.py:344, tested by tests/test_eval.py:129)
```

### 5.3 Full test suite

```powershell
pytest -q
# →  ~40 tests, all green (test_api, test_eval, test_agents, test_graph_integration, test_rag, test_store, ...)

# single harness test (the judging-critical one):
pytest tests/test_eval.py::test_eval_agent_beats_baseline_fake -q
```

---

## 6. What data is required & what output to expect

| Step | Required data | Produces | Where to look |
|---|---|---|---|
| `generate_incidents.py` | none | `incidents/*.json` + `memory_seed.json` | `incidents/` |
| `graph.py --fake` / `api.py run --fake` | `incidents/INC-*.json` | SQLite `incident`+`evidence`+`postmortem`+`verification` rows + `trajectories/{id}/` | `trajectories/` + `app.db` (if not `:memory:`) |
| `eval.py --fake` | `incidents/` | comparison table above + `evaluation_run` rows (when `--db` file) + `trajectories/INC-00{4..11}/` | stdout + `trajectories/` |
| No external data, no credentials | Synthetic incidents are synthetic & committed; no private data | All outputs are local files | |

Typical trace for a single incident `INC-001` (offline):

- `trajectories/INC-001/ingest.json` — `evidence_ids:[E1..E8]`, `tool_calls:[recall_incidents{n:5}]`
- `timeline.json` — 2-3 `TimelineEvent{ts, description, evidence_refs}` (strictly `E1..E8`)
- `analyze.json` — `RootCauseCandidate{rank:1, root_cause_label:config_timeout_drop, supporting:[E1,E2], contradicting:[E5]}` + `consulted:[INC-002,…]` when applicable
- `writer.json` — `Postmortem{summary, impact, root_cause, timeline, action_items, claims[{evidence_refs:[E1,E4]}, {evidence_refs:[E5]}]}` (+ `retries` if `E99` was rejected)
- `verifier.json` — `verification_score:1.0`, `claim_reports[{backed:true, missing:[]}]`, `verifier_math`
- `human_gate.json` — `human_decision:"approved"/"rejected"` + `applied_incidents`
- `memory_writer.json` — `embedded:true/false` (false only on consult-only leak)

---

## 7. Versions, runtime & cost

### Versions

As printed by `pip show` at submission (see §1). Key transitive: `pydantic-core==2.33.2`, `chromadb==1.5.9` pulls `onnxruntime`, `grpcio` etc. — not needed for `--fake` (in-memory path `rag.py:97` has no native dep).

### Runtime (measured, `Python 3.12.3` on Windows 11, `pip install` already done)

| Command | Approx. wall clock | Notes |
|---|---|---|
| `pip install -r requirements.txt` (first time) | 30-90 s | `sentence-transformers`/`chromadb` are the heavy deps; offline `--fake` does not import their models |
| `python generate_incidents.py` | <0.2 s | |
| `python graph.py --incident INC-001 --fake` | 0.6-1.2 s | 2 parallel LLM stubs + SQLite + trace writes |
| `python api.py run INC-001 --fake --db app.db` | 0.6-1.0 s | same as graph, no auto-approve |
| `python eval.py --fake` (8+8 runs) | 3-6 s | `pytest tests/test_eval.py::test_eval_agent_beats_baseline_fake` is ~4.8 s |
| `pytest -q` (full suite) | 8-15 s | |

### Cost

| Mode | LLM | Cost |
|---|---|---|
| `--fake` (offline, default) | `FakeLLMAdapter` + hash-embed — no network | **$0** |
| `--live` | `MODEL=gpt-4o-mini` chat + `EMBEDDINGS_MODEL=text-embedding-3-small` | ~**$0.01-0.03 / incident** (3 chat calls + 1 embed per incident) — not required for judging |

---

## 8. Troubleshooting

- **`ModuleNotFoundError: chromadb`** — normal without `chromadb` installed; `rag.py:265` auto-falls back to `InMemoryMemory`. To use Chroma, `pip install chromadb`.
- **`CacheMissError: No replay fixture … and LIVE is off`** — you ran `LLMAdapter(live=False)` without `--fake` and without `fixtures/llm_cache.jsonl`. Use `graph.py --fake` or `eval.py --fake` (no fixture needed) or run once with `--live` + key to record the fixture.
- **`incident INC-XYZ not found`** — run `python generate_incidents.py --out incidents` or check `incidents/INC-XYZ.json` exists. Incident id must match `^[A-Za-z0-9._-]+$` `graph.py:91`.
- **`backend mismatch: run used --fake …`** — `api.py:310` guard. Re-run `approve` with the same flag as `run` (`--fake` if `run --fake`).
- **`evaluation_run` empty** — you used `--db :memory:` (default for `tests`). Pass a file path `--db app.db` to persist rows.
- **`sentence-transformers` import fails** — offline fallback `llm_adapter.py:342` handles it; set `USE_ST_EMBED=0` (default) to force the hash-embed path.

---

## 9. Minimal end-to-end (copy-paste)

```powershell
pip install -r requirements.txt
python generate_incidents.py --out incidents          # optional (already committed)

python eval.py --fake                                 # the judging-critical comparison (offline, deterministic)

python graph.py --incident INC-001 --fake            # single-incident agent run + trajectories

python api.py run INC-004 --fake --db app.db         # real human gate demo
python api.py show INC-004 --db app.db
python api.py approve INC-004 --by "judge@example.com" --fake --db app.db
python api.py show INC-004 --db app.db               # now status=approved, embedded=True

pytest -q                                             # all green
```

*Every claim about results is tied to evidence you can re-run: `eval.py:308` prints the table, `tests/test_eval.py:30` asserts `agent 1.0 > baseline 0.0 + 0.5`, `trajectories/` carries the §B.4 evidence pack, and `evaluation_run` rows persist the scores.*
