# Agent Trajectories — Agentic Incident Postmortem Synthesizer

> Hackathon Final Deliverables item 04 / spec §B.4: *representative trajectories for **every** agent you used — instructions → result, tool responses, feedback/retries, human checkpoints.* Each agent run writes `trajectories/{incident_id}/{agent}.json` via `tracer.py:35` + `manifest.json`. This file is the readable index over that evidence pack.

## Graph overview

```
ingest ──┬──→ timeline ──┐
         │               ├──→ writer → verifier → human_gate → memory_writer
         └──→ analyze ──┘      (cites)   (set-check)  (checkpoint)  (embed)
              ↑ recall
```

- `graph.py:49` `build_graph()` binds 7 nodes `agents.py:27` as `StateGraph(AgentState)` with async fan-out `graph.py:62` `Send("timeline")/Send("analyze")` and fan-in `graph.py:65`.
- Every node is `async def node(state, *, llm, tracer, conn, collection) -> dict` `agents.py:9`, config passed via `functools.partial` closure `graph.py:42` so `AgentState:43` stays pure data (+ `recall_task: asyncio.Task` for overlap PRD §7).
- Trajectories are written by `tracer.py:46` `Tracer.record(node, system_prompt, user_prompt, input_state, tool_calls, retries, output, human_decision, verifier_math)` to `trajectories/{incident_id}/{node}.json:72` (JSON-safe via `model_dump()` `tracer.py:24`). `manifest.json:75` indexes `{incident_id, agents:[…]}`.

Representative eval run produces 8 incidents `INC-004..011`, each with 7 files. Committed sample: `trajectories/INC-001/` (seed incident, no prior consulted) + `INC-004..011` stubs. To regenerate the full pack:

```powershell
python graph.py --incident INC-001 --fake   # writes trajectories/INC-001/{ingest,timeline,analyze,writer,verifier,human_gate,memory_writer}.json
python eval.py --fake                       # writes trajectories/INC-00{4..11}/ (isolated per incident)
Get-Content trajectories/INC-001/manifest.json
# → {"incident_id":"INC-001","agents":["ingest","timeline","analyze","writer","verifier","human_gate","memory_writer"]}
```

---

## 1. ingest — writes evidence, fires memory recall

**Code:** `agents.py:109`, prompt `_SYSTEM_INGEST:60`

| Field | Value |
|---|---|
| system_prompt | `"ingest node (no LLM; writes evidence and fires memory recall)"` |
| input_state | `{"incident_id":"INC-001","evidence_ids":["E1","E2","E3","E4","E5","E6","E7","E8"]}` |
| tool | `recall_incidents` `memory.py:107` `rag.py:117` — `{"symptom_text":"Checkout failures and elevated payment errors…","n":5}` |
| response | `"<async recall task fired>"` (`agents.py:137`) — `asyncio.create_task(recall_incidents(...))` `agents.py:134`, awaited later in `analyze_node` `agents.py:202` |
| output | `{"incident_id":"INC-001","recall_fired":true}` |
| file | `trajectories/INC-001/ingest.json:1` |

Excerpt `trajectories/INC-001/ingest.json`:

```json
{
  "node": "ingest",
  "system_prompt": "ingest node (no LLM; writes evidence and fires memory recall)",
  "user_prompt": "",
  "input_state": {"incident_id": "INC-001", "evidence_ids": ["E1","E2","E3","E4","E5","E6","E7","E8"]},
  "tool_calls": [{"tool": "recall_incidents", "args": {"symptom_text": "Checkout failures and elevated payment errors after the 14:02 payment-service deploy.", "n": 5}, "response": "<async recall task fired>"}],
  "output": {"incident_id": "INC-001", "recall_fired": true}
}
```

Feedback: none (no LLM). Next: `timeline` + `analyze` fan-out `graph.py:62` runs in parallel via `asyncio.gather` `api.py:186`.

---

## 2. timeline — reconstructs chronology strictly from evidence

**Code:** `agents.py:160`, prompt `_SYSTEM_TIMELINE:61`

| Field | Value |
|---|---|
| system_prompt | `"You are the timeline agent. Reconstruct a chronological timeline strictly from the supplied evidence. Only reference evidence ids that appear in the evidence list; never invent ids."` |
| user_prompt | `"Incident INC-001: Checkout failures…\n\nEvidence:\nE1 [deploys 2026-08-20T14:02:00] payment-service v2.3.1 deployed; config change payment_timeout_ms 2000 -> 200\nE2 [metrics ...] checkout p99 800ms -> 4.2s\n…" ` (`agents.py:171`, built from `_evidence_block:80`) |
| input_state | `{"evidence_ids":["E1","E2","E3","E4","E5","E6","E7","E8"]}` |
| tool_calls | `[]` (reads `query_evidence` `store.py:126` via `agents.py:169` before LLM) |
| output | `List[TimelineEvent:28]` — e.g. `[{"ts":"2026-08-20T14:02:00","description":"payment-service v2.3.1 deployed; config change payment_timeout_ms 2000 -> 200","evidence_refs":["E1"]}, {"ts":"2026-08-20T14:03:30","description":"checkout p99 latency rose 800ms -> 4.2s","evidence_refs":["E2"]}]` |
| retries | `[]` (retry lives in writer, not timeline) |
| file | `trajectories/INC-001/timeline.json:2` |

The agent never invents ids — enforced by `_bad_refs:98` downstream + `helpers.py:170` `FakeLLMAdapter` returning only `E1`/`E2` for this incident.

---

## 3. analyze — proposes ranked candidates, explicitly rejects red herrings

**Code:** `agents.py:190`, prompt `_SYSTEM_ANALYZE:66`

| Field | Value |
|---|---|
| system_prompt | `"You are the root-cause analysis agent. Propose ranked root-cause candidates. Use contradicting_evidence to explicitly reject red herrings. You MAY be motivated by consulted prior incidents but MUST cite the CURRENT incident's evidence for every conclusion. Only reference current evidence ids; never invent ids."` |
| user_prompt | `"Incident INC-001: …\n\nEvidence:\nE1 …\n…E8 …\n\nConsulted prior incidents (hypotheses ONLY, never cite as evidence):\n(none)"` (`agents.py:207`, `consulted_text:205` or `"(none)"`) |
| input_state | `{"evidence_ids":["E1".. "E8"], "consulted":["INC-002"] or []}` — `consulted` comes from `await state["recall_task"]` `agents.py:202` |
| tool_calls | `[]` (consulted is the memory tool's result, surfaced in `input_state`) |
| output | `List[RootCauseCandidate:34]` — e.g. `[{"rank":1,"confidence":0.9,"hypothesis":"Root cause is config_timeout_drop","root_cause_label":"config_timeout_drop","supporting_evidence":["E1","E2"],"contradicting_evidence":["E8"],"from_prior_incident":null}]` (`helpers.py:184` incident-specific) |
| file | `trajectories/INC-001/analyze.json:3` |

In `INC-004` (`dns_resolution_failure` vs `load_balancer_flap`): `analyze.json:output[0]` has `root_cause_label:"dns_resolution_failure"`, `supporting:["E1","E3"]`, `contradicting:["E4"]` (`LB backend health 100%`). That `contradicting_evidence` is what scores `red_herring_correct=1` `eval.py:91`.

---

## 4. writer — produces publishable postmortem (citation guard with retry)

**Code:** `agents.py:232`, prompt `_SYSTEM_WRITER:72`, retry `agents.py:268`

| Field | Value |
|---|---|
| system_prompt | `"You are the postmortem writer. Produce a publishable postmortem. Every Claim must have non-empty evidence_refs that are REAL ids from the supplied evidence list only. Never invent ids. Set from_recalled_incident only when a claim is motivated by a consulted prior incident (it is then excluded from verification)."` |
| user_prompt | `"Incident INC-001: …\n\nTimeline:\n- 2026-08-20T14:02:00: payment-service v2.3.1 deployed… (refs ['E1'])\n- …\n\nRoot-cause candidates:\n- 1. Root cause is config_timeout_drop (label=config_timeout_drop, support=['E1','E2'], contradict=['E8'])\n\nConsulted prior incidents (hypotheses ONLY): (none)"` (`agents.py:249`) |
| input_state | `{"valid_ids":["E1".. "E8"],"consulted":[]}` |
| retries | `[]` in the happy path; on violation: `[{"reason":"citation violations ['E99']","re_prompt":"…Every Claim.evidence_refs … must be non-empty and use only valid ids from ['E1','E2',…]. Revise…","result":"ok"}]` — `agents.py:282` logs the offending `sorted(bad)` `agents.py:99`. Exercised by `FakeLLMAdapter(bad_postmortem_first=True)` `helpers.py:147` |
| output | `Postmortem:57` — `{"incident_id":"INC-001","summary":"Checkout failures after payment timeout drop.","impact":"5.3% checkout error rate…","root_cause":"config_timeout_drop","timeline":[…],"action_items":["Restore payment_timeout_ms to 2000."],"claims":[{"statement":"14:02 deploy dropped timeout causing failures","evidence_refs":["E1","E4"]}, {"statement":"DB CPU spike at 14:25 was coincidental","evidence_refs":["E5"]}],"consulted_incidents":[]}` `agents.py:291` ensures `consulted_incidents` + `incident_id` are attached regardless of LLM output |
| file | `trajectories/INC-001/writer.json:2` |

This is the **make-or-break** step for `verification 0 → 1.0` (see `improvement_changelog.md` Iter 2). The trajectory captures the re-prompt verbatim when it occurs, satisfying spec "retries / feedback that shaped next step."

---

## 5. verifier — deterministic SQLite set-check (no LLM)

**Code:** `agents.py:310`, math `store.py:146`

| Field | Value |
|---|---|
| system_prompt | `"verifier node (deterministic SQLite set-check, no LLM)"` `agents.py:352` |
| input_state | `{"incident_id":"INC-001","recalled_ids":[]}` |
| verifier_math | Per-claim `{claimed_refs, valid_ids, from_recalled_incident, backed, missing_evidence}` `agents.py:339` — e.g. `{"claim_index":0,"claimed_refs":["E1","E4"],"valid_ids":["E1","E2","E3","E4","E5","E6","E7","E8"],"from_recalled_incident":null,"backed":true,"missing_evidence":[]}` |
| output | `VerificationReport:68` — `{"incident_id":"INC-001","claim_reports":[{"claim_index":0,"statement":"14:02 deploy dropped timeout causing failures","evidence_refs":["E1","E4"],"from_recalled_incident":null,"backed":true,"missing_evidence":[]}, {"claim_index":1,…}],"verification_score":1.0}` `store.py:189` |
| scoring | `verify_postmortem` `store.py:164` calls `assert_consult_only` `memory.py:138` (hard ban: recalled id in `evidence_refs` → `ValueError`, caught as `verification_score=0.0` `agents.py:326`) then `set_check` `store.py:146` per claim: `backed = len(missing)==0 && len(refs)>0 && from_recalled is None` |
| file | `trajectories/INC-001/verifier.json:2` |

The verifier also covers the "every score traces to a file" reproducibility requirement — `verifier_math` is the explicit audit trail judges can read without running code.

---

## 6. human_gate — mandatory approval checkpoint

**Code:** `agents.py:363` (graph auto-gate) + `api.py:282` (real CLI gate)

| Variant | Behavior |
|---|---|
| **Eval (auto-approve)** `agents.py:363` | Approves iff `verification_score>=1.0 && no consult-only leak` `agents.py:389`; persists via `upsert_postmortem` `store.py:212` with `status:"approved"` + `time_approved`. Note printed in eval footer `eval.py:325`: "eval simulates human checkpoint; real gate is CLI `api.py approve`." |
| **CLI (real human)** `api.py:282` | `run` leaves `pending_approval` `api.py:205`; `approve`/`reject` `api.py:282`/`api.py:401` require a human. `approve --apply INC-014` flips `consulted_incidents[].applied=True` `api.py:334` + writes `human_gate` trajectory `api.py:361` |

| Field | Value |
|---|---|
| system_prompt | `"human gate (simulated auto-approve in eval; real CLI in live use)"` (`agents.py:413`) / `"human gate (CLI approve)"` (`api.py:363`) |
| input_state | `{"incident_id":"INC-001","applied_incidents":[]}` |
| human_decision | `"approved"` (or `"rejected"` after `verifier_score<1.0` or leak) `agents.py:394` |
| output | `{"status":"approved","approved_by":"auto-eval" or "human","time_approved":"2026-08-20T…","applied":[]}` |
| file | `trajectories/INC-001/human_gate.json:3` (`graph.py` path) / `api.py` writes its own on `approve`/`reject` |

Excerpt `trajectories/INC-001/human_gate.json`:

```json
{"node":"human_gate","system_prompt":"human gate (CLI approve)","user_prompt":"","input_state":{"incident_id":"INC-001","applied_incidents":[]},"human_decision":"approved","output":{"status":"approved","approved_by":"human","applied":[]}}
```

---

## 7. memory_writer — embeds approved postmortem into incident_memory (consult-only ban)

**Code:** `agents.py:427` + `memory.py:78` + `rag.py:106`

| Field | Value |
|---|---|
| system_prompt | `"memory_writer node (embed postmortems into incident_memory)"` `agents.py:448` |
| input_state | `{"incident_id":"INC-001","verification_score":1.0}` or `{"incident_id":"…","skipped":"consult-only leak"}` |
| tool_calls | `[{"tool":"store_memory","args":{"incident_id":"INC-001","verification_score":1.0},"response":"embedded into incident_memory"}]` `agents.py:467` |
| logic | `is_consult_only_leak` `memory.py:91` (`from_recalled_incident != None` OR `evidence_refs ∩ recalled_ids != ∅`) → `tracer.record:446` `embedded:false, reason:"consult-only leak"` and return `{}` — **never embedded**, even though gate already rejected. Otherwise `store_memory:82` → `embed_postmortem:51` (`generate_incidents.memory_doc` shape `generate_incidents.py:313`) → `collection.add` `rag.py:106` (scalar-metadata guard `rag.py:57`, duplicate-id guard). Quality `verification_score` stamped in metadata `memory.py:74` |
| file | `trajectories/INC-001/memory_writer.json:3` |

---

## End-to-end trajectory (one incident, INC-001 — no prior consulted)

```
manifest.json: {"incident_id":"INC-001","agents":["ingest","timeline","analyze","writer","verifier","human_gate","memory_writer"]}
ingest       tool recall_incidents n=5 → <async recall task fired>
timeline     LLM[List[TimelineEvent]] 2 events (E1,E2)
analyze      await recall → (none) → LLM[List[RootCauseCandidate]] rank1=config_timeout_drop contradict=[E8]
writer       LLM[Postmortem] claims [E1,E4],[E5] → valid (no retry)
verifier     set-check 2/2 backed → score=1.0
human_gate   decision=approved (score>=1 && no leak)
memory_writer embed Postmortem → incident_memory id=INC-001 metadata{verification_score:1.0}
```

With consulted priors (e.g., `INC-004` when seed `INC-001..003` are present): `ingest` fires recall → `analyze` receives `consulted=[ConsultedIncident(incident_id=INC-002, similarity_score≈0.71, note:"redis_cache_eviction: …")]` → `writer` lists `consulted_incidents` in output but never cites their ids in `evidence_refs` (enforced by both writer instruction and `is_consult_only_leak`).

Retry trajectory (exercised by `helpers.py:147` `bad_postmortem_first=True`):

```
writer  LLM[Postmortem] claims [E1,E99] → _bad_refs={E99} → re_prompt "…must use only valid ids from [E1..E8]. Revise…"
writer  LLM[Postmortem] claims [E1,E4],[E5] → valid
retries: [{"reason":"citation violations ['E99']","re_prompt":"…Revise…","result":"ok"}]
```

Human-checkpoint trajectory (CLI path):

```
api.py run INC-005 --fake → trajectories/INC-005/{ingest,timeline,analyze,writer,verifier}.json  status=pending_approval
api.py approve INC-005 --apply INC-002 --by "oncall" --fake → trajectories/INC-005/human_gate.json {human_decision:"approved", applied:["INC-002"]}
                                        → trajectories/INC-005/memory_writer.json {embedded:true}
api.py reject INC-005 "uncertain"      → trajectories/INC-005/human_gate.json {human_decision:"rejected"}
```

---

## How to inspect / reproduce

```powershell
# single incident (writes all 7)
python graph.py --incident INC-001 --fake
Get-Content trajectories/INC-001/timeline.json | ConvertFrom-Json | ForEach-Object output
Get-Content trajectories/INC-001/verifier.json | ConvertFrom-Json | ForEach-Object verifier_math

# full eval pack (8 incidents)
python eval.py --fake

# CLI gate pack
python api.py run INC-004 --fake --db app.db
python api.py show INC-004 --db app.db
python api.py approve INC-004 --fake --db app.db
Get-Content trajectories/INC-004/human_gate.json
```

*All trajectories are JSON with keys `node, system_prompt, user_prompt, input_state, tool_calls, retries, output, human_decision, verifier_math` — exactly the §B.4 checklist. They are the evidence pack for "would another person reach the same result?" — every score in `improvement_changelog.md` traces to a `verifier_math` / `evidence_refs` set-check in these files.*
