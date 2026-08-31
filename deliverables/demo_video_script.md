# Demo Video Script — Agentic Incident Postmortem Synthesizer
**Hackathon:** micro1 Agentic Workflows (≤5 min, Deliverable 03)
**Run style:** LIVE terminal, offline (`--fake`, no API key needed), deterministic
**Tool runner used on camera (avoids the `uv run --with …` long prefix):**
```
python -c "import pydantic,langgraph,openai,chromadb" 2>/dev/null || uv run --with pydantic --with pytest --with pytest-asyncio --with openai --with langgraph --with "chromadb>=0.5" --with sentence-transformers python -c "import pydantic,langgraph,openai"
PY="uv run --with pydantic --with pytest --with pytest-asyncio --with openai --with langgraph --with chromadb --with sentence-transformers python"
```
(The grader env may not have uv; fallback: `python eval.py --fake` etc. if deps are installed.)

---

## 0. Title card (0:00–0:08)
**Frame (talk):**
> Postmortem Buddy: an agentic synthesizer that turns on-call incident evidence into *citable, verified* postmortems — with a deterministic set-check verifier and a human-in-the-loop gate.

---

## 1. The problem + who has it (0:08–0:40)
**Show: terminal prompt. Talk over it.**

> On-call engineers write postmortems from scattered observability evidence — deploys, metrics, logs, chat — under time pressure. The bottleneck isn't typing the doc; it's **causal reasoning + evidence-backed writing**. A single LLM prompt hallucinates ids, blames the wrong cause, and isn't auditable. Ground rules §04/§05 require a human checkpoint before anything publishes, and §09 says "connect every claim to evidence." Today that's manual.

**Live:**
```bash
$ python -c "import json,glob,os; [print(i['id'],i.get('true_root_cause'),'|| herring:',i.get('red_herring'),'|| evidence:',[e['id'] for e in i.get('evidence',[])]) for i in (json.load(open(f)) for f in sorted(glob.glob('incidents/INC-00*.json')))]"
# (11 synthetic incidents, each true_root_cause + red_herring + 5-8 evidence rows)
```
> Synthetic data (no private data, ground rule §07). Here's INC-004 — `dns_resolution_failure` is the truth, `load_balancer_flap` is the planted herring.

---

## 2. The baseline vs. the agent — the comparison table (0:40–1:10)
**Live (the judging-critical command):**
```bash
$ $PY eval.py --fake
=== Agentic Postmortem — eval comparison (offline replay) ===
Model: fake
Seeds: ['INC-001', 'INC-002', 'INC-003']  Eval: ['INC-004', 'INC-005', '…INC-011']
mode        verification  red_herring  completeness
----------------------------------------------------
agent              1.000        1.000         1.000
baseline           0.000        0.000         1.000
Note: eval simulates human checkpoint (auto-approve); real gate is CLI `api.py approve`.
```
> **Baseline A** = a single typed prompt that blames the red herring and cites a hallucinated `E99`. So `verification 0.0` — every claim is invented. **Agent** = the 7-node graph. `verification 1.0`, `red_herring 1.0` on all 8 incidents. Completeness is identical — that's the honest structural delta: citations are real, cause is right.

---

## 3. One realistic run, start to finish (1:10–3:00)
**Live — pick the challenging case INC-004 (dns vs load_balancer):**
```bash
$ $PY api.py run INC-004 --fake --db app.db
# → incident=INC-004 status=pending_approval verification_score=1.00 claims=2 trajectory=trajectories/INC-004
```
> That ran `api.py run` — NOT auto-approve (ground rule §04/§05). Draft is `pending_approval`. Inspect it:
```bash
$ $PY api.py show INC-004 --db app.db
# { "status":"pending_approval", "verification":{ "verification_score":1.0, "claim_reports":[ { "evidence_refs":["E1","E3"], "backed":true, "missing_evidence":[] }, … ] } }
```

**Now the §B.4 evidence pack — open a trajectory, not me telling you:**
```bash
$ ls trajectories/INC-004/
# ingest.json  timeline.json  analyze.json  writer.json  verifier.json  human_gate.json  memory_writer.json  manifest.json
$ python -c "import json; print(json.dumps(json.load(open('trajectories/INC-004/analyze.json'))['output'], indent=1))"
```
Show (talk over):
```json
[
  { "rank": 1, "confidence": 0.9,
    "root_cause_label": "dns_resolution_failure",          ← correct (not the herring)
    "supporting_evidence": ["E1","E3"], "contradicting_evidence": ["E4"] }   ← E4 = LB health, rejected
]
```
> The `analyze` agent explicitly rejected the red herring via `contradicting_evidence`. That `E4`/`E1`/`E3` set-check is what scores red_herring=1. Now the verifier:
```bash
$ python -c "import json; v=json.load(open('trajectories/INC-004/verifier.json')); print('score=',v['verifier_math'][0]['score'],'| per-claim:',[(c['backed'],c['evidence_refs']) for c in v['verifier_math']])"
# score= 1.0 | per-claim: [(True, ['E1','E3']), (True, ['E2'])]
```
> Every claim's `evidence_refs ⊆ valid_ids` — the deterministic set-check (store.py:146). Now the real human gate:
```bash
$ $PY api.py approve INC-004 --by "oncall@example.com" --fake --db app.db
# → incident=INC-004 status=approved approved_by=human embedded=True
```
> Human flipped `pending_approval → approved` and embedded it into `incident_memory` (consult-only hard-ban enforced — leaks get `embedded:false`). **That's the end-to-end: evidence → verified timeline → ranked root cause → cited postmortem → human gate → memory.**

---

## 4. The changelog — where the improvement came from (3:00–4:00)
**Show `deliverables/improvement_changelog.md` (open in editor / read_file), reading the table aloud:**
```
Stage            | verification | learning
Baseline (stub)  | 0.000        | Invented E99; completeness alone is theatre
Iter 1 Verifier    | ~0.2         | Set-check makes failure visible & deterministic
Iter 2 Citation guard + retry | 1.000 | _bad_refs + single re-prompt is the make-or-break lift
Iter 3 Async fan-out | rh 0→1.000 | Timeline‖Analyze → explicit contradicting_evidence rejects herrings
Iter 4 Consult-only memory | (no regression) | Prior incidents as hypotheses, never as evidence
Iter 6 Replay + hash-embed | reproducible | Same scores with no key (Reproducibility 15)
Final:           | v 1.00 vs 0.00 | **+1.00 verification, +1.00 red-herring**
```
> **Most-contributing change:** the citation-integrity guard + single retry (Iter 2). Without it the typed schemas still hallucinate `E99`; `verification 0.2 → 1.0`. That's the delta.

---

## 5. One experiment removed (4:00–4:25)
> We tried a `evidence_chunks` Chroma collection (semantic retrieval of evidence rows for verification). Removed: verification needs `E# ∈ valid_ids`, not a semantic match — so it only added cost + metadata bugs. Also tried 2-shot writer exemplars — no lift once Iter 2 landed, and overfit to timeout phrasing. **Lesson: constrain the LLM with schemas + a deterministic oracle at the boundary, don't ask the LLM to judge itself.**

---

## 6. Close + ground-rules compliance (4:25–5:00)
**Live (prove reproducibility / clean-env):**
```bash
$ $PY pytest -q 2>&1 | tail -1
# 113 passed, 3 skipped in ~8-15s
```
> 113 green, 0 secrets — API key never appears, `.env` is never read, the whole thing runs with no network. §04 sandbox (CLI, no live-system writes), §05 human gate (`api.py approve`), §08 credentials outside repo, §09 every score → `verifier.json`/`verifier_math` + `evaluation_run` rows. The repo is also a working **human postmortem reviewer**: the trajectories (§B.4) are the evidence pack judges can read without running code.

**Final frame (talk):**
> Postmortem Buddy: auditable, reproducible, and gated. Code + changelog + reproduction guide in the repo.

---

## Production cues
- **Camera:** terminal in focus; cursor visible; type commands, don't paste the giant `uv run` prefix — alias `PY` first (shown above).
- **Pacing:** each live block ≤25s; if `eval.py` (3-6s) or `pytest` (8-15s) runs long, cut to a "already-running" screen and voice over.
- **Fallback if deps installed locally (no uv):** drop the `uv run --with … python` wrappers → `python eval.py --fake`, `python api.py …`, `python -m pytest -q`.
- **Artifact check before recording:** `rm -f app.db; rm -rf trajectories/INC-004` so the demo starts clean and writes are fresh on camera.
