"""SQLite canonical store + deterministic claim verifier set-check.

Evidence is canonical here; Chroma holds only `incident_memory` embeddings. The
verifier is a PURE set-check (no LLM): a claim is `backed` iff every id in
`evidence_refs` exists in `evidence` for that incident AND `from_recalled_incident`
is None (recalled incidents may never be cited as evidence).
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, List, Optional, Tuple

from schemas import Claim, ClaimVerification, Postmortem, VerificationReport

SCHEMA = """
CREATE TABLE IF NOT EXISTS incident (
    id           TEXT PRIMARY KEY,
    window_start TEXT NOT NULL,
    window_end   TEXT NOT NULL,
    description  TEXT,
    status       TEXT NOT NULL DEFAULT 'running',
    time_created TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    incident_id  TEXT NOT NULL REFERENCES incident(id) ON DELETE CASCADE,
    id           TEXT NOT NULL,
    ts           TEXT NOT NULL,
    source       TEXT NOT NULL,
    source_url   TEXT,
    content      TEXT NOT NULL,
    PRIMARY KEY (incident_id, id)
);
CREATE INDEX IF NOT EXISTS idx_ev_ts     ON evidence(incident_id, ts);
CREATE INDEX IF NOT EXISTS idx_ev_source ON evidence(incident_id, source);

CREATE TABLE IF NOT EXISTS postmortem (
    incident_id      TEXT PRIMARY KEY REFERENCES incident(id) ON DELETE CASCADE,
    draft_json       TEXT NOT NULL,
    verification_json TEXT,
    consulted_json   TEXT,
    status           TEXT NOT NULL DEFAULT 'pending_approval',
    approved_by      TEXT,
    time_approved    TEXT,
    time_created     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verification (
    incident_id      TEXT NOT NULL REFERENCES incident(id) ON DELETE CASCADE,
    claim_index      INTEGER NOT NULL,
    statement        TEXT NOT NULL,
    evidence_refs    TEXT NOT NULL,
    from_recalled    TEXT,
    backed           INTEGER NOT NULL,
    missing_evidence TEXT,
    PRIMARY KEY (incident_id, claim_index)
);

CREATE TABLE IF NOT EXISTS evaluation_run (
    run_id       TEXT PRIMARY KEY,
    mode         TEXT NOT NULL,
    model        TEXT,
    incident_id  TEXT,
    verification_score REAL,
    red_herring_correct INTEGER,
    time_created TEXT
);
"""


def connect(db_path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def insert_incident(conn: sqlite3.Connection, incident: dict, status: str = "running",
                    time_created: str = "1970-01-01T00:00:00") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO incident "
        "(id, window_start, window_end, description, status, time_created) "
        "VALUES (?,?,?,?,?,?)",
        (incident["id"], incident["window_start"], incident["window_end"],
         incident.get("description"), status, time_created),
    )
    conn.commit()


def insert_evidence(conn: sqlite3.Connection, incident_id: str,
                    evidence: Iterable[dict]) -> None:
    rows = [
        (incident_id, e["id"], e["ts"], e["source"], e.get("source_url"), e["content"])
        for e in evidence
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO evidence "
        "(incident_id, id, ts, source, source_url, content) VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def query_evidence(conn: sqlite3.Connection, incident_id: str,
                   start: Optional[str] = None, end: Optional[str] = None) -> List[dict]:
    sql = "SELECT * FROM evidence WHERE incident_id=? "
    params: list = [incident_id]
    if start is not None:
        sql += "AND ts >= ? "
        params.append(start)
    if end is not None:
        sql += "AND ts <= ? "
        params.append(end)
    sql += "ORDER BY ts"
    cur = conn.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]


def valid_evidence_ids(conn: sqlite3.Connection, incident_id: str) -> set:
    cur = conn.execute("SELECT id FROM evidence WHERE incident_id=?", (incident_id,))
    return {r["id"] for r in cur.fetchall()}


def set_check(conn: sqlite3.Connection, incident_id: str, claim: Claim) -> Tuple[bool, List[str]]:
    """Deterministic claim verification.

    backed = (every ref is a real supplied id) AND (no recalled incident cited)
             AND (at least one ref present).
    Returns (backed, missing_refs).
    """
    valid = valid_evidence_ids(conn, incident_id)
    refs = list(claim.evidence_refs)
    missing = [r for r in refs if r not in valid]
    backed = (
        len(missing) == 0
        and len(refs) > 0
        and claim.from_recalled_incident is None
    )
    return backed, missing


def verify_postmortem(conn: sqlite3.Connection, incident_id: str,
                      postmortem: Postmortem) -> VerificationReport:
    reports: List[ClaimVerification] = []
    for i, claim in enumerate(postmortem.claims):
        backed, missing = set_check(conn, incident_id, claim)
        reports.append(ClaimVerification(
            claim_index=i,
            statement=claim.statement,
            evidence_refs=list(claim.evidence_refs),
            from_recalled_incident=claim.from_recalled_incident,
            backed=backed,
            missing_evidence=missing,
        ))
    score = sum(r.backed for r in reports) / len(reports) if reports else 0.0
    return VerificationReport(
        incident_id=incident_id,
        claim_reports=reports,
        verification_score=score,
    )


def insert_verification_rows(conn: sqlite3.Connection, incident_id: str,
                             report: VerificationReport) -> None:
    rows = [
        (incident_id, r.claim_index, r.statement, json_dumps(r.evidence_refs),
         r.from_recalled_incident, int(r.backed), json_dumps(r.missing_evidence))
        for r in report.claim_reports
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO verification "
        "(incident_id, claim_index, statement, evidence_refs, from_recalled, backed, missing_evidence) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def upsert_postmortem(conn: sqlite3.Connection, incident_id: str, draft_json: str,
                      verification_json: Optional[str] = None,
                      consulted_json: Optional[str] = None,
                      status: str = "pending_approval",
                      time_created: str = "1970-01-01T00:00:00") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO postmortem "
        "(incident_id, draft_json, verification_json, consulted_json, status, approved_by, time_approved, time_created) "
        "VALUES (?,?,?,?,?,NULL,NULL,?)",
        (incident_id, draft_json, verification_json, consulted_json, status, time_created),
    )
    conn.commit()


def insert_evaluation_run(conn: sqlite3.Connection, run_id: str, mode: str,
                          model: Optional[str], incident_id: Optional[str],
                          verification_score: float, red_herring_correct: int,
                          time_created: str = "1970-01-01T00:00:00") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO evaluation_run "
        "(run_id, mode, model, incident_id, verification_score, red_herring_correct, time_created) "
        "VALUES (?,?,?,?,?,?,?)",
        (run_id, mode, model, incident_id, verification_score, red_herring_correct, time_created),
    )
    conn.commit()


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
