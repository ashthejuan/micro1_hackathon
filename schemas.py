"""Pydantic schemas for the Agentic Incident Postmortem Synthesizer.

All agent I/O are Pydantic models so the LLM is forced through
`with_structured_output(Model)` — it returns the schema or fails. `extra="forbid"`
guarantees the model cannot smuggle in prose or undeclared fields.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(_Base):
    id: str
    incident_id: str
    ts: str  # ISO-8601 text (lexicographically sortable)
    source: str  # deploys | metrics | logs | chat
    source_url: Optional[str] = None
    content: str


class TimelineEvent(_Base):
    ts: str
    description: str
    evidence_refs: List[str]


class RootCauseCandidate(_Base):
    rank: int
    confidence: float
    hypothesis: str
    root_cause_label: str  # slug vs fixture true_root_cause / red_herring (scoring key)
    supporting_evidence: List[str]
    contradicting_evidence: List[str]  # explicit red-herring rejection
    from_prior_incident: Optional[str] = None  # id if motivated by memory (NOT as fact)


class Claim(_Base):
    statement: str
    evidence_refs: List[str]  # MUST be non-empty & valid Evidence ids
    from_recalled_incident: Optional[str] = None  # must be None to count as verified


class ConsultedIncident(_Base):
    incident_id: str
    similarity_score: float
    applied: bool = False  # default dismissed; human can flip in gate
    note: str


class Postmortem(_Base):
    incident_id: str
    summary: str
    impact: str
    root_cause: str
    timeline: List[TimelineEvent]
    action_items: List[str]
    claims: List[Claim]  # writer commits its assertions explicitly
    consulted_incidents: List[ConsultedIncident]


class VerificationReport(_Base):
    incident_id: str
    claim_reports: List["ClaimVerification"]
    verification_score: float  # backed / total  (DETERMINISTIC set-check)


class ClaimVerification(_Base):
    claim_index: int
    statement: str
    evidence_refs: List[str]
    from_recalled_incident: Optional[str]
    backed: bool
    missing_evidence: List[str]


VerificationReport.model_rebuild()
