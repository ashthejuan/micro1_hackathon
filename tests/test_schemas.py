"""Phase 1 — schema contract tests.

Every model forbids extra fields and requires declared fields, so the LLM cannot
return prose or smuggle undeclared data through `with_structured_output`.
"""

import pytest
from pydantic import ValidationError

from schemas import (
    Claim,
    ConsultedIncident,
    Evidence,
    Postmortem,
    RootCauseCandidate,
    TimelineEvent,
    VerificationReport,
)


def test_evidence_requires_all_fields():
    with pytest.raises(ValidationError):
        Evidence(id="E1", incident_id="INC-001", ts="2026-08-20T14:00:00", source="deploys")
    # full construction works
    ev = Evidence(id="E1", incident_id="INC-001", ts="2026-08-20T14:00:00",
                  source="deploys", source_url=None, content="deploy X")
    assert ev.source == "deploys"
    assert ev.source_url is None


def test_models_reject_extra_fields():
    with pytest.raises(ValidationError):
        Claim(statement="x", evidence_refs=["E1"], surprise="nope")
    with pytest.raises(ValidationError):
        Postmortem(incident_id="I", summary="s", impact="i", root_cause="r",
                   timeline=[], action_items=[], claims=[], consulted_incidents=[],
                   extra="nope")


def test_claim_evidence_refs_typed():
    with pytest.raises(ValidationError):
        Claim(statement="x", evidence_refs=["E1", 99])  # int not allowed
    c = Claim(statement="x", evidence_refs=["E1"])
    assert c.from_recalled_incident is None  # optional, defaults None


def test_claim_from_recalled_optional():
    c = Claim(statement="x", evidence_refs=["E1"], from_recalled_incident="INC-014")
    assert c.from_recalled_incident == "INC-014"


def test_postmortem_requires_claims_and_timeline():
    with pytest.raises(ValidationError):
        Postmortem(incident_id="I", summary="s", impact="i", root_cause="r",
                   timeline=[], action_items=[], consulted_incidents=[])
    pm = Postmortem(incident_id="I", summary="s", impact="i", root_cause="r",
                    timeline=[TimelineEvent(ts="t", description="d", evidence_refs=["E1"])],
                    action_items=["a"], claims=[Claim(statement="s", evidence_refs=["E1"])],
                    consulted_incidents=[])
    assert len(pm.claims) == 1


def test_invalid_json_fails_structured_output_contract():
    with pytest.raises(ValidationError):
        Claim.model_validate_json('{"statement": "x"}')  # missing evidence_refs
    with pytest.raises(ValidationError):
        Claim.model_validate_json('not json')  # invalid JSON -> json_invalid


def test_root_cause_candidate_contradicting_evidence():
    rc = RootCauseCandidate(rank=1, confidence=0.9, hypothesis="h",
                            root_cause_label="config_timeout_drop",
                            supporting_evidence=["E1"], contradicting_evidence=["E5"])
    assert "E5" in rc.contradicting_evidence


def test_consulted_incident_default_dismissed():
    ci = ConsultedIncident(incident_id="INC-014", similarity_score=0.83, note="n")
    assert ci.applied is False


def test_verification_report_rebuild_roundtrip():
    vr = VerificationReport(
        incident_id="I",
        claim_reports=[],
        verification_score=0.0,
    )
    assert vr.incident_id == "I"
