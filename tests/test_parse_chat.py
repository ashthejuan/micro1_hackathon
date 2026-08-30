"""Small regression lock for LLMAdapter._parse_chat list unwrap.

Live API with response_format=json_object returns an object even when the
schema is List[Model].  _parse_chat must accept both a JSON array and a
JSON object wrapping the array, and must never raise
AttributeError: type object 'list' has no attribute 'model_validate'.

See llm_adapter.py:230-294.
"""
import json
import os
import sys
from typing import List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_adapter import LLMAdapter
from schemas import TimelineEvent


def test_parse_chat_list_array():
    """Proper array input parses OK (case B)."""
    raw = json.dumps([{"ts": "2026-08-20T14:02:00", "description": "d", "evidence_refs": ["E1"]}])
    out = LLMAdapter._parse_chat(raw, List[TimelineEvent])
    assert isinstance(out, list)
    assert out[0].ts == "2026-08-20T14:02:00"


def test_parse_chat_list_object_unwrap():
    """JSON object wrapping the list parses OK (case A) — no AttributeError."""
    raw = json.dumps({"events": [{"ts": "2026-08-20T14:02:00", "description": "d", "evidence_refs": ["E1"]}]})
    out = LLMAdapter._parse_chat(raw, List[TimelineEvent])
    assert isinstance(out, list)
    assert isinstance(out[0], TimelineEvent)

    # generic fallback: single list value under unknown key
    raw2 = json.dumps({"my_wrapper": [{"ts": "2026-08-20T14:02:00", "description": "d", "evidence_refs": ["E1"]}]})
    out2 = LLMAdapter._parse_chat(raw2, List[TimelineEvent])
    assert len(out2) == 1


def test_parse_chat_list_object_invalid_raises_value_error():
    """Invalid object must raise ValueError, not AttributeError (the bug)."""
    raw = json.dumps({"foo": "bar"})
    with pytest.raises(ValueError, match="Failed to parse"):
        LLMAdapter._parse_chat(raw, List[TimelineEvent])
