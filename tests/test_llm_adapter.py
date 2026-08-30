"""Phase 3 — llm_adapter tests.

Covers: hash determinism, replay with no network, offline chat miss raises, local
embedding fallback (shape + determinism), and OPENAI_BASE_URL override honored on a
compatible (non-OpenAI) endpoint. All network paths are monkeypatched so the suite
runs fully offline.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_adapter import CacheMissError, FALLBACK_EMBED_DIM, LLMAdapter  # noqa: E402
from schemas import Postmortem  # noqa: E402


def _chat_request_hash(adapter, system, messages, model, schema):
    full_system = system + adapter._schema_instruction(schema)
    return adapter._hash("chat", {
        "system": full_system, "messages": messages, "model": model,
        "params": {}, "schema": adapter._schema_fingerprint(schema),
    })


def _tmp_cache(tmp_path):
    return str(tmp_path / "llm_cache.jsonl")


def _seed_cache(path, request_hash, response):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"request_hash": request_hash, "response": response}) + "\n")


def _make_adapter(tmp_path, live=False, base_url=None, model="gpt-4o-mini"):
    return LLMAdapter(
        api_key="sk-test",
        base_url=base_url or "https://api.openai.com/v1",
        model=model,
        live=live,
        cache_path=_tmp_cache(tmp_path),
    )


def test_hash_is_deterministic_and_input_sensitive():
    a = _make_adapter("dummy") if False else LLMAdapter(cache_path=":mem:")
    h1 = a._hash("chat", {"system": "s", "messages": [{"role": "user", "content": "x"}], "model": "m", "params": {}, "schema": ""})
    h2 = a._hash("chat", {"system": "s", "messages": [{"role": "user", "content": "x"}], "model": "m", "params": {}, "schema": ""})
    assert h1 == h2
    h3 = a._hash("chat", {"system": "s", "messages": [{"role": "user", "content": "y"}], "model": "m", "params": {}, "schema": ""})
    assert h3 != h1


def test_replay_returns_stored_response_without_network(tmp_path, monkeypatch):
    a = _make_adapter(tmp_path, live=False)
    # Force any live call to hard-fail so we prove the replay path is taken.
    monkeypatch.setattr(a, "_live_chat", lambda *args, **kw: (_ for _ in ()).throw(RuntimeError("NETWORK USED")))

    stored = {
        "incident_id": "INC-001", "summary": "s", "impact": "i", "root_cause": "config_timeout_drop",
        "timeline": [], "action_items": ["a"],
        "claims": [{"statement": "x", "evidence_refs": ["E1"]}],
        "consulted_incidents": [],
    }
    h = _chat_request_hash(a, "sys", [{"role": "user", "content": "go"}], "gpt-4o-mini", Postmortem)
    _seed_cache(a.cache_path, h, stored)

    out = asyncio.run(a.chat("sys", [{"role": "user", "content": "go"}], schema=Postmortem))
    assert isinstance(out, Postmortem)
    assert out.incident_id == "INC-001"
    assert out.root_cause == "config_timeout_drop"


def test_chat_miss_in_replay_raises(tmp_path, monkeypatch):
    a = _make_adapter(tmp_path, live=False)
    monkeypatch.setattr(a, "_live_chat", lambda *args, **kw: (_ for _ in ()).throw(RuntimeError("NETWORK USED")))
    with pytest.raises(CacheMissError):
        asyncio.run(a.chat("sys", [{"role": "user", "content": "go"}], schema=Postmortem))


def test_embedding_fallback_shape_and_determinism(tmp_path):
    a = _make_adapter(tmp_path, live=False)
    texts = ["payment timeout dropped", "redis eviction storm"]
    v1 = asyncio.run(a.embed(texts))
    v2 = asyncio.run(a.embed(texts))
    assert len(v1) == 2
    assert all(len(vec) == FALLBACK_EMBED_DIM for vec in v1)
    assert v1 == v2  # deterministic
    # Different text -> different (not all-equal) vectors.
    assert any(abs(x - y) > 1e-9 for x, y in zip(v1[0], v1[1]))


def test_embedding_replay_returns_stored(tmp_path, monkeypatch):
    a = _make_adapter(tmp_path, live=False)
    monkeypatch.setattr(a, "_fallback_embed", lambda texts: (_ for _ in ()).throw(RuntimeError("FALLBACK USED")))
    stored = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    h = a._hash("embed", {"texts": ["a", "b"], "model": "text-embedding-3-small"})
    _seed_cache(a.cache_path, h, stored)
    out = asyncio.run(a.embed(["a", "b"]))
    assert out == stored


def test_base_url_override_honored_on_compatible_endpoint(tmp_path, monkeypatch):
    # Simulate an Ollama-style local OpenAI-compatible endpoint.
    base_url = "http://localhost:11434/v1"
    a = _make_adapter(tmp_path, live=True, base_url=base_url, model="llama3.1")

    captured = {}

    class FakeChoice:
        message = type("M", (), {"content": json.dumps({
            "incident_id": "INC-002", "summary": "s", "impact": "i", "root_cause": "redis_cache_eviction",
            "timeline": [], "action_items": ["a"],
            "claims": [{"statement": "x", "evidence_refs": ["E1"]}],
            "consulted_incidents": [],
        })})()

    class FakeResp:
        choices = [FakeChoice()]

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeResp()

    # Patch the live client so no real socket is opened.
    monkeypatch.setattr(a.client.chat.completions, "create", fake_create)

    out = asyncio.run(a.chat("sys", [{"role": "user", "content": "go"}], schema=Postmortem))
    assert isinstance(out, Postmortem)
    assert out.incident_id == "INC-002"
    # The request reached the endpoint with the configured base url + model.
    assert captured["model"] == "llama3.1"
    assert "response_format" in captured
    assert a.base_url == base_url


def test_live_call_appends_fixture(tmp_path, monkeypatch):
    a = _make_adapter(tmp_path, live=True, model="gpt-4o-mini")

    class FakeChoice:
        message = type("M", (), {"content": json.dumps({
            "incident_id": "INC-003", "summary": "s", "impact": "i", "root_cause": "kafka_consumer_lag",
            "timeline": [], "action_items": ["a"],
            "claims": [{"statement": "x", "evidence_refs": ["E1"]}],
            "consulted_incidents": [],
        })})()

    class FakeResp:
        choices = [FakeChoice()]

    async def fake_create(**kwargs):
        return FakeResp()

    monkeypatch.setattr(a.client.chat.completions, "create", fake_create)

    out = asyncio.run(a.chat("sys", [{"role": "user", "content": "go"}], schema=Postmortem))
    assert out.root_cause == "kafka_consumer_lag"
    # Fixture persisted for offline replay.
    assert os.path.exists(a.cache_path)
    with open(a.cache_path, "r", encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    assert len(lines) == 1 and lines[0]["response"]["incident_id"] == "INC-003"


# ------------------------------------------------------------------ _parse_chat List[X] robustness (live-path bug)
# The live API with response_format=json_object always returns a JSON object,
# even for List[X] schemas that want an array.  _parse_chat must handle both.
def test_parse_chat_list_array_ok():
    from typing import List

    from schemas import TimelineEvent

    data = json.dumps([{"ts": "2026-08-20T14:02:00", "description": "deploy", "evidence_refs": ["E1"]}])
    out = LLMAdapter._parse_chat(data, List[TimelineEvent])
    assert isinstance(out, list)
    assert isinstance(out[0], TimelineEvent)
    assert out[0].evidence_refs == ["E1"]


def test_parse_chat_list_object_wrapper_events_key():
    from typing import List

    from schemas import TimelineEvent

    data = json.dumps({"events": [{"ts": "2026-08-20T14:02:00", "description": "deploy", "evidence_refs": ["E1"]}]})
    out = LLMAdapter._parse_chat(data, List[TimelineEvent])
    assert isinstance(out, list) and isinstance(out[0], TimelineEvent)


def test_parse_chat_list_object_wrapper_items_key():
    from typing import List

    from schemas import TimelineEvent

    data = json.dumps({"items": [{"ts": "2026-08-20T14:02:00", "description": "deploy", "evidence_refs": ["E1"]}]})
    out = LLMAdapter._parse_chat(data, List[TimelineEvent])
    assert isinstance(out[0], TimelineEvent)


def test_parse_chat_list_object_wrapper_candidates_key():
    from typing import List

    from schemas import RootCauseCandidate

    data = json.dumps(
        {"candidates": [{"rank": 1, "confidence": 0.9, "hypothesis": "h", "root_cause_label": "config_timeout_drop", "supporting_evidence": ["E1"], "contradicting_evidence": ["E2"]}]}
    )
    out = LLMAdapter._parse_chat(data, List[RootCauseCandidate])
    assert out[0].root_cause_label == "config_timeout_drop"


def test_parse_chat_list_object_generic_single_list_fallback():
    from typing import List

    from schemas import TimelineEvent

    data = json.dumps({"my_wrapper": [{"ts": "2026-08-20T14:02:00", "description": "deploy", "evidence_refs": ["E1"]}]})
    out = LLMAdapter._parse_chat(data, List[TimelineEvent])
    assert isinstance(out, list) and len(out) == 1


def test_parse_chat_list_object_invalid_raises_value_error_not_attribute_error():
    from typing import List

    from schemas import TimelineEvent

    data = json.dumps({"foo": "bar"})
    with pytest.raises(ValueError, match="Failed to parse"):
        LLMAdapter._parse_chat(data, List[TimelineEvent])
    # Ensure the original bug (AttributeError: type object 'list' has no attribute 'model_validate') is gone
    try:
        LLMAdapter._parse_chat(data, List[TimelineEvent])
    except AttributeError as exc:
        pytest.fail(f"should not raise AttributeError, got {exc}")
    except ValueError:
        pass


def test_parse_chat_list_direct_python_list_not_string():
    from typing import List

    from schemas import TimelineEvent

    data = [{"ts": "2026-08-20T14:02:00", "description": "deploy", "evidence_refs": ["E1"]}]
    out = LLMAdapter._parse_chat(data, List[TimelineEvent])
    assert out[0].ts == "2026-08-20T14:02:00"


def test_parse_chat_list_direct_python_dict_wrapper():
    from typing import List

    from schemas import TimelineEvent

    data = {"timeline": [{"ts": "2026-08-20T14:02:00", "description": "deploy", "evidence_refs": ["E1"]}]}
    out = LLMAdapter._parse_chat(data, List[TimelineEvent])
    assert isinstance(out, list) and out[0].description == "deploy"
