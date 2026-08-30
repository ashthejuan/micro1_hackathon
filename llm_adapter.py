"""OpenAI-compatible LLM adapter with record/replay and local embedding fallback.

Phase 3 of the Agentic Incident Postmortem Synthesizer.

Design goals
------------
- **Endpoint-agnostic:** talks to any OpenAI-compatible server (OpenAI, Ollama,
  vLLM, LiteLLM, OpenRouter, ...) via `OPENAI_BASE_URL` (default
  ``https://api.openai.com/v1``), `OPENAI_API_KEY`, and `MODEL`. Only the generic
  chat-completions + embeddings surface is used (json-object mode + pydantic parse)
  so it works everywhere, not just on endpoints that expose `beta` structured output.
- **Reproducible offline:** every call is keyed by a sha256 request hash and looked
  up in ``fixtures/llm_cache.jsonl``. On a hit, the stored response is replayed with
  *no network*. On a miss while `LIVE` is unset, chat raises (forcing `--live`) and
  embed falls back to a deterministic local embedding so the eval loop never blocks.
- **Embeddings:** live path calls the `/embeddings` endpoint; replay path returns the
  stored vectors; miss-in-replay path uses a deterministic hash bag-of-words vector
  (optionally sentence-transformers when `USE_ST_EMBED=1` and loadable).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:  # openai is required only for the live path.
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - allows import without the dep present
    AsyncOpenAI = None  # type: ignore


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_PATH = os.path.join(ROOT, "fixtures", "llm_cache.jsonl")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
FALLBACK_EMBED_DIM = 256


class CacheMissError(RuntimeError):
    """Raised when a chat request is not in the replay cache and LIVE is off."""


class LLMAdapter:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        live: Optional[bool] = None,
        cache_path: Optional[str] = None,
        embeddings_model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self.model = model if model is not None else os.environ.get("MODEL", "gpt-4o-mini")
        # LIVE defaults to the OPENAI_API_KEY being present + LIVE env flag.
        if live is None:
            live = os.environ.get("LIVE", "") not in ("", "0", "false", "False")
        self.live = bool(live)
        self.cache_path = cache_path if cache_path is not None else DEFAULT_CACHE_PATH
        self.embeddings_model = embeddings_model or os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")
        self._cache: Optional[Dict[str, Any]] = None
        self.client = self._build_client() if (self.live and AsyncOpenAI is not None) else None

    # ------------------------------------------------------------------ config
    def _build_client(self) -> "AsyncOpenAI":
        if AsyncOpenAI is None:  # pragma: no cover
            raise RuntimeError("openai package is required for LIVE mode")
        return AsyncOpenAI(
            api_key=self.api_key or "sk-noauth",
            base_url=self.base_url,
        )

    # ------------------------------------------------------------- hashing/core
    @staticmethod
    def _stable(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)

    def _hash(self, kind: str, payload: Dict[str, Any]) -> str:
        blob = self._stable({"kind": kind, **payload})
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ----------------------------------------------------------------- cache IO
    def _load_cache(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache
        cache: Dict[str, Any] = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    cache[rec["request_hash"]] = rec["response"]
        self._cache = cache
        return cache

    def _append(self, request_hash: str, response: Any) -> None:
        # Keep the in-memory copy warm and persist atomically-ish (append).
        if self._cache is None:
            self._load_cache()
        assert self._cache is not None
        self._cache[request_hash] = response
        parent = os.path.dirname(self.cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"request_hash": request_hash, "response": response}, ensure_ascii=False) + "\n")

    @staticmethod
    def _schema_fingerprint(schema: Optional[Any]) -> str:
        if schema is None:
            return ""
        try:
            if hasattr(schema, "model_json_schema"):
                return LLMAdapter._stable(schema.model_json_schema())
        except Exception:
            pass
        try:
            from pydantic import TypeAdapter

            return LLMAdapter._stable(TypeAdapter(schema).json_schema())
        except Exception:  # pragma: no cover
            return getattr(schema, "__name__", str(schema))

    @staticmethod
    def _schema_instruction(schema: Optional[Any]) -> str:
        if schema is None:
            return ""
        try:
            if hasattr(schema, "model_json_schema"):
                js = schema.model_json_schema()
            else:
                raise AttributeError("no model_json_schema")
        except Exception:
            try:
                from pydantic import TypeAdapter

                js = TypeAdapter(schema).json_schema()
            except Exception:
                return ""
        return (
            "\n\nRespond with a single JSON object and no prose, conforming exactly to "
            "this JSON schema:\n" + json.dumps(js, ensure_ascii=False)
        )

    # -------------------------------------------------------------------- chat
    async def chat(
        self,
        system: str,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        schema: Optional[Any] = None,
    ) -> Any:
        """Call the chat endpoint (or replay) and return a parsed structured object.

        Returns a ``schema`` instance when ``schema`` is given, otherwise the raw
        assistant message content string.
        """
        model = model or self.model
        params = dict(params or {})
        full_system = system + self._schema_instruction(schema)
        payload = {
            "system": full_system,
            "messages": messages,
            "model": model,
            "params": params,
            "schema": self._schema_fingerprint(schema),
        }
        request_hash = self._hash("chat", payload)
        cache = self._load_cache()
        if request_hash in cache:
            return self._parse_chat(cache[request_hash], schema)

        if not self.live:
            raise CacheMissError(
                f"No replay fixture for request_hash={request_hash[:12]}... "
                "and LIVE is off. Re-run with LIVE=1 (eval.py --live) to record it."
            )

        content = await self._live_chat(full_system, messages, model, params)
        parsed = self._parse_chat(content, schema, raw=True)
        # Persist the parsed object so replay never re-parses.
        # For List[Model] schemas the parsed value is a list; use TypeAdapter dump.
        if schema is None:
            to_store = parsed
        else:
            try:
                # BaseModel instance
                to_store = parsed.model_dump()  # type: ignore
            except AttributeError:
                # List[BaseModel] or other generic — serialize each element
                try:
                    from pydantic import TypeAdapter

                    to_store = TypeAdapter(schema).dump_python(parsed)
                except Exception:
                    to_store = parsed
        self._append(request_hash, to_store)
        return parsed

    async def _live_chat(
        self,
        system: str,
        messages: List[Dict[str, str]],
        model: str,
        params: Dict[str, Any],
    ) -> str:
        if self.client is None:  # pragma: no cover
            self.client = self._build_client()
        # OpenAI's json_object mode requires the word "json" in the prompt.
        # Ensure it is present so the live path does not 400 when system prompts
        # are plain instructions (agents currently say "You are the timeline agent...").
        if "json" not in system.lower():
            system = system + "\n\nRespond with JSON."
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}] + list(messages),
        }
        kwargs.update(params)
        if "response_format" not in kwargs:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    @staticmethod
    def _parse_chat(data: Any, schema: Optional[Any], raw: bool = False) -> Any:
        if schema is None:
            return data
        if isinstance(data, str):
            data = json.loads(data)
        # Direct BaseModel
        if hasattr(schema, "model_validate"):
            return schema.model_validate(data)
        # Generic like List[Model] — use TypeAdapter
        try:
            from pydantic import TypeAdapter

            adapter = TypeAdapter(schema)
            return adapter.validate_python(data)
        except Exception as orig_exc:
            # When response_format=json_object is used (llm_adapter line 190),
            # the live API always returns a JSON object, even for List[X] schemas
            # that logically want an array.  Try to unwrap a dict-wrapped list
            # before giving up — this makes object-vs-array robust.
            if isinstance(data, dict):
                # Known wrapper keys (including generic envelope keys)
                for key in (
                    "events",
                    "timeline",
                    "timeline_events",
                    "candidates",
                    "root_cause_candidates",
                    "items",
                    "data",
                    "results",
                    "values",
                ):
                    if key in data and isinstance(data[key], list):
                        try:
                            from pydantic import TypeAdapter as _TA

                            return _TA(schema).validate_python(data[key])
                        except Exception:
                            continue
                # Fallback: dict with a single list value — use that list
                list_vals = [v for v in data.values() if isinstance(v, list)]
                if len(list_vals) == 1:
                    try:
                        from pydantic import TypeAdapter as _TA

                        return _TA(schema).validate_python(list_vals[0])
                    except Exception:
                        pass
                # Last resort: single-key dict whose sole value is a list
                if len(data) == 1:
                    sole = next(iter(data.values()))
                    if isinstance(sole, list):
                        try:
                            from pydantic import TypeAdapter as _TA

                            return _TA(schema).validate_python(sole)
                        except Exception:
                            pass
            # Never call schema.model_validate on a generic alias like List[X]:
            # list has no attribute model_validate and would raise a confusing
            # AttributeError.  Raise a clear ValueError that preserves the
            # original validation cause.
            raise ValueError(
                f"Failed to parse chat response for schema {schema!r}: {orig_exc}"
            ) from orig_exc

    # ------------------------------------------------------------------ embed
    async def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Return embeddings for ``texts`` (replay, live, or local fallback)."""
        model = model or self.embeddings_model
        payload = {"texts": list(texts), "model": model}
        request_hash = self._hash("embed", payload)
        cache = self._load_cache()
        if request_hash in cache:
            return [list(v) for v in cache[request_hash]]

        if self.live:
            if self.client is None:  # pragma: no cover
                self.client = self._build_client()
            resp = await self.client.embeddings.create(model=model, input=list(texts))
            vectors = [list(d.embedding) for d in resp.data]
            self._append(request_hash, vectors)
            return vectors

        # Offline miss: deterministic local embedding, never touches the network.
        return self._fallback_embed(texts)

    def _fallback_embed(self, texts: List[str]) -> List[List[float]]:
        if os.environ.get("USE_ST_EMBED", "0") not in ("", "0", "false", "False"):
            vec = self._sentence_transformers_embed(texts)
            if vec is not None:
                return vec
        return self._hash_embed(texts)

    @staticmethod
    def _hash_embed(texts: List[str], dim: int = FALLBACK_EMBED_DIM) -> List[List[float]]:
        import re

        out: List[List[float]] = []
        for t in texts:
            vec = [0.0] * dim
            toks = re.findall(r"[a-z0-9_]+", (t or "").lower())
            for tok in toks:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out

    @staticmethod
    def _sentence_transformers_embed(texts: List[str]) -> Optional[List[List[float]]]:
        try:  # pragma: no cover - optional, network/model dependent
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            vecs = model.encode(texts, normalize_embeddings=True)
            return [list(map(float, v)) for v in vecs]
        except Exception:
            return None
