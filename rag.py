"""RAG layer for the Agentic Incident Postmortem Synthesizer — Phase 4.

Single `incident_memory` collection. Stores *approved* postmortems as consultable
hypotheses so the agent can recall prior incidents during ingest.

Contract / Chroma rules (do not diverge):
- Metadata MUST be scalar-only (str | int | float | bool). Lists/dicts are rejected
  by Chroma. `symptom_keywords` is therefore a comma-joined string, never null.
  `add()` validates this boundary for BOTH backends (bug: None/list/dict slipped
  through to Chroma and raised).
- `recall(text, n)` returns `ConsultedIncident` objects whose `similarity_score`
  is `1 - distance` (i.e. bounded in [0, 1] for cosine distance). The in-memory
  backend now clamps its raw cosine to [0, 1] too, matching Chroma (bug: negative
  similarity leaked).
- IDs MUST be unique. `add()` enforces this for both backends instead of letting
  the in-memory store silently append while Chroma rejects (bug: count divergence).
- `n <= 0` returns no results in BOTH backends (memory used to return [], Chroma
  forced n_results=1; now Chroma queries max(1, n) and slices to n for parity).
- The collection metadata persists `embedding_model` + `purpose` so reloads can
  detect model/dimension incompatibility (bug: they were omitted).
- `where` filtering supports Chroma-style operators (exact, $in, $ne, $gt, $gte,
  $lt, $lte); see `_matches`.

Backend: uses Chroma when `chromadb` is importable, otherwise falls back to an
in-memory cosine store so the suite runs fully offline with no native dep.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

from schemas import ConsultedIncident

try:  # chromadb is optional; the in-memory backend works everywhere.
    import chromadb  # noqa: F401
    _HAS_CHROMA = True
except Exception:  # pragma: no cover - depends on install
    _HAS_CHROMA = False


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _clamp_sim(sim: float) -> float:
    """Clamp a similarity score into [0, 1] and round for stable output."""
    return round(min(1.0, max(0.0, sim)), 6)


def _validate_scalar_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validate metadata at the boundary so BOTH backends reject bad values.

    Chroma only accepts str | int | float | bool (bool is an int subclass, so it
    is checked first). Rejects None / list / dict / tuple before they reach Chroma.
    """
    clean: Dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, bool):
            clean[k] = v
        elif isinstance(v, (int, float)):
            clean[k] = v
        elif isinstance(v, str):
            clean[k] = v
        else:
            raise TypeError(
                f"metadata[{k!r}] must be a scalar (str|int|float|bool), "
                f"got {type(v).__name__}"
            )
    return clean


async def _default_embed(texts: List[str]) -> List[List[float]]:
    """Default embedder: route through LLMAdapter.embed().

    This honours the record/replay cache and the live 1536-dim path instead of
    calling the 256-dim `_hash_embed` offline fallback directly. `rag` always
    awaits the result via `_maybe_await`, so returning a coroutine is fine.
    """
    from llm_adapter import LLMAdapter

    return await LLMAdapter().embed(texts)


async def _maybe_await(fn: Callable, texts: List[str]) -> List[List[float]]:
    out = fn(texts)
    if hasattr(out, "__await__"):
        out = await out
    return out


class _InMemoryMemory:
    """Cosine-similarity store used when Chroma is unavailable."""

    def __init__(self, name: str, embed_fn: Callable) -> None:
        self.name = name
        self._embed = embed_fn
        self._docs: List[Dict[str, Any]] = []
        self._ids: set = set()

    async def add(self, id: str, document: str, metadata: Dict[str, Any]) -> None:
        if id in self._ids:
            raise ValueError(f"duplicate id {id!r} in memory store (ids must be unique)")
        metadata = _validate_scalar_metadata(metadata)
        vec = (await _maybe_await(self._embed, [document]))[0]
        self._docs.append(
            {"id": id, "document": document, "metadata": metadata, "vector": vec}
        )
        self._ids.add(id)

    async def recall(self, symptom_text: str, n: int = 5,
                     where: Optional[Dict[str, Any]] = None) -> List[ConsultedIncident]:
        if not self._docs or n <= 0:
            return []
        q = (await _maybe_await(self._embed, [symptom_text]))[0]
        scored = []
        for d in self._docs:
            if where and not _matches(d["metadata"], where):
                continue
            sim = _cosine(q, d["vector"])
            scored.append((sim, d))
        scored.sort(key=lambda x: -x[0])
        return [
            ConsultedIncident(
                incident_id=d["metadata"].get("incident_id", d["id"]),
                similarity_score=_clamp_sim(sim),
                applied=False,
                note=_note(d),
            )
            for sim, d in scored[:n]
        ]

    def count(self) -> int:
        return len(self._docs)


class _ChromaMemory:
    """Thin wrapper over a single Chroma collection (cosine space)."""

    def __init__(self, name: str, embed_fn: Callable,
                 persist_dir: Optional[str] = None,
                 embedding_model: str = "unknown",
                 purpose: str = "incident_postmortem_recall") -> None:
        import chromadb

        client = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir
            else chromadb.Client()
        )
        self._client = client
        self._embed = embed_fn
        self._coll = client.get_or_create_collection(
            name=name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_model,
                "purpose": purpose,
            },
        )

    async def add(self, id: str, document: str, metadata: Dict[str, Any]) -> None:
        metadata = _validate_scalar_metadata(metadata)
        existing = self._coll.get(ids=[id])
        if existing and existing.get("ids"):
            raise ValueError(f"duplicate id {id!r} in Chroma collection (ids must be unique)")
        vec = (await _maybe_await(self._embed, [document]))[0]
        self._coll.add(ids=[id], documents=[document], embeddings=[vec], metadatas=[metadata])

    async def recall(self, symptom_text: str, n: int = 5,
                     where: Optional[Dict[str, Any]] = None) -> List[ConsultedIncident]:
        if n <= 0:
            return []
        q = (await _maybe_await(self._embed, [symptom_text]))[0]
        res = self._coll.query(
            query_embeddings=[q], n_results=max(1, n),
            where=where, include=["metadatas", "documents", "distances"],
        )
        out: List[ConsultedIncident] = []
        for i, dist in enumerate(res["distances"][0]):
            meta = res["metadatas"][0][i] or {}
            out.append(
                ConsultedIncident(
                    incident_id=meta.get("incident_id", res["ids"][0][i]),
                    similarity_score=_clamp_sim(1.0 - dist),
                    applied=False,
                    note=_note({"document": res["documents"][0][i], "metadata": meta}),
                )
            )
        return out[:n]

    def count(self) -> int:
        return self._coll.count()


def _matches(metadata: Dict[str, Any], where: Dict[str, Any]) -> bool:
    """In-memory equivalent of Chroma's `where` filter.

    Supports the operators Chroma exposes: exact scalar equality plus the
    `$in`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte` documents. Nested `$and`/`$or`
    are not implemented here (documented limitation); add them if the eval loop
    needs composite filters.
    """
    for k, cond in where.items():
        actual = metadata.get(k)
        if isinstance(cond, dict) and cond and all(
            str(op).startswith("$") for op in cond
        ):
            for op, val in cond.items():
                if op == "$in":
                    if actual not in val:
                        return False
                elif op == "$ne":
                    if actual == val:
                        return False
                elif op == "$gt":
                    if not (actual is not None and actual > val):
                        return False
                elif op == "$gte":
                    if not (actual is not None and actual >= val):
                        return False
                elif op == "$lt":
                    if not (actual is not None and actual < val):
                        return False
                elif op == "$lte":
                    if not (actual is not None and actual <= val):
                        return False
                else:
                    # Unknown operator: fail closed (no match) rather than silent pass.
                    return False
        else:
            if actual != cond:
                return False
    return True


def _note(doc: Dict[str, Any]) -> str:
    rc = doc.get("metadata", {}).get("root_cause_label")
    first = (doc.get("document") or "").splitlines()
    first = first[0] if first else ""
    return f"{rc}: {first}" if rc else first


def create_memory_collection(name: str = "incident_memory",
                            embed_fn: Optional[Callable] = None,
                            backend: str = "auto",
                            persist_dir: Optional[str] = None,
                            embedding_model: str = "unknown",
                            purpose: str = "incident_postmortem_recall") -> Any:
    """Return an object with async `add(id, document, metadata)` and
    `recall(text, n, where)` plus a sync `count()`.

    `backend` is one of: "auto" (Chroma if available else in-memory),
    "chroma", "memory".

    `embedding_model`/`purpose` are persisted in the Chroma collection metadata
    so a reload can detect model/dimension incompatibility.
    """
    embed_fn = embed_fn or _default_embed
    if backend == "auto":
        backend = "chroma" if _HAS_CHROMA else "memory"
    if backend == "chroma":
        if not _HAS_CHROMA:  # pragma: no cover
            raise RuntimeError("chromadb not installed; use backend='memory'")
        return _ChromaMemory(name, embed_fn, persist_dir, embedding_model, purpose)
    return _InMemoryMemory(name, embed_fn)


async def aseed_from_memory_docs(collection: Any, docs: List[Dict[str, Any]]) -> None:
    """Async bulk-load of pre-seeded `incident_memory` documents.

    Safe to call from within a running event loop (e.g. the eval loop). Each
    `doc` has shape {"incident_id", "document", "metadata"}.
    """
    for d in docs:
        await collection.add(
            id=d["incident_id"], document=d["document"], metadata=d["metadata"]
        )


def seed_from_memory_docs(collection: Any, docs: List[Dict[str, Any]]) -> None:
    """Sync wrapper around `aseed_from_memory_docs`.

    Raises if called from inside a running event loop (where `asyncio.run`
    would raise RuntimeError) — in that case await `aseed_from_memory_docs`
    directly instead.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    else:
        loop = True
    if loop is not None:
        raise RuntimeError(
            "seed_from_memory_docs() cannot run inside a running event loop; "
            "await aseed_from_memory_docs() instead"
        )

    async def _run():
        for d in docs:
            await collection.add(
                id=d["incident_id"], document=d["document"], metadata=d["metadata"]
            )

    asyncio.run(_run())
