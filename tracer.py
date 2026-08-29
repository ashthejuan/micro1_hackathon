"""Trajectory tracer for the Agentic Incident Postmortem Synthesizer — Phase 5.

Writes per-agent trajectory files required by hackathon §B.4 (Final Deliverables
item 04, a submission blocker). Each agent run produces
`trajectories/{incident_id}/{agent}.json` capturing instructions → result, tool
responses, retries, and the human checkpoint; a `manifest.json` indexes them.

Schema per agent event (§B.4 compliant):
  node, system_prompt, user_prompt, input_state, tool_calls, retries, output,
  human_decision (+ verifier_math for the verifier node).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAJ_DIR = os.path.join(ROOT, "trajectories")


def _serialize(obj: Any) -> Any:
    """Best-effort JSON-safe conversion (handles Pydantic models / lists / dicts)."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


class Tracer:
    def __init__(self, incident_id: str, out_dir: str = DEFAULT_TRAJ_DIR) -> None:
        self.incident_id = incident_id
        self.out_dir = out_dir
        self.traj_dir = os.path.join(out_dir, incident_id)
        self._agents: List[str] = []

    def _write_json(self, path: str, obj: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)

    def record(
        self,
        node: str,
        *,
        system_prompt: str,
        user_prompt: str,
        input_state: Dict[str, Any],
        output: Any,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        retries: Optional[List[Dict[str, Any]]] = None,
        human_decision: Optional[Any] = None,
        verifier_math: Optional[Any] = None,
    ) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "node": node,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "input_state": _serialize(input_state),
            "tool_calls": _serialize(tool_calls or []),
            "retries": _serialize(retries or []),
            "output": _serialize(output),
            "human_decision": human_decision,
        }
        if verifier_math is not None:
            event["verifier_math"] = _serialize(verifier_math)

        self._write_json(os.path.join(self.traj_dir, f"{node}.json"), event)
        if node not in self._agents:
            self._agents.append(node)
        self._write_json(
            os.path.join(self.traj_dir, "manifest.json"),
            {"incident_id": self.incident_id, "agents": self._agents},
        )
        return event

    def trace_dir(self) -> str:
        return self.traj_dir
