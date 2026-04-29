from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..tree.llm_client import LLM


_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "matching"
_SYSTEM_MATCH = (_PROMPTS_DIR / "choose_best_match_system.txt").read_text(encoding="utf-8").strip()


def _truncate(s: str, n: int) -> str:
    t = (s or "").strip()
    return t if len(t) <= n else t[: n - 3] + "..."


def _one_line(s: str) -> str:
    # JSONL safety: remove any line-separator characters from model outputs.
    return " ".join((s or "").split())


def _retry_invalid_prediction(
    *,
    llm: LLM,
    model: str,
    system: str,
    user_obj: Dict[str, Any],
    allowed_ids: List[str],
) -> Dict[str, Any]:
    """
    If the model returns an invalid prediction id, we do a single retry by explicitly
    providing the allowed ids. This avoids rule-based "forced fallback" behavior.
    """
    if not allowed_ids:
        return {}
    retry_obj = dict(user_obj)
    retry_obj["allowed_ids"] = allowed_ids
    retry_obj["note"] = (
        "Your previous `prediction` was not one of the candidate ids. "
        "Return `prediction` as EXACTLY one string from allowed_ids."
    )
    out = llm.chat_json(
        model=model,
        system=system,
        user="USER:\n" + json.dumps(retry_obj, ensure_ascii=False),
        response_format_json_object=True,
    )
    return out if isinstance(out, dict) else {}


@dataclass
class Candidate:
    column_id: str
    description: str
    context: Dict[str, Any]


def choose_best_match(
    *,
    llm: LLM,
    model: str,
    source: Dict[str, Any],
    candidates: List[Candidate],
    source_diff: Optional[Dict[str, Any]] = None,
    candidate_diffs: Optional[List[Dict[str, Any]]] = None,
    require_in_candidates: bool = True,
) -> Dict[str, Any]:
    """
    Ask the LLM to select the best target column among candidates.
    Returns a dict with at least {prediction, rationale}.
    """
    cand_payload = []
    for c in candidates:
        leaf_window = c.context.get("leaf_window") or []
        local_window = c.context.get("local_window") or []
        rel_snips = c.context.get("relation_snippets") or []
        wlr = c.context.get("sibling_relation") or c.context.get("within_leaf_relation") or {}
        meta = c.context.get("meta") or {}
        cand_payload.append(
            {
                "column_id": c.column_id,
                "description": _truncate(c.description, 340),
                "path_summaries": [p.get("summary", "") for p in (c.context.get("path") or [])][-6:],
                "leaf_window": leaf_window,
                "local_window": local_window,
                "relation_snippets": rel_snips,
                "within_leaf_relation": wlr,
                "meta": meta,
            }
        )

    src_leaf_window = (source.get("context", {}) or {}).get("leaf_window") or []
    src_local_window = (source.get("context", {}) or {}).get("local_window") or []
    src_rel_snips = (source.get("context", {}) or {}).get("relation_snippets") or []
    src_wlr = (source.get("context", {}) or {}).get("sibling_relation") or (source.get("context", {}) or {}).get("within_leaf_relation") or {}
    user_obj = {
        "task": "schema_match",
        "source": {
            "column_id": source["column_id"],
            "description": _truncate(source.get("description", ""), 600),
            "path_summaries": [p.get("summary", "") for p in (source.get("context", {}).get("path") or [])][-6:],
            "leaf_window": src_leaf_window,
            "local_window": src_local_window,
            "relation_snippets": src_rel_snips,
            "sibling_relation": src_wlr,
        },
        "candidates": cand_payload,
        "source_diff": source_diff or {},
        "candidate_diffs": candidate_diffs or [],
        "output_schema": {"prediction": "one of candidates.column_id", "rationale": "short"},
    }

    system = _SYSTEM_MATCH
    user = "USER:\n" + json.dumps(user_obj, ensure_ascii=False)
    out = llm.chat_json(model=model, system=system, user=user, response_format_json_object=True)

    if not isinstance(out, dict):
        raise ValueError("LLM returned non-dict JSON")
    pred = str(out.get("prediction") or "").strip()
    if require_in_candidates:
        allowed = {c.column_id for c in candidates}
        if pred not in allowed:
            out2 = _retry_invalid_prediction(
                llm=llm,
                model=model,
                system=system,
                user_obj=user_obj,
                allowed_ids=[c.column_id for c in candidates],
            )
            if isinstance(out2, dict):
                pred2 = str(out2.get("prediction") or "").strip()
                if pred2 in allowed:
                    pred = pred2
                    out = out2
    return {"prediction": pred, "rationale": _one_line(str(out.get("rationale") or ""))}

