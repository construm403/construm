from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..tree.llm_client import LLM
from ..matching.tree_context import TreeContext, context_for_column


_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "hypergraph"
_DIFF_SYSTEM = (_PROMPTS_DIR / "diff_block_system.txt").read_text(encoding="utf-8").strip()
_DIFF_SYSTEM_HASH = hashlib.sha1(_DIFF_SYSTEM.encode("utf-8")).hexdigest()[:8]


def _default_cache_path(model: str, *, namespace: str = "") -> Path:
    out_dir = Path(__file__).resolve().parents[1] / "output" / "hypergraph"
    out_dir.mkdir(parents=True, exist_ok=True)
    ns = f"_{namespace}" if namespace else ""
    return out_dir / f"diff_cache_{model.replace('/','_')}_{_DIFF_SYSTEM_HASH}{ns}.json"


def _key(year: int, members: List[str]) -> str:
    return f"{year}|" + ",".join(sorted(members))


@dataclass
class DiffBlockCache:
    model: str
    path: Path
    data: Dict[str, Any]

    @classmethod
    def load(
        cls, model: str, path: Optional[Path] = None, *, namespace: str = ""
    ) -> "DiffBlockCache":
        if path is None:
            path = _default_cache_path(model, namespace=namespace)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}
        return cls(model=str(model), path=path, data=data if isinstance(data, dict) else {})

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_diff_block(
    *,
    llm: LLM,
    model: str,
    year: int,
    members: List[str],
    tree: Optional[TreeContext],
    descriptions: Dict[str, str],
    cache: Optional[DiffBlockCache] = None,
) -> Dict[str, Any]:
    """
    Build a ConStruM-style grouped differentiation block for one non-singleton similarity group.
    Output schema (JSON): { "summary": str, "cues": [{column_id, cue}] }
    """
    members = [m for m in members if m in descriptions]
    if len(members) <= 1:
        return {"summary": "", "cues": []}

    if cache is not None:
        k = _key(year, members)
        if k in cache.data:
            return cache.data[k]

    payload = []
    for cid in members:
        ctx = context_for_column(tree, cid) if tree is not None else {}
        leaf_ids = ctx.get("leaf_window") or []
        leaf_desc = []
        for wid in leaf_ids[:14]:
            if wid == cid:
                continue
            d = (descriptions.get(wid, "") or "").strip()
            if len(d) > 160:
                d = d[:157] + "..."
            leaf_desc.append({"column_id": wid, "description": d})
        payload.append(
            {
                "column_id": cid,
                "description": descriptions.get(cid, ""),
                "path_summaries": [p.get("summary", "") for p in (ctx.get("path") or [])][-6:],
                "leaf_window": leaf_ids,
                "leaf_window_desc": leaf_desc,
                "sibling_relation": ctx.get("sibling_relation") or ctx.get("within_leaf_relation") or {},
            }
        )

    system = _DIFF_SYSTEM
    user = "USER:\n" + json.dumps({"year": year, "group": payload}, ensure_ascii=False)
    out = llm.chat_json(model=model, system=system, user=user, response_format_json_object=True)

    if not isinstance(out, dict):
        out = {"summary": "", "cues": []}

    if cache is not None:
        cache.data[_key(year, members)] = out
        cache.save()
    return out

