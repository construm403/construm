from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TreeContext:
    year: int
    tree_path: Path
    root: Dict[str, Any]
    column_index: Dict[str, Any]
    node_by_id: Dict[str, Dict[str, Any]]


def _index_nodes(root: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    def _walk(n: Dict[str, Any]) -> None:
        nid = n.get("node_id")
        if isinstance(nid, str):
            out[nid] = n
        for ch in (n.get("children") or []):
            if isinstance(ch, dict):
                _walk(ch)

    _walk(root)
    return out


def load_tree_for_year(
    year: int,
    *,
    model: str = "gpt-5",
    trees_dir: Optional[Path] = None,
) -> TreeContext:
    safe_model = model.replace("/", "_")

    if trees_dir is not None:
        tree_path = trees_dir / f"tree_hrs_b_{year}_{safe_model}.json"
        if not tree_path.exists():
            raise FileNotFoundError(f"Missing tree JSON: {tree_path}")
    else:
        base = Path(__file__).resolve().parents[1] / "output"
        tree_path = base / "trees" / f"tree_hrs_b_{year}_{safe_model}.json"
        if not tree_path.exists():
            raise FileNotFoundError(
                "Missing tree JSON. Build it first with:\n"
                f"  python -m experiments.build_tree --year {year} --model {safe_model}\n"
                f"Expected:\n  {tree_path}"
            )
    obj = json.loads(tree_path.read_text(encoding="utf-8"))
    root = obj["root"]
    column_index = obj["column_index"]
    node_by_id = _index_nodes(root)
    return TreeContext(year=year, tree_path=tree_path, root=root, column_index=column_index, node_by_id=node_by_id)


def context_for_column(
    tree: TreeContext,
    column_id: str,
    *,
    max_path_nodes: int = 8,
    max_leaf_columns: int = 40,
) -> Dict[str, Any]:
    """
    Return a compact, prompt-friendly context dict for one column_id.
    """
    entry = tree.column_index.get(column_id)
    if not entry:
        return {"column_id": column_id, "found": False}

    path_ids = entry.get("path") or []
    if not isinstance(path_ids, list):
        path_ids = []
    path_ids = [p for p in path_ids if isinstance(p, str)]
    if len(path_ids) > max_path_nodes:
        path_ids = path_ids[-max_path_nodes:]

    path_nodes: List[Dict[str, Any]] = []
    for nid in path_ids:
        n = tree.node_by_id.get(nid)
        if not n:
            continue
        path_nodes.append(
            {
                "node_id": nid,
                "table_name": n.get("table_name", ""),
                "summary": (n.get("summary") or ""),
                "span": n.get("span") or {},
            }
        )

    leaf_node_id = entry.get("leaf_node")
    leaf_node = tree.node_by_id.get(leaf_node_id) if isinstance(leaf_node_id, str) else None
    leaf_cols: List[str] = []
    leaf_window: List[str] = []
    leaf_summary: str = ""
    leaf_size_total: int = 0
    sibling_relation: Dict[str, Any] = {}
    if leaf_node:
        leaf_cols = list(leaf_node.get("column_ids") or [])
        leaf_size_total = len(leaf_cols)
        leaf_summary = str(leaf_node.get("summary") or "")
        try:
            j = leaf_cols.index(column_id)
            lo = max(0, j - 10)
            hi = min(len(leaf_cols), j + 11)
            leaf_window = leaf_cols[lo:hi]
        except ValueError:
            leaf_window = []
        if len(leaf_cols) > max_leaf_columns:
            leaf_cols = leaf_cols[:max_leaf_columns]

        # Leaf-level within-leaf relations (optional; produced by within-leaf prompt)
        rels = leaf_node.get("within_leaf_relations", [])
        if isinstance(rels, list):
            for r in rels:
                if not isinstance(r, dict):
                    continue
                members = r.get("members")
                if not isinstance(members, list):
                    continue
                ids = []
                for m in members:
                    if isinstance(m, dict):
                        mid = str(m.get("id") or "").strip()
                        if mid:
                            ids.append(mid)
                if column_id in ids:
                    sibling_relation = {
                        "repeat_key": str(r.get("repeat_key") or "").strip(),
                        "members": members,
                        "anchors": r.get("anchors") if isinstance(r.get("anchors"), dict) else {},
                        "note": str(r.get("note") or "").strip(),
                    }
                    break

    return {
        "column_id": column_id,
        "found": True,
        "leaf_node": leaf_node_id,
        "leaf_summary": leaf_summary,
        "leaf_size": int(leaf_size_total),
        "path": path_nodes,
        "leaf_columns": leaf_cols,
        "leaf_window": leaf_window,
        # Public-facing name (preferred)
        "sibling_relation": sibling_relation,
        # Backward-compatible alias (older prompts / outputs)
        "within_leaf_relation": sibling_relation,
    }

