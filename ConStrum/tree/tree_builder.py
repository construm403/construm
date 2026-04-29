from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import Column, Table
from .llm_client import LLM
from .llm_chunking import cut_table_into_chunks_llm, chunk_result_to_subtables


@dataclass
class ChunkNode:
    node_id: str
    table_name: str
    summary: str
    span: Tuple[int, int]  # inclusive pos range
    children: List["ChunkNode"] = field(default_factory=list)
    # Exact column ids contained in this node (leaf nodes populate this).
    column_ids: List[str] = field(default_factory=list)
    # Optional, leaf-only: within-leaf repeated/parallel slot structures.
    within_leaf_relations: List[Dict[str, Any]] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return not self.children


def _node_to_dict(node: ChunkNode) -> Dict[str, Any]:
    def _inner(n: ChunkNode) -> Dict[str, Any]:
        return {
            "node_id": n.node_id,
            "table_name": n.table_name,
            "summary": n.summary,
            "span": {"start_pos": n.span[0], "end_pos": n.span[1]},
            "children": [_inner(c) for c in n.children],
            "column_ids": list(n.column_ids),
            "within_leaf_relations": n.within_leaf_relations,
        }

    return _inner(node)


def _span_for_table(table: Table) -> Tuple[int, int]:
    if not table.columns:
        return (0, 0)
    return (table.columns[0].pos, table.columns[-1].pos)


def build_context_tree_llm(
    table: Table,
    *,
    model: str = "gpt-5",
    prompts_dir: Optional[Path] = None,
    min_leaf_size: int = 10,
    max_leaf_size: int = 50,
    chunk_size: int = 250,
    output_json_path: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    max_depth: int = 6,
    # Optional annotations / cost controls.
    build_leaf_summaries: bool = True,
    leaf_summary_min_cols: int = 5,
) -> Dict[str, Any]:
    """
    REAL ConStruM-style context tree builder using the LLM chunking pipeline.

    This reuses the 4-agent chunking strategy (Agent1-4) to split tables into
    conceptual subtables, then recurses until leaves are small.
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM()
    node_id_counter = {"n": 0}

    def _next_id() -> str:
        node_id_counter["n"] += 1
        return f"node_{node_id_counter['n']}"

    def _ckpt_filename(span: Tuple[int, int]) -> Optional[Path]:
        if checkpoint_dir is None:
            return None
        s, e = span
        return checkpoint_dir / f"node_{s}_{e}.json"

    def _save_ckpt(node: ChunkNode) -> None:
        p = _ckpt_filename(node.span)
        if p is None:
            return
        try:
            p.write_text(json.dumps(_node_to_dict(node), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _summarize_leaf(cur: Table) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Write a compact, fine-grained summary for a leaf span.
        This helps expose repeated loops / parallel question blocks inside leaves.
        """
        cols = cur.columns
        payload = [{"pos": int(c.pos), "id": c.column_id, "desc": (c.description or "")} for c in cols]

        # Prompt is externalized to keep the open-source version easy to inspect/edit.
        prompt_path = prompts_dir / "tree" / "within_leaf_relations.txt"
        prompt = prompt_path.read_text(encoding="utf-8")

        user_obj = {"table": cur.table_name, "table_description": cur.description, "n": len(cols), "items": payload}
        out = llm.chat_json(
            model=model,
            system="You summarize an ordered span and optionally output within-leaf relations for repeated slots.",
            user=prompt + "\n\n=== Input ===\n" + json.dumps(user_obj, ensure_ascii=False),
            response_format_json_object=True,
        )
        summary = ""
        rels: List[Dict[str, Any]] = []
        if isinstance(out, dict):
            summary = str(out.get("summary") or "").strip()
            ca = out.get("within_leaf_relations")
            if isinstance(ca, list):
                leaf_ids = {c.column_id for c in cols}
                for item in ca:
                    if not isinstance(item, dict):
                        continue
                    members = item.get("members")
                    if not isinstance(members, list) or len(members) < 2:
                        continue
                    filtered_members: List[Dict[str, Any]] = []
                    for m in members:
                        if not isinstance(m, dict):
                            continue
                        mid = str(m.get("id") or "").strip()
                        if mid and (mid in leaf_ids):
                            filtered_members.append(
                                {
                                    "id": mid,
                                    "note": str(m.get("note") or "").strip(),
                                }
                            )
                    if len(filtered_members) < 2:
                        continue
                    rels.append(
                        {
                            "repeat_key": str(item.get("repeat_key") or "").strip(),
                            "members": filtered_members,
                            "anchors": item.get("anchors") if isinstance(item.get("anchors"), dict) else {},
                            "note": str(item.get("note") or "").strip(),
                        }
                    )

        if not summary:
            summary = f"Leaf: {cur.table_name} ({len(cols)} columns)"
        return summary, rels

    def _recurse(cur: Table, depth: int) -> ChunkNode:
        span = _span_for_table(cur)
        ncols = len(cur.columns)

        # Depth cap for practical testing; still produces a real (LLM) tree prefix.
        if (depth >= max_depth) or (ncols <= max_leaf_size):
            if build_leaf_summaries and (ncols >= int(leaf_summary_min_cols)):
                leaf_summary, rels = _summarize_leaf(cur)
            else:
                leaf_summary = f"Leaf: {cur.table_name} ({ncols} columns)"
                rels = []
            node = ChunkNode(
                node_id=_next_id(),
                table_name=cur.table_name,
                summary=leaf_summary,
                span=span,
                column_ids=[c.column_id for c in cur.columns],
                within_leaf_relations=rels,
            )
            _save_ckpt(node)
            return node

        # LLM chunking for this node
        chunk_result = cut_table_into_chunks_llm(
            cur,
            llm=llm,
            model=model,
            prompts_dir=prompts_dir,
            chunk_size=chunk_size,
        )
        subtables = chunk_result_to_subtables(cur, chunk_result)

        # Fallback: if chunking fails to produce progress, force a leaf at this node.
        if (not subtables) or (len(subtables) == 1 and len(subtables[0].columns) == ncols):
            return _recurse(cur, depth=max_depth)

        ordered_sequence = []
        if isinstance(chunk_result.get("conceptual_chunks"), dict):
            osq = chunk_result["conceptual_chunks"].get("ordered_sequence")
            if isinstance(osq, list):
                ordered_sequence = [str(x) for x in osq]

        table_summary = ""
        if isinstance(chunk_result.get("conceptual_chunks"), dict):
            table_summary = str(chunk_result["conceptual_chunks"].get("table_summary") or "")
        node = ChunkNode(
            node_id=_next_id(),
            table_name=cur.table_name,
            summary=table_summary or f"Internal: {cur.table_name} ({ncols} columns)",
            span=span,
        )

        node.children = [_recurse(st, depth + 1) for st in subtables]
        _save_ckpt(node)
        return node

    root = _recurse(table, 0)

    column_index: Dict[str, Dict[str, Any]] = {}

    def _walk(node: ChunkNode, path: List[str]) -> None:
        new_path = path + [node.node_id]
        if node.is_leaf():
            for col_id in node.column_ids:
                column_index[col_id] = {"leaf_node": node.node_id, "path": new_path}
            return
        for ch in node.children:
            _walk(ch, new_path)

    _walk(root, [])

    result = {
        "root": _node_to_dict(root),
        "column_index": column_index,
        "meta": {
            "builder": "llm",
            "model": model,
            "min_leaf_size": int(min_leaf_size),
            "max_leaf_size": int(max_leaf_size),
            "chunk_size": int(chunk_size),
            "max_depth": int(max_depth),
            "table_name": table.table_name,
            "n_columns": len(table.columns),
            "build_leaf_summaries": bool(build_leaf_summaries),
            "leaf_summary_min_cols": int(leaf_summary_min_cols),
        },
    }

    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result

