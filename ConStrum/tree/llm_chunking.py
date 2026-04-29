from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import LLM
from .schema import Column, Table


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _columns_to_json_block(cols: List[Column]) -> str:
    # idx must be an int index; name is the cleaned non-sensitive variable token.
    return json.dumps(
        [{"idx": int(c.pos), "name": c.column_id, "description": c.description} for c in cols],
        ensure_ascii=False,
    )


def _merge_assignments(across_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segs: List[Dict[str, Any]] = []
    for out in across_chunks:
        segs.extend(out.get("assignment", []) or [])
    segs.sort(key=lambda x: (int(x.get("start", 0)), int(x.get("end", 0))))
    merged: List[Dict[str, Any]] = []
    for s in segs:
        if not merged:
            merged.append(dict(s))
            continue
        last = merged[-1]
        if (
            last.get("conceptual_chunk") == s.get("conceptual_chunk")
            and int(last.get("end")) + 1 == int(s.get("start"))
        ):
            last["end"] = s["end"]
        else:
            merged.append(dict(s))
    return merged


def _sample_every_k(table: Table, k: int, start_offset: int) -> List[Column]:
    return [c for i, c in enumerate(table.columns) if i >= start_offset and ((i - start_offset) % k == 0)]


def cut_table_into_chunks_llm(
    table: Table,
    *,
    llm: LLM,
    model: str,
    prompts_dir: Path,
    chunk_size: int = 250,
    sample_interval: int = 10,
    sample_offsets: Tuple[int, ...] = (0, 5),
    agent4_min_run: int = 5,
    agent4_max_switches: int = 3,
) -> Dict[str, Any]:
    """
    Real ConStruM-style 4-agent chunking:
    Agent1 (coarse grouping) per chunk
    Agent2 (theme inference) on samples
    Agent3 (synthesize conceptual chunks)
    Agent4 (boundaries / assignment) per chunk
    """
    p1 = _load_prompt(prompts_dir / "chunking" / "rough_grouping.txt")
    p2 = _load_prompt(prompts_dir / "chunking" / "infer_theme.txt")
    p3 = _load_prompt(prompts_dir / "chunking" / "clear_grouping.txt")
    p4 = _load_prompt(prompts_dir / "chunking" / "find_boundaries.txt")

    print(f"[Chunking] {table.table_name}: {len(table.columns)} columns, chunk_size={chunk_size}")

    # Agent 2 runs (sampling)
    agent2_runs: List[Dict[str, Any]] = []
    for offset in sample_offsets:
        sampled = _sample_every_k(table, sample_interval, int(offset))
        if not sampled:
            continue
        print(f"[Agent2] sample_offset={offset} sampled={len(sampled)}")
        user = p2 + "\n\n=== Sampled Columns ===\n" + _columns_to_json_block(sampled)
        out = llm.chat_json(model=model, system="You infer broad themes from samples.", user=user)
        agent2_runs.append(out)

    # Agent 1 runs per chunk
    cols = table.columns
    agent1_all_chunks: List[Any] = []
    chunk_ranges: List[Tuple[int, int]] = []
    for i in range(0, len(cols), chunk_size):
        sub = cols[i : i + chunk_size]
        if not sub:
            break
        print(f"[Agent1] chunk {i//chunk_size + 1}: pos {sub[0].pos}-{sub[-1].pos} ({len(sub)} cols)")
        user = p1 + "\n\n=== Input Columns (Chunk) ===\n" + _columns_to_json_block(sub)
        out = llm.chat_json(model=model, system="You make a coarse first-pass grouping.", user=user)
        agent1_all_chunks.append(out)
        chunk_ranges.append((int(sub[0].pos), int(sub[-1].pos)))

    # Agent 3 synthesis
    print("[Agent3] synthesize conceptual chunks")
    user3 = (
        p3.replace("<INSERT_AGENT_1_RESULT_HERE>", json.dumps(agent1_all_chunks, ensure_ascii=False))
        .replace("<INSERT_AGENT_2_RESULTS_HERE>", json.dumps(agent2_runs, ensure_ascii=False))
    )
    agent3_out = llm.chat_json(model=model, system="You synthesize coarse groups and sampled overviews.", user=user3)

    # Agent 4 boundaries per chunk
    per_chunk_outputs: List[Dict[str, Any]] = []
    last_assigned_label: Optional[str] = None
    ordered_sequence = agent3_out.get("ordered_sequence", []) if isinstance(agent3_out, dict) else []
    for i, (lo, hi) in enumerate(chunk_ranges):
        sub = cols[i * chunk_size : (i + 1) * chunk_size]
        print(f"[Agent4] chunk {i+1}: pos {lo}-{hi} ({len(sub)} cols)")
        start_label_hint = None
        if last_assigned_label and ordered_sequence and isinstance(ordered_sequence, list):
            if last_assigned_label in ordered_sequence:
                start_label_hint = last_assigned_label

        constraints = {"min_run": int(agent4_min_run), "max_switches": int(agent4_max_switches)}
        start_hint = start_label_hint if start_label_hint else "null"
        user4 = (
            p4.replace("<INSERT_CONCEPTUAL_CHUNKS_JSON_HERE>", json.dumps(agent3_out, ensure_ascii=False))
            .replace("<INSERT_START_LABEL_HINT_OR_NULL>", start_hint)
            .replace('{"min_run": <int>, "max_switches": <int>}', json.dumps(constraints))
            .replace("<INSERT_COLUMN_LIST_HERE>", _columns_to_json_block(sub))
        )
        a4 = llm.chat_json(model=model, system="You determine local boundaries inside this chunk.", user=user4)
        per_chunk_outputs.append({"chunk_start": lo, "chunk_end": hi, "agent4_output": a4})
        assignments = (a4 or {}).get("assignment", []) if isinstance(a4, dict) else []
        if assignments:
            last_assigned_label = assignments[-1].get("conceptual_chunk")

    global_segments = _merge_assignments([o["agent4_output"] for o in per_chunk_outputs])
    print(f"[Chunking] global_segments={len(global_segments)}")
    return {
        "conceptual_chunks": agent3_out,
        "global_segments": global_segments,
        "per_chunk_outputs": per_chunk_outputs,
        "table_info": {"name": table.table_name, "total_columns": len(table.columns), "chunk_size": int(chunk_size)},
    }


def _slice_by_pos_range(table: Table, start_pos: int, end_pos: int) -> List[Column]:
    return [c for c in table.columns if start_pos <= c.pos <= end_pos]


def chunk_result_to_subtables(table: Table, chunk_result: Dict[str, Any]) -> List[Table]:
    """
    Convert `global_segments` from Agent4 merge into actual subtable objects.
    """
    concepts = (chunk_result.get("conceptual_chunks") or {}).get("conceptual_chunks", [])
    desc_by_label = {c.get("label"): c.get("description", "") for c in concepts if isinstance(c, dict)}

    subtables: List[Table] = []
    for i, seg in enumerate(chunk_result.get("global_segments", []) or [], start=1):
        label = seg.get("conceptual_chunk", f"Chunk_{i}")
        start_pos = int(seg.get("start"))
        end_pos = int(seg.get("end"))
        cols = _slice_by_pos_range(table, start_pos, end_pos)
        if not cols:
            continue
        desc = desc_by_label.get(label, "")
        subtables.append(
            Table(
                table_name=f"{table.table_name}_{str(label).replace(' ', '_')}",
                description=str(desc or ""),
                columns=cols,
            )
        )
    return subtables

