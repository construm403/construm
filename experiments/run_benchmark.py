#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

from ConStrum.embeddings.store import load_year, top_k_by_cosine
from ConStrum.hypergraph.diff_blocks import DiffBlockCache, build_diff_block
from ConStrum.hypergraph.similarity import materialize_groups_within_set, neighbors_above_tau
from ConStrum.matching.llm_matcher import Candidate, choose_best_match
from ConStrum.matching.tree_context import context_for_column, load_tree_for_year
from ConStrum.tree.hrs_b_loader import load_hrs_b_year
from ConStrum.tree.llm_client import LLM


def _load_pairs(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        out: List[Tuple[str, str]] = []
        for row in r:
            s = (row.get("source") or "").strip()
            t = (row.get("target") or "").strip()
            if s and t:
                out.append((s, t))
        return out


def _local_window(table, col_id: str, *, radius: int = 6, max_desc: int = 180):
    cols = table.columns
    idx = next((i for i, c in enumerate(cols) if c.column_id == col_id), None)
    if idx is None:
        raise KeyError(f"column_id not found in table: {col_id}")
    lo = max(0, idx - radius)
    hi = min(len(cols), idx + radius + 1)
    out = []
    for c in cols[lo:hi]:
        d = (c.description or "").strip()
        if len(d) > max_desc:
            d = d[: max_desc - 3] + "..."
        out.append({"column_id": c.column_id, "description": d})
    return out


def _construm_root() -> Path:
    return Path(__file__).resolve().parents[1] / "ConStrum"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the HRS-B schema matching benchmark with ConStruM.")
    ap.add_argument("--from-year", type=int, required=True)
    ap.add_argument("--to-year", type=int, required=True)
    ap.add_argument("--answers", type=str, required=True)
    ap.add_argument("--model", type=str, default="gpt-5.4")
    ap.add_argument("--embed-model", type=str, default="text-embedding-3-small")
    ap.add_argument("--no-tree", action="store_true", help="Disable tree context for the diff-only ablation.")
    ap.add_argument("--top-k", type=int, default=20, help="Upstream shortlist size C0 (embedding top-k)")
    ap.add_argument("--llm-k", type=int, default=24, help="How many candidates to show final LLM (after expansion)")
    ap.add_argument("--tau", type=float, default=0.95, help="Similarity threshold for hypergraph grouping/expansion")
    ap.add_argument("--expand-top", type=int, default=5, help="Expand neighbors for top-N strong candidates in C0")
    ap.add_argument("--expand-per", type=int, default=3, help="Max neighbors to add per strong candidate")
    ap.add_argument(
        "--no-diff",
        action="store_true",
        help="Disable similarity-group expansion and grouped differentiation for the tree-only ablation.",
    )
    ap.add_argument("--max-cases", type=int, default=0, help="0 means all cases")
    ap.add_argument("--window-radius", type=int, default=6, help="How many neighbors each side to include as local window")
    ap.add_argument("--out", type=str, default=None, help="JSONL output path (default under ConStrum/output/matching)")
    args = ap.parse_args()

    if bool(args.no_diff):
        args.expand_top = 0
        args.expand_per = 0

    from_year = int(args.from_year)
    to_year = int(args.to_year)
    answers_path = Path(args.answers)
    pairs = _load_pairs(answers_path)
    if args.max_cases and int(args.max_cases) > 0:
        pairs = pairs[: int(args.max_cases)]
    if not pairs:
        raise ValueError(f"no (source,target) pairs found in answers CSV: {answers_path}")

    # Load tables
    src_table = load_hrs_b_year(from_year)
    tgt_table = load_hrs_b_year(to_year)
    src_desc: Dict[str, str] = {c.column_id: c.description for c in src_table.columns}
    tgt_desc: Dict[str, str] = {c.column_id: c.description for c in tgt_table.columns}
    src_pos: Dict[str, int] = {c.column_id: int(c.pos) for c in src_table.columns}
    tgt_pos: Dict[str, int] = {c.column_id: int(c.pos) for c in tgt_table.columns}
    src_n = len(src_table.columns) or 1
    tgt_n = len(tgt_table.columns) or 1

    use_diff = not bool(args.no_diff)
    use_local_window = not bool(args.no_tree)

    src_tree = None if bool(args.no_tree) else load_tree_for_year(from_year, model=str(args.model), trees_dir=None)
    tgt_tree = None if bool(args.no_tree) else load_tree_for_year(to_year, model=str(args.model), trees_dir=None)

    # Embeddings (upstream matcher + similarity hypergraph basis)
    src_emb = load_year(from_year, embed_model=str(args.embed_model), build_if_missing=True)
    tgt_emb = load_year(to_year, embed_model=str(args.embed_model), build_if_missing=True)

    # Output
    out_path: Path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = _construm_root() / "output" / "matching" / f"bench_{from_year}_{to_year}_{args.model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Robust JSONL writing:
    # - Write to a unique temp file first (prevents concurrent-run interleaving/corruption)
    # - fsync + atomic replace into the requested output path
    tmp_path = out_path.parent / f".{out_path.name}.pid{os.getpid()}.{int(time.time() * 1000)}.tmp"
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)

    def _write_jsonl(obj: Dict[str, object]) -> None:
        s = json.dumps(obj, ensure_ascii=False) + "\n"
        b = s.encode("utf-8", "replace")
        off = 0
        while off < len(b):
            n = os.write(fd, b[off:])
            if n <= 0:
                raise RuntimeError("os.write returned 0")
            off += n

    llm = LLM()
    diff_cache = None
    if use_diff:
        cache_namespace = "notree" if bool(args.no_tree) else ""
        diff_cache = DiffBlockCache.load(model=str(args.model), namespace=cache_namespace)
    correct = 0
    recall_at_k = 0
    total = 0

    try:
        for (src_id, true_tgt) in pairs:
            total += 1
            # ---------- Upstream matcher (embedding retrieval): C0 ----------
            qv = src_emb.get(src_id)
            ranked = top_k_by_cosine(tgt_emb, qv, k=int(args.top_k))
            cand_ids = [cid for (cid, _sim) in ranked]
            if not cand_ids:
                raise RuntimeError("embedding retrieval returned empty candidate set")
            if true_tgt in cand_ids:
                recall_at_k += 1

            # Candidate expansion using similarity-neighbor links (tau)
            tau = float(args.tau)
            if int(args.expand_top) > 0 and int(args.expand_per) > 0:
                # We keep some slots for expanded candidates so that expansion can actually
                # change what the LLM sees (otherwise C0 can crowd out all new neighbors).
                llm_k = int(args.llm_k)
                if llm_k < 1:
                    raise ValueError("--llm-k must be >= 1")

                # Reserve a fixed number of slots for expansion so that expansion can actually
                # change what the LLM sees (otherwise C0 can crowd out all new neighbors).
                # Kept small/fixed for the lightweight demo.
                exp_slots = min(12, llm_k - 1)
                base_keep = min(len(cand_ids), llm_k - exp_slots)

                c0 = list(cand_ids)
                c0_set = set(c0)
                extras: List[str] = []
                extras_set = set()

                def _scan_seeds(seeds: List[str]) -> None:
                    if exp_slots <= 0:
                        return
                    per_seed_cap = max(int(args.expand_per), exp_slots)
                    for seed in seeds:
                        if len(extras) >= exp_slots:
                            return
                        for nb, _sim in neighbors_above_tau(
                            tgt_emb, seed, tau=tau, max_neighbors=per_seed_cap
                        ):
                            if nb in c0_set or nb in extras_set:
                                continue
                            extras.append(nb)
                            extras_set.add(nb)
                            if len(extras) >= exp_slots:
                                return

                # Fast path: expand only top-N seeds first.
                seed_n = min(len(c0), int(args.expand_top))
                _scan_seeds(c0[:seed_n])

                # Fallback: if expansion didn't add anything, keep scanning deeper seeds in C0.
                if not extras and seed_n < len(c0):
                    _scan_seeds(c0[seed_n:])

                # Final list: keep C0 head, then extras, then backfill with remaining C0.
                final: List[str] = []
                final_set = set()

                for cid in c0[:base_keep]:
                    if cid not in final_set:
                        final.append(cid)
                        final_set.add(cid)
                for cid in extras[:exp_slots]:
                    if cid not in final_set:
                        final.append(cid)
                        final_set.add(cid)
                for cid in c0[base_keep:]:
                    if len(final) >= llm_k:
                        break
                    if cid not in final_set:
                        final.append(cid)
                        final_set.add(cid)

                cand_ids = final[:llm_k]
            else:
                # Cap working set shown to LLM
                cand_ids = cand_ids[: int(args.llm_k)]
            # ---------- Context packs (tree) ----------
            src_ctx = context_for_column(src_tree, src_id) if src_tree is not None else {}
            win_r = int(args.window_radius)
            src_payload = {
                "column_id": src_id,
                "description": src_desc.get(src_id, ""),
                "context": {
                    **src_ctx,
                    "local_window": _local_window(src_table, src_id, radius=win_r) if use_local_window else [],
                },
            }

            # ---------- Similarity groups (match-time materialization within set) ----------
            source_diff = {}
            candidate_diffs: List[Dict[str, object]] = []
            if use_diff:
                src_neighbors = [nb for (nb, _sim) in neighbors_above_tau(src_emb, src_id, tau=tau, max_neighbors=24)]
                src_group = [src_id] + src_neighbors
                if len(src_group) > 1:
                    src_group = sorted(set(src_group), key=lambda x: int(src_pos.get(x, 10**9)))
                    source_diff = build_diff_block(
                        llm=llm,
                        model=str(args.model),
                        year=from_year,
                        members=src_group[:24],
                        tree=src_tree,
                        descriptions=src_desc,
                        cache=diff_cache,
                    )
                groups = materialize_groups_within_set(tgt_emb, cand_ids, tau=tau)
                for g in groups[:6]:
                    g = sorted(set(g), key=lambda x: int(tgt_pos.get(x, 10**9)))
                    candidate_diffs.append(
                        {
                            "members": g,
                            "diff": build_diff_block(
                                llm=llm,
                                model=str(args.model),
                                year=to_year,
                                members=g[:24],
                                tree=tgt_tree,
                                descriptions=tgt_desc,
                                cache=diff_cache,
                            ),
                        }
                    )

            cands: List[Candidate] = []
            # Provide embedding sim + normalized position metadata for disambiguation
            sim_by_cid = {cid: sim for (cid, sim) in ranked}
            rs = float(src_pos.get(src_id, 1)) / float(src_n)
            for cid in cand_ids:
                rt = float(tgt_pos.get(cid, 1)) / float(tgt_n)
                cands.append(
                    Candidate(
                        column_id=cid,
                        description=tgt_desc.get(cid, ""),
                        context={
                            **(context_for_column(tgt_tree, cid) if tgt_tree is not None else {}),
                            "local_window": _local_window(tgt_table, cid, radius=win_r) if use_local_window else [],
                            "meta": {
                                "embed_cosine": float(sim_by_cid.get(cid, 0.0)),
                                "pos_norm": float(rt),
                                "pos_delta": float(abs(rt - rs)),
                            },
                        },
                    )
                )

            pred = choose_best_match(
                llm=llm,
                model=str(args.model),
                source=src_payload,
                candidates=cands,
                source_diff=source_diff,
                candidate_diffs=candidate_diffs,
            )
            is_correct = (pred["prediction"] == true_tgt)
            if is_correct:
                correct += 1

            rec = {
                "source": src_id,
                "target_true": true_tgt,
                "prediction": pred["prediction"],
                "correct": bool(is_correct),
                "top_k": int(args.top_k),
                "hit_at_k": bool(true_tgt in [cid for (cid, _sim) in ranked]),
                "candidates": cand_ids,
                "rationale": pred.get("rationale", ""),
                "mode": "construm",
            }
            _write_jsonl(rec)
    finally:
        try:
            os.fsync(fd)
        except Exception:
            pass
        os.close(fd)

    os.replace(str(tmp_path), str(out_path))

    acc = (correct / total) if total else 0.0
    rck = (recall_at_k / total) if total else 0.0
    summary = {
        "from_year": from_year,
        "to_year": to_year,
        "model": str(args.model),
        "embed_model": str(args.embed_model),
        "answers": str(answers_path),
        "n": total,
        "accuracy": acc,
        "recall_at_k": rck,
        "top_k": int(args.top_k),
        "tau": float(args.tau),
        "use_tree": bool(src_tree is not None and tgt_tree is not None),
        "use_diff": bool(use_diff),
        "use_local_window": bool(use_local_window),
        "pipeline": "construm",
        "output": str(out_path),
    }
    summary_path = out_path.parent / f"{out_path.name}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

