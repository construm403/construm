#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

from ConStrum.embeddings.store import load_year, top_k_by_cosine


def _load_pairs(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: List[Tuple[str, str]] = []
        for row in reader:
            source = (row.get("source") or "").strip()
            target = (row.get("target") or "").strip()
            if source and target:
                out.append((source, target))
        return out


def _construm_root() -> Path:
    return Path(__file__).resolve().parents[2] / "ConStrum"


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline: embedding cosine top-1.")
    parser.add_argument("--from-year", type=int, required=True)
    parser.add_argument("--to-year", type=int, required=True)
    parser.add_argument("--answers", type=str, required=True)
    parser.add_argument("--embed-model", type=str, default="text-embedding-3-small")
    parser.add_argument("--top-k", type=int, default=20, help="Recall@k reporting")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases")
    parser.add_argument("--out", type=str, default=None, help="JSONL output path")
    args = parser.parse_args()

    from_year = int(args.from_year)
    to_year = int(args.to_year)
    answers_path = Path(args.answers)
    pairs = _load_pairs(answers_path)
    if args.max_cases and int(args.max_cases) > 0:
        pairs = pairs[: int(args.max_cases)]

    src_emb = load_year(from_year, embed_model=str(args.embed_model), build_if_missing=True)
    tgt_emb = load_year(to_year, embed_model=str(args.embed_model), build_if_missing=True)

    if args.out:
        out_path = Path(args.out)
    else:
        safe_model = str(args.embed_model).replace("/", "_")
        out_path = _construm_root() / "output" / "matching" / f"embed_top1_{from_year}_{to_year}_{safe_model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    recall_at_k = 0
    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for source_id, true_target in pairs:
            total += 1
            query_vec = src_emb.get(source_id)
            ranked = top_k_by_cosine(tgt_emb, query_vec, k=int(args.top_k))
            candidate_ids = [column_id for column_id, _sim in ranked]
            hit_at_k = true_target in candidate_ids
            if hit_at_k:
                recall_at_k += 1
            prediction = candidate_ids[0] if candidate_ids else ""
            is_correct = prediction == true_target
            if is_correct:
                correct += 1

            f.write(
                json.dumps(
                    {
                        "source": source_id,
                        "target_true": true_target,
                        "prediction": prediction,
                        "correct": bool(is_correct),
                        "top_k": int(args.top_k),
                        "hit_at_k": bool(hit_at_k),
                        "candidates": candidate_ids,
                        "mode": "embedding_top1",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "mode": "embedding_top1",
        "from_year": from_year,
        "to_year": to_year,
        "embed_model": str(args.embed_model),
        "answers": str(answers_path),
        "n": total,
        "accuracy": (correct / total) if total else 0.0,
        "recall_at_k": (recall_at_k / total) if total else 0.0,
        "top_k": int(args.top_k),
        "output": str(out_path),
    }
    summary_path = out_path.parent / f"{out_path.name}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
