#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ConStrum.tree.hrs_b_loader import load_hrs_b_year
from ConStrum.tree.tree_builder import build_context_tree_llm


def _construm_root() -> Path:
    return Path(__file__).resolve().parents[1] / "ConStrum"


def main() -> None:
    p = argparse.ArgumentParser(description="Build HRS-B context tree (ConStruM; LLM-driven).")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--model", type=str, default="gpt-5.4")
    p.add_argument("--max-leaf-size", type=int, default=50)
    p.add_argument("--chunk-size", type=int, default=250)
    p.add_argument("--max-depth", type=int, default=6, help="Depth cap (raise to improve granularity; costs more)")
    p.add_argument("--no-leaf-summaries", action="store_true", help="Disable LLM leaf summaries (default on)")
    p.add_argument("--leaf-summary-min-cols", type=int, default=5, help="Minimum leaf size to LLM-summarize")
    args = p.parse_args()

    out_dir = _construm_root() / "output" / "trees"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = str(args.model).replace("/", "_")
    out_path = out_dir / f"tree_hrs_b_{args.year}_{safe_model}.json"
    ckpt_dir = out_dir / f"{out_path.stem}.parts"
    manifest_path = out_dir / f"manifest_hrs_b_{args.year}_{safe_model}.json"

    # Demo philosophy: overwriting is the default.
    # Remove old artifacts so checkpoints never mix across runs.
    try:
        out_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        manifest_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    except OSError:
        pass

    table = load_hrs_b_year(args.year)
    result = build_context_tree_llm(
        table,
        model=str(args.model),
        max_leaf_size=int(args.max_leaf_size),
        chunk_size=int(args.chunk_size),
        max_depth=int(args.max_depth),
        build_leaf_summaries=(not bool(args.no_leaf_summaries)),
        leaf_summary_min_cols=int(args.leaf_summary_min_cols),
        output_json_path=out_path,
        checkpoint_dir=ckpt_dir,
    )

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "year": int(args.year),
        "model": str(args.model),
        "out_path": str(out_path),
        "checkpoint_dir": str(ckpt_dir),
        "meta": result.get("meta", {}),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved tree: {out_path}")


if __name__ == "__main__":
    main()

