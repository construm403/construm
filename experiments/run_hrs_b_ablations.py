#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def _construm_root() -> Path:
    return Path(__file__).resolve().parents[1] / "ConStrum"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper ablations for the HRS-B benchmark.")
    parser.add_argument("--answers-dir", type=str, default="hrs_b/answers")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--embed-model", type=str, default="text-embedding-3-small")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--max-files", type=int, default=0, help="0 means all answer files")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases per answer file")
    args = parser.parse_args()

    answers_dir = Path(args.answers_dir)
    paths = sorted(answers_dir.glob("*.csv"))
    if args.max_files and int(args.max_files) > 0:
        paths = paths[: int(args.max_files)]
    if not paths:
        raise FileNotFoundError(f"No answers CSVs found under: {answers_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        _construm_root() / "output" / "ablations" / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("embedding_top1", ["-m", "experiments.baselines.run_embedding_top1"]),
        ("construm_full", ["-m", "experiments.run_benchmark"]),
        ("tree_only", ["-m", "experiments.run_benchmark", "--no-diff"]),
        ("diff_only", ["-m", "experiments.run_benchmark", "--no-tree"]),
    ]

    summaries = []
    for answers_path in paths:
        match = re.match(r"^(\d{4})-(\d{4})\.csv$", answers_path.name)
        if not match:
            continue
        from_year, to_year = int(match.group(1)), int(match.group(2))

        for variant_name, module_args in variants:
            out_jsonl = out_dir / f"{variant_name}_{from_year}_{to_year}.jsonl"
            cmd = [
                sys.executable,
                *module_args,
                "--from-year",
                str(from_year),
                "--to-year",
                str(to_year),
                "--answers",
                str(answers_path),
                "--out",
                str(out_jsonl),
            ]
            if int(args.max_cases) > 0:
                cmd += ["--max-cases", str(args.max_cases)]
            if variant_name == "embedding_top1":
                cmd += ["--embed-model", str(args.embed_model)]
            else:
                cmd += ["--model", str(args.model), "--embed-model", str(args.embed_model)]

            subprocess.run(cmd, check=True)

            summary_path = out_jsonl.parent / f"{out_jsonl.name}.summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                summary = {
                    "mode": variant_name,
                    "from_year": from_year,
                    "to_year": to_year,
                    "output": str(out_jsonl),
                }
            summary["variant"] = variant_name
            summary["answers_file"] = answers_path.name
            summaries.append(summary)

    aggregate_path = out_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "n_summaries": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
