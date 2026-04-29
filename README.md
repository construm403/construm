# ConStruM HRS-B Experiments

This folder contains the minimal code and data needed to run the ConStruM experiments on the cleaned HRS-B benchmark.

## Contents

- `hrs_b/`: cleaned benchmark tables and answer files.
- `ConStrum/`: implementation of tree construction, similarity grouping, and LLM matching.
- `experiments/`: command-line entrypoints for building trees, running one benchmark pair, and running ablations.

Generated files such as trees, embeddings, JSONL predictions, and summaries are written to `ConStrum/output/`.

## Setup

Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key
```

## Build Context Trees

Build a tree for every year used by a benchmark pair. For example, before evaluating `2006 -> 2008`:

```bash
python3 -m experiments.build_tree --year 2006
python3 -m experiments.build_tree --year 2008
```

## Run One Benchmark Pair

```bash
python3 -m experiments.run_benchmark \
  --from-year 2006 \
  --to-year 2008 \
  --answers hrs_b/answers/2006-2008.csv
```

For a quick smoke test, add `--max-cases 5`.

## Run Ablations

The ablation script runs:

- `construm_full`: tree context + similarity-group differentiation.
- `tree_only`: tree context without similarity-group differentiation.
- `diff_only`: similarity-group differentiation without tree context.
- `embedding_top1`: embedding nearest-neighbor baseline.

Smoke test:

```bash
python3 -m experiments.run_hrs_b_ablations \
  --max-files 1 \
  --max-cases 5
```

Full run:

```bash
python3 -m experiments.run_hrs_b_ablations
```

Results are written under `ConStrum/output/ablations/`.

