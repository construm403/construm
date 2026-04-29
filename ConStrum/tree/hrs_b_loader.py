from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

from .schema import Column, Table


def _read_csv_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def load_hrs_b_year(
    year: int,
    *,
    hrs_b_dir: Optional[Path] = None,
) -> Table:
    """
    Load one HRS-B cleaned dataset year as a single Table.

    Expected file layout (relative to repo root):
      hrs_b/data/{YEAR}.csv

    The development workspace may also use `hrs_b_benchmark/`; the loader accepts
    either directory name.

    Expected CSV schema:
      column_id,column_description
    """
    if hrs_b_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        hrs_b_dir = repo_root / "hrs_b"
        if not hrs_b_dir.exists():
            hrs_b_dir = repo_root / "hrs_b_benchmark"

    csv_path = hrs_b_dir / "data" / f"{year}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing HRS-B year CSV: {csv_path}")

    cols: list[Column] = []
    for i, row in enumerate(_read_csv_rows(csv_path), start=1):
        cid = (row.get("column_id") or "").strip()
        desc = (row.get("column_description") or "").strip()
        if not cid:
            raise ValueError(f"Empty column_id at row {i} in {csv_path}")
        cols.append(Column(pos=i, column_id=cid, description=desc))

    return Table(
        table_name=f"HRS_B_{year}",
        description=f"HRS-B cleaned codebook table for year {year}.",
        columns=cols,
    )

