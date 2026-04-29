from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Column:
    """
    A single questionnaire/codebook row.

    - `pos` is a 1-based position in the CSV (stable ordering for chunk spans).
    - `column_id` is the cleaned, non-sensitive identifier (e.g. HRS2006Q123).
    """

    pos: int
    column_id: str
    description: str


@dataclass(frozen=True)
class Table:
    table_name: str
    description: str
    columns: List[Column]
