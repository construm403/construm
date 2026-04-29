from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..embeddings.store import Embeddings


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def neighbors_above_tau(
    emb: Embeddings,
    column_id: str,
    *,
    tau: float,
    max_neighbors: int = 24,
) -> List[Tuple[str, float]]:
    """
    Return up to max_neighbors neighbors with cosine >= tau (excluding self), sorted by cosine desc.
    """
    v = emb.get(column_id)
    sims = emb.mat @ v
    idx = np.where(sims >= float(tau))[0]
    out: List[Tuple[str, float]] = []
    for i in idx.tolist():
        cid = emb.ids[i]
        if cid == column_id:
            continue
        out.append((cid, float(sims[i])))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[: int(max_neighbors)]


def materialize_groups_within_set(
    emb: Embeddings,
    cols: Sequence[str],
    *,
    tau: float,
) -> List[List[str]]:
    """
    Match-time: materialize thresholded links within a working set C, then take connected components.
    """
    cols = [c for c in cols if c in emb.id_to_row]
    m = len(cols)
    if m <= 1:
        return []
    idx = np.array([emb.id_to_row[c] for c in cols], dtype=np.int64)
    sub = emb.mat[idx]  # [m, d]
    sims = sub @ sub.T
    uf = _UnionFind(m)
    for i in range(m):
        # vectorized threshold check for row i
        js = np.where(sims[i] >= float(tau))[0]
        for j in js.tolist():
            if j <= i:
                continue
            uf.union(i, j)
    comp: Dict[int, List[str]] = {}
    for i, cid in enumerate(cols):
        comp.setdefault(uf.find(i), []).append(cid)
    groups = [sorted(v) for v in comp.values() if len(v) > 1]
    groups.sort(key=len, reverse=True)
    return groups

