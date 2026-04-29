from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..tree.hrs_b_loader import load_hrs_b_year
from ..tree.llm_client import load_env


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _default_out_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "output" / "embeddings"


def _schema_hash(ids: List[str], texts: List[str], *, salt: str = "") -> str:
    """
    Hash the embedding input schema (ids + description texts).
    This lets us auto-rebuild embeddings when the benchmark CSV changes.
    """
    h = hashlib.sha256()
    h.update((salt or "").encode("utf-8", "replace"))
    h.update(b"\n---\n")
    for cid, t in zip(ids, texts):
        h.update((cid or "").encode("utf-8", "replace"))
        h.update(b"\0")
        h.update((t or "").encode("utf-8", "replace"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass(frozen=True)
class Embeddings:
    year: int
    model: str
    ids: List[str]                 # column_id in table order
    mat: np.ndarray                # shape [n, d], float32, L2-normalized
    id_to_row: Dict[str, int]

    def get(self, column_id: str) -> np.ndarray:
        i = self.id_to_row.get(column_id)
        if i is None:
            raise KeyError(f"Unknown column_id: {column_id}")
        return self.mat[i]


def _invalidate_embedding_cache_file(npz_path: Path) -> None:
    """Remove a year's embedding cache (.npz + sidecar .meta.json) if present."""
    meta_path = npz_path.with_name(npz_path.stem + ".meta.json")
    for p in (npz_path, meta_path):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _load_year_npz(
    path: Path,
    *,
    build_if_missing: bool,
    year: int,
    embed_model: str,
    tree_model: str,
    out_dir: Path,
):
    """
    Load embeddings npz; if the archive is corrupt (e.g. interrupted write) and
    rebuild is allowed, delete the cache and rebuild once.

    npz members are read lazily, so we fully touch ids/mat here so CRC/truncation
    errors surface before returning the handle.
    """
    data = None
    try:
        data = np.load(path, allow_pickle=True)
        _ = [str(x) for x in data["ids"].tolist()]
        _ = np.asarray(data["mat"], dtype=np.float32).shape
        return data
    except Exception:
        if data is not None:
            try:
                data.close()
            except Exception:
                pass
        if not build_if_missing:
            raise
        _invalidate_embedding_cache_file(path)
        build_and_save_year(year, embed_model=embed_model, tree_model=tree_model, out_dir=out_dir)
        data2 = np.load(path, allow_pickle=True)
        _ = [str(x) for x in data2["ids"].tolist()]
        _ = np.asarray(data2["mat"], dtype=np.float32).shape
        return data2


def build_and_save_year(
    year: int,
    *,
    embed_model: str = "text-embedding-3-small",
    tree_model: str = "gpt-5.4",
    out_dir: Optional[Path] = None,
    batch_size: int = 128,
) -> Path:
    """
    Compute embeddings for all columns in HRS-B `{year}` and save to a compressed `.npz`.
    """
    load_env()
    from openai import OpenAI

    if out_dir is None:
        out_dir = _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_hrs_b_year(year)
    ids = [c.column_id for c in table.columns]
    # Paper-aligned default: embedding retrieval uses only per-column text.
    texts = [f"{c.column_id}\n{(c.description or '').strip()}".strip() for c in table.columns]
    schema_hash = _schema_hash(ids, texts, salt=f"mode=desc;embed_model={embed_model}")

    client = OpenAI()  # relies on OPENAI_API_KEY in env; load_env() sets it if in .env

    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=embed_model, input=chunk)
        # preserve order
        vecs.extend([d.embedding for d in resp.data])

    mat = np.asarray(vecs, dtype=np.float32)
    mat = _normalize_rows(mat)

    out_path = out_dir / f"hrs_b_{year}_{embed_model.replace('/','_')}.npz"
    np.savez_compressed(out_path, ids=np.array(ids, dtype=object), mat=mat, schema_hash=np.array(schema_hash))
    meta = {
        "year": int(year),
        "model": str(embed_model),
        "embed_mode": "desc",
        "n": int(len(ids)),
        "dim": int(mat.shape[1]),
        "schema_hash": schema_hash,
    }
    (out_dir / f"hrs_b_{year}_{embed_model.replace('/','_')}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return out_path


def load_year(
    year: int,
    *,
    embed_model: str = "text-embedding-3-small",
    tree_model: str = "gpt-5.4",
    out_dir: Optional[Path] = None,
    build_if_missing: bool = False,
) -> Embeddings:
    if out_dir is None:
        out_dir = _default_out_dir()
    path = out_dir / f"hrs_b_{year}_{embed_model.replace('/','_')}.npz"
    if (not path.exists()) and build_if_missing:
        build_and_save_year(year, embed_model=embed_model, tree_model=tree_model, out_dir=out_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {path}")

    # If we can, verify the cache matches the current benchmark CSV schema.
    want_hash: Optional[str] = None
    try:
        table = load_hrs_b_year(year)
        cur_ids = [c.column_id for c in table.columns]
        cur_texts = [f"{c.column_id}\n{(c.description or '').strip()}".strip() for c in table.columns]
        want_hash = _schema_hash(cur_ids, cur_texts, salt=f"mode=desc;embed_model={embed_model}")
    except Exception:
        want_hash = None

    data = _load_year_npz(
        path,
        build_if_missing=build_if_missing,
        year=year,
        embed_model=embed_model,
        tree_model=tree_model,
        out_dir=out_dir,
    )
    ids = [str(x) for x in data["ids"].tolist()]
    mat = np.asarray(data["mat"], dtype=np.float32)
    have_hash: Optional[str] = None
    try:
        if "schema_hash" in data:
            have_hash = str(data["schema_hash"].tolist())
    except Exception:
        have_hash = None

    if want_hash is not None:
        # Old caches won't have a hash; treat as stale.
        if (have_hash is None) or (have_hash != want_hash):
            if build_if_missing:
                build_and_save_year(year, embed_model=embed_model, tree_model=tree_model, out_dir=out_dir)
                data = _load_year_npz(
                    path,
                    build_if_missing=build_if_missing,
                    year=year,
                    embed_model=embed_model,
                    tree_model=tree_model,
                    out_dir=out_dir,
                )
                ids = [str(x) for x in data["ids"].tolist()]
                mat = np.asarray(data["mat"], dtype=np.float32)
            else:
                raise ValueError(
                    "Embeddings cache does not match current dataset schema. "
                    f"Rebuild embeddings for year={year}, model={embed_model}."
                )

    id_to_row = {cid: i for i, cid in enumerate(ids)}
    return Embeddings(year=int(year), model=str(embed_model), ids=ids, mat=mat, id_to_row=id_to_row)


def top_k_by_cosine(
    tgt: Embeddings,
    query_vec: np.ndarray,
    *,
    k: int = 20,
) -> List[Tuple[str, float]]:
    """
    Return top-k (column_id, cosine) in descending order.
    Assumes both query_vec and tgt.mat are L2-normalized.
    """
    sims = tgt.mat @ query_vec.astype(np.float32)
    k = min(int(k), sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(tgt.ids[i], float(sims[i])) for i in idx]

