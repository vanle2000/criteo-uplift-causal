"""Ingest: gzipped source CSV -> validated, typed Parquet.

The raw file is 311 MB gzipped / ~2.5 GB as text. We stream it in chunks so
peak memory stays flat regardless of file size, narrow the dtypes on the way
through (float64 -> float32 for features, int64 -> int8 for the four binary
columns), and write one Parquet row group per chunk.

Validation is done *during* the pass, not after, so a malformed row fails the
ingest rather than silently poisoning every downstream estimate.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C

CHUNK_ROWS = 2_000_000


def _arrow_schema() -> pa.Schema:
    """Explicit Parquet schema.

    Declared rather than inferred: an explicit schema is what makes the
    ParquetWriter reject a chunk whose dtypes drifted, instead of writing a
    file whose row groups disagree with each other.
    """
    fields = [(C.ROW_ID, pa.int32())]
    fields += [(f, pa.float32()) for f in C.FEATURES]
    fields += [(b, pa.int8()) for b in C.BINARY_COLUMNS]
    return pa.schema(fields)


def _read_dtypes() -> dict[str, str]:
    d = {f: "float32" for f in C.FEATURES}
    d.update({b: "int8" for b in C.BINARY_COLUMNS})
    return d


@dataclass
class IngestStats:
    """Running totals accumulated across chunks, written out as the manifest."""

    n_rows: int = 0
    n_chunks: int = 0
    binary_sums: dict[str, int] = field(default_factory=dict)
    feature_min: dict[str, float] = field(default_factory=dict)
    feature_max: dict[str, float] = field(default_factory=dict)
    feature_sum: dict[str, float] = field(default_factory=dict)

    def update(self, df: pd.DataFrame) -> None:
        self.n_rows += len(df)
        self.n_chunks += 1
        for b in C.BINARY_COLUMNS:
            self.binary_sums[b] = self.binary_sums.get(b, 0) + int(df[b].sum())
        for f in C.FEATURES:
            col = df[f].to_numpy()
            lo, hi = float(col.min()), float(col.max())
            self.feature_min[f] = min(self.feature_min.get(f, lo), lo)
            self.feature_max[f] = max(self.feature_max.get(f, hi), hi)
            # float64 accumulator: summing 14M float32s in float32 loses
            # precision badly (~1e-3 relative), which would corrupt the mean.
            self.feature_sum[f] = self.feature_sum.get(f, 0.0) + float(
                col.astype(np.float64).sum()
            )

    def feature_means(self) -> dict[str, float]:
        return {f: self.feature_sum[f] / self.n_rows for f in C.FEATURES}


def _validate_chunk(df: pd.DataFrame, chunk_idx: int) -> None:
    """Fail loudly on anything that would invalidate a causal estimate."""
    if list(df.columns[: len(C.RAW_COLUMNS)]) != C.RAW_COLUMNS:
        raise ValueError(
            f"chunk {chunk_idx}: unexpected columns {list(df.columns)}; "
            f"expected {C.RAW_COLUMNS}"
        )
    n_null = int(df.isna().sum().sum())
    if n_null:
        raise ValueError(f"chunk {chunk_idx}: {n_null} null values; source is documented as complete")
    for b in C.BINARY_COLUMNS:
        uniq = pd.unique(df[b])
        if not set(uniq.tolist()) <= {0, 1}:
            raise ValueError(f"chunk {chunk_idx}: column {b!r} is not binary; saw {uniq[:10]}")
    bad = [f for f in C.FEATURES if not np.isfinite(df[f].to_numpy()).all()]
    if bad:
        raise ValueError(f"chunk {chunk_idx}: non-finite values in {bad}")


def sha256(path: Path, buf_size: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(buf_size):
            h.update(block)
    return h.hexdigest()


def ingest(
    src: Path = C.RAW_GZ,
    dst: Path = C.PARQUET,
    manifest_path: Path = C.INGEST_MANIFEST,
    chunk_rows: int = CHUNK_ROWS,
    force: bool = False,
) -> dict:
    """Stream `src` into `dst` as Parquet. Returns the manifest dict.

    Idempotent: re-running with an existing output and a matching source
    checksum is a no-op, so the whole pipeline can be re-run cheaply.
    """
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Download it first:\n  curl -L -o {src} {C.SOURCE_URL}"
        )

    src_digest = sha256(src)
    if dst.exists() and manifest_path.exists() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("source_sha256") == src_digest:
            print(f"[ingest] up to date ({prior['n_rows']:,} rows) -> {dst}")
            return prior

    dst.parent.mkdir(parents=True, exist_ok=True)
    schema = _arrow_schema()
    stats = IngestStats()
    t0 = time.perf_counter()

    # ZSTD over Snappy: these are high-entropy standardised floats that barely
    # compress, so we take the slower codec for the ~25% the file size buys.
    writer = pq.ParquetWriter(dst, schema, compression="zstd", compression_level=3)
    try:
        reader = pd.read_csv(src, chunksize=chunk_rows, dtype=_read_dtypes(), compression="gzip")
        for i, chunk in enumerate(reader):
            _validate_chunk(chunk, i)
            # Stable global row id, assigned before any shuffling or sampling
            # so that Step 2's confounded subsample and Step 3's matched pairs
            # can always be traced back to specific source rows.
            chunk.insert(0, C.ROW_ID, np.arange(stats.n_rows, stats.n_rows + len(chunk), dtype=np.int32))
            stats.update(chunk)
            writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
            print(f"[ingest] chunk {i}: {stats.n_rows:,} rows", flush=True)
    finally:
        writer.close()

    elapsed = time.perf_counter() - t0
    manifest = {
        "source_url": C.SOURCE_URL,
        "source_file": src.name,
        "source_bytes": src.stat().st_size,
        "source_sha256": src_digest,
        "output": str(dst.relative_to(C.PROJECT_ROOT)),
        "output_bytes": dst.stat().st_size,
        "n_rows": stats.n_rows,
        "n_chunks": stats.n_chunks,
        "columns": [C.ROW_ID] + C.FEATURES + C.BINARY_COLUMNS,
        "binary_rates": {k: v / stats.n_rows for k, v in stats.binary_sums.items()},
        "binary_counts": stats.binary_sums,
        "feature_means": stats.feature_means(),
        "feature_min": stats.feature_min,
        "feature_max": stats.feature_max,
        "ingest_seconds": round(elapsed, 1),
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[ingest] {stats.n_rows:,} rows in {elapsed:.1f}s -> {dst}")
    return manifest


def load(columns: list[str] | None = None, path: Path = C.PARQUET) -> pd.DataFrame:
    """Load the analysis frame. ~700 MB for all columns; fits comfortably."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/00_ingest.py")
    return pq.read_table(path, columns=columns).to_pandas()


def cluster_ids(df: pd.DataFrame, cache: Path | None = C.CLUSTER_CACHE) -> np.ndarray:
    """Proxy user id: a dense integer per distinct value of the stable features.

    Cached to disk because the groupby over 11 float columns costs ~40s and
    every downstream step needs the same ids to stay comparable.
    """
    if cache is not None and cache.exists():
        ids = np.load(cache)
        if len(ids) == len(df):
            return ids
    ids = df.groupby(C.CLUSTER_KEY, sort=False, observed=True).ngroup().to_numpy(np.int64)
    if cache is not None:
        np.save(cache, ids)
    return ids
