"""Step 0 - download check + CSV -> Parquet ingest.

Usage:
    python scripts/00_ingest.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import data  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-ingest even if output is current")
    ap.add_argument("--chunk-rows", type=int, default=data.CHUNK_ROWS)
    args = ap.parse_args()

    m = data.ingest(force=args.force, chunk_rows=args.chunk_rows)
    print("\n--- ingest manifest ---")
    print(f"rows              {m['n_rows']:,}")
    print(f"parquet size      {m['output_bytes'] / 1e6:.0f} MB (from {m['source_bytes'] / 1e6:.0f} MB .gz)")
    for k, v in m["binary_rates"].items():
        print(f"{k:<18}{v:.6f}  ({m['binary_counts'][k]:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
