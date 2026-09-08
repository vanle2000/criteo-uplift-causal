"""Paths, column names and constants shared by every step.

Anything that more than one script needs to agree on lives here, so that a
change to (say) the feature list cannot silently desynchronise the ingest
from the estimators.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

# Source of record. The URL published in most papers and blog posts
# (go.criteo.net/criteo-research-uplift-v2.1.csv.gz) is dead as of 2026-09;
# Criteo's own HuggingFace mirror serves the identical v2.1 file.
SOURCE_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/"
    "resolve/main/criteo-research-uplift-v2.1.csv.gz"
)
RAW_GZ = DATA_RAW / "criteo-research-uplift-v2.1.csv.gz"
PARQUET = DATA_PROCESSED / "criteo_uplift.parquet"
INGEST_MANIFEST = DATA_PROCESSED / "ingest_manifest.json"
CLUSTER_CACHE = DATA_PROCESSED / "cluster_id.npy"

# 12 anonymised, already-standardised behavioural features.
FEATURES = [f"f{i}" for i in range(12)]

TREATMENT = "treatment"          # randomised: 1 = eligible for targeting
OUTCOMES = ["visit", "conversion"]
PRIMARY_OUTCOME = "conversion"
EXPOSURE = "exposure"            # POST-TREATMENT: was an ad actually served?
ROW_ID = "row_id"

# The file is one row per IMPRESSION, not per user, and ships no user id.
# f2 is the only feature that varies within an otherwise-identical covariate
# vector, so the remaining 11 act as a proxy cluster key. It is a proxy, not a
# true id: ~3.4% of clusters contain both arms, which a real user id would not.
VARYING_FEATURE = "f2"
CLUSTER_KEY = [f for f in FEATURES if f != VARYING_FEATURE]
CLUSTER_ID = "cluster_id"

# Every column the raw CSV is expected to contain, in order.
RAW_COLUMNS = FEATURES + [TREATMENT, "conversion", "visit", EXPOSURE]
BINARY_COLUMNS = [TREATMENT, "conversion", "visit", EXPOSURE]

SEED = 20260908
