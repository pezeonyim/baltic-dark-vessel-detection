"""Canonical on-disk layout. Everything under data/ is regenerable and gitignored."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
REGIONS = CONFIGS / "regions"
DATA = ROOT / "data"
RAW = DATA / "raw"              # downloads exactly as the source provides them
PROCESSED = DATA / "processed"  # canonical-schema Parquet, one tree per region
REFERENCE = DATA / "reference"  # small curated files (ports, zones, sanctions lists)


def region_ais_dir(region_name: str) -> Path:
    """Hive-partitioned AIS for a region: <region>/ais/date=YYYY-MM-DD/part.parquet"""
    return PROCESSED / region_name / "ais"


def region_features_dir(region_name: str) -> Path:
    """Derived tables (outages, gaps, segments, coverage) for a region."""
    return PROCESSED / region_name / "features"
