"""One entry point for the portfolio used by the app and trainer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_FILE, LENDING_CLUB_SPEC, FeatureSpec
from .lendingclub import load_lending_club


@dataclass(frozen=True)
class Dataset:
    name: str
    df: pd.DataFrame
    spec: FeatureSpec = LENDING_CLUB_SPEC
    source_file: Path | None = None


def resolve_dataset(sample_size: int | None = None) -> Dataset:
    """Load a reproducible sample from the expected Lending Club CSV."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    frame = load_lending_club(DATA_FILE, sample_size=sample_size)
    return Dataset(name="Lending Club", df=frame, source_file=DATA_FILE)
