"""Shared paths, feature definitions, and model assumptions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "accepted_2007_to_2018Q4.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_JSON = REPORTS_DIR / "metrics.json"
SCORECARD_CSV = REPORTS_DIR / "scorecard.csv"
ECL_CSV = REPORTS_DIR / "ecl_results.csv"
# Precomputed dashboard artifacts (avoid recomputing on every app start).
# The two large ones are Parquet, not CSV — at ~1.3M rows, CSV parsing alone
# was a meaningful chunk of the dashboard's startup time.
SCORED_PORTFOLIO_PARQUET = REPORTS_DIR / "scored_portfolio.parquet"
FULL_PORTFOLIO_PARQUET = REPORTS_DIR / "full_portfolio.parquet"
ROC_CURVES_CSV = REPORTS_DIR / "roc_curves.csv"
KS_CURVE_CSV = REPORTS_DIR / "ks_curve.csv"
DASHBOARD_FIGURES_DIR = REPORTS_DIR / "dashboard_figures"
GAINS_TABLE_CSV = REPORTS_DIR / "gains_table.csv"
CALIBRATION_CSV = REPORTS_DIR / "calibration_table.csv"
MASTERSCALE_CSV = REPORTS_DIR / "masterscale.csv"
PSI_VINTAGE_CSV = REPORTS_DIR / "psi_vintage.csv"
IV_SUMMARY_CSV = REPORTS_DIR / "iv_summary.csv"
RANDOM_STATE = 42
TARGET = "default"
TRAIN_SAMPLE_SIZE = 150_000


@dataclass(frozen=True)
class FeatureSpec:
    """The input columns expected by both models."""

    numeric: list[str]
    categorical: list[str]
    target: str = TARGET

    @property
    def model_features(self) -> list[str]:
        return self.numeric + self.categorical

    def to_dict(self) -> dict:
        return {"numeric": self.numeric, "categorical": self.categorical, "target": self.target}

    @classmethod
    def from_dict(cls, values: dict) -> "FeatureSpec":
        return cls(list(values["numeric"]), list(values["categorical"]), values.get("target", TARGET))


LENDING_CLUB_SPEC = FeatureSpec(
    numeric=[
        "annual_income", "employment_length", "loan_amount", "loan_term",
        "interest_rate", "dti", "revolving_utilization", "num_delinquencies_2yr",
        "credit_history_length", "num_open_accounts", "fico", "installment",
        "mort_acc", "pub_rec", "pub_rec_bankruptcies", "total_acc",
    ],
    categorical=["home_ownership", "purpose", "sub_grade", "verification_status", "application_type"],
)


@dataclass(frozen=True)
class ScorecardScaling:
    base_score: int = 600
    base_odds: float = 50.0
    pdo: int = 20


@dataclass(frozen=True)
class ECLParams:
    """Assumptions for the three-stage ECL model."""

    lgd_base: float = 0.45
    sicr_pd_multiple: float = 2.0
    sicr_pd_floor: float = 0.05
    sicr_pd_absolute: float = 0.20
    low_credit_risk_pd: float = 0.02
    dpd_stage2: int = 30
    dpd_stage3: int = 90
    macro_pd_multiplier: float = 1.0
    rating_bands: tuple = field(default=(("AAA", 720), ("AA", 680), ("A", 640), ("BBB", 600), ("BB", 560), ("B", 520), ("C", 0)))


DEFAULT_SCALING = ScorecardScaling()
DEFAULT_ECL = ECLParams()
