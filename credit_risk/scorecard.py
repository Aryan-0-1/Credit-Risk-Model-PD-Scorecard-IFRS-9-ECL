"""
Logistic-Regression PD scorecard with points scaling.

The model is a logistic regression on WoE-transformed features — transparent,
monotonic in WoE and easy to sign-check. Coefficients are then converted to an
additive points system using the standard *points-to-double-odds* (PDO)
convention, so a business user can read a borrower's score as a sum of
per-attribute points.

Scaling maths (odds expressed as good:bad):
    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset - factor * (intercept + Σ βᵢ · woeᵢ)
Per-attribute points distribute the intercept evenly across the *k* features:
    pointsᵢ(bin) = offset/k - factor * (intercept/k + βᵢ · woe(bin))
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import DEFAULT_SCALING, ScorecardScaling
from .woe import WoEEncoder


class Scorecard:
    """A WoE logistic-regression PD model plus a points scorecard."""

    def __init__(self, encoder: WoEEncoder, scaling: ScorecardScaling = DEFAULT_SCALING):
        self.encoder = encoder
        self.scaling = scaling
        self.model = LogisticRegression(max_iter=1000, C=1.0)
        self.feature_names_: list[str] = []

    # --- fitting / prediction ------------------------------------------
    def fit(self, X_woe: pd.DataFrame, y) -> "Scorecard":
        self.feature_names_ = list(X_woe.columns)
        self.model.fit(X_woe, y)
        return self

    def predict_pd(self, X_woe: pd.DataFrame) -> np.ndarray:
        """Probability of default (the positive/event class)."""
        return self.model.predict_proba(X_woe[self.feature_names_])[:, 1]

    # --- points scaling ------------------------------------------------
    @property
    def _factor(self) -> float:
        return self.scaling.pdo / math.log(2)

    @property
    def _offset(self) -> float:
        return self.scaling.base_score - self._factor * math.log(self.scaling.base_odds)

    def score(self, X_woe: pd.DataFrame) -> np.ndarray:
        """Total scorecard points (higher = lower risk)."""
        beta = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        linear = X_woe[self.feature_names_].to_numpy() @ beta + intercept
        return self._offset - self._factor * linear

    def rating(self, score: np.ndarray, bands=None) -> np.ndarray:
        """Map scores to letter grades using descending score bands."""
        from .config import DEFAULT_ECL

        bands = bands or DEFAULT_ECL.rating_bands
        score = np.asarray(score)
        out = np.empty(score.shape, dtype=object)
        assigned = np.zeros(score.shape, dtype=bool)
        for label, lower in bands:  # bands are ordered high -> low
            mask = (~assigned) & (score >= lower)
            out[mask] = label
            assigned |= mask
        out[~assigned] = bands[-1][0]
        return out

    def scorecard_table(self) -> pd.DataFrame:
        """Human-readable scorecard: per feature, per bin, WoE → points."""
        beta = dict(zip(self.feature_names_, self.model.coef_[0]))
        intercept = self.model.intercept_[0]
        k = len(self.feature_names_)
        factor, offset = self._factor, self._offset

        rows = []
        for woe_name in self.feature_names_:
            feat = woe_name[:-4] if woe_name.endswith("_woe") else woe_name
            binning = self.encoder.bins_[feat]
            b = beta[woe_name]
            for _, r in binning.table.iterrows():
                woe = r["woe"]
                points = offset / k - factor * (intercept / k + b * woe)
                rows.append(
                    {
                        "feature": feat,
                        "bin": r["bin"],
                        "count": int(r["n"]),
                        "event_rate": round(float(r["event_rate"]), 4),
                        "woe": round(float(woe), 4),
                        "coef": round(float(b), 4),
                        "points": int(round(points)),
                    }
                )
        return pd.DataFrame(rows)
