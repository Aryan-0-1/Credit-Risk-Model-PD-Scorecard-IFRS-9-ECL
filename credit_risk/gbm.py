"""
XGBoost challenger model.

Gradient boosting typically edges out a linear scorecard on discrimination by
capturing non-linearities and interactions. We keep it as a *challenger*: the
scorecard stays the interpretable champion, while XGBoost quantifies the
performance left on the table. Native categorical support avoids one-hot
blow-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import RANDOM_STATE, LENDING_CLUB_SPEC, FeatureSpec


class GBMModel:
    """Thin wrapper around :class:`xgboost.XGBClassifier`."""

    def __init__(self, spec: FeatureSpec = LENDING_CLUB_SPEC, **params):
        self.spec = spec
        defaults = dict(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=5,
            eval_metric="auc",
            enable_categorical=True,
            tree_method="hist",
            random_state=RANDOM_STATE,
        )
        defaults.update(params)
        self.model = XGBClassifier(**defaults)

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select model columns and cast categoricals to pandas ``category`` dtype."""
        X = X[self.spec.model_features].copy()
        for col in self.spec.categorical:
            X[col] = X[col].astype("category")
        return X

    def fit(self, X: pd.DataFrame, y) -> "GBMModel":
        self.model.fit(self.prepare_features(X), y)
        return self

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.prepare_features(X))[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {"feature": self.spec.model_features, "importance": self.model.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
