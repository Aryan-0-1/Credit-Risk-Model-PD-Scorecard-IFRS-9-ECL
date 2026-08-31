"""Random Forest challenger model for borrower default prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import FeatureSpec, LENDING_CLUB_SPEC, RANDOM_STATE


class RandomForestModel:
    """Thin wrapper around :class:`sklearn.ensemble.RandomForestClassifier`.

    This model is useful as a non-linear alternative to the logistic scorecard and
    can be used as a second benchmark alongside the XGBoost challenger.
    """

    def __init__(self, spec: FeatureSpec = LENDING_CLUB_SPEC, **params):
        self.spec = spec
        defaults = dict(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=10,
            min_samples_split=20,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        defaults.update(params)
        self.model = RandomForestClassifier(**defaults)

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select the configured columns and one-hot encode categorical variables."""
        X = X[self.spec.model_features].copy()
        for col in self.spec.categorical:
            X[col] = X[col].astype(str)
        return pd.get_dummies(X, columns=self.spec.categorical, drop_first=False)

    def fit(self, X: pd.DataFrame, y) -> "RandomForestModel":
        self.model.fit(self.prepare_features(X), y)
        return self

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        prepared = self.prepare_features(X)
        # Align columns to the training schema in case some dummy columns are missing
        if hasattr(self.model, "feature_names_in_"):
            missing = set(self.model.feature_names_in_) - set(prepared.columns)
            for col in sorted(missing):
                prepared[col] = 0
            prepared = prepared.reindex(columns=self.model.feature_names_in_, fill_value=0)
        return self.model.predict_proba(prepared)[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        feature_names = self.model.feature_names_in_ if hasattr(self.model, "feature_names_in_") else self.spec.model_features
        importance = pd.DataFrame(
            {"feature": feature_names, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        return importance
