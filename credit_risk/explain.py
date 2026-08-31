"""Simple explainability helpers for the XGBoost challenger.

This module exposes a single function `shap_importance` which returns a
simple table of per-feature importance. If TreeSHAP is available it uses
that; otherwise it falls back to the model's native feature_importances_.
The public signature is kept small and clear for the rest of the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .gbm import GBMModel


def shap_importance(gbm: GBMModel, X: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with mean absolute contribution per feature.

    Attempts to use XGBoost TreeSHAP (if available) and falls back to the
    model's feature_importances_ when TreeSHAP isn't accessible.
    """
    try:
        import xgboost as xgb
        prepared = gbm.prepare_features(X)
        dmatrix = xgb.DMatrix(prepared, enable_categorical=True)
        contribs = gbm.model.get_booster().predict(dmatrix, pred_contribs=True)[:, :-1]
        mean_abs = np.abs(contribs).mean(axis=0)
        vals = mean_abs
    except Exception:
        # Fallback: use feature_importances_ provided by the fitted sklearn-like wrapper
        vals = getattr(gbm.model, "feature_importances_", np.zeros(len(gbm.spec.model_features)))

    return (
        pd.DataFrame({"feature": gbm.spec.model_features, "mean_abs_shap": vals})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
