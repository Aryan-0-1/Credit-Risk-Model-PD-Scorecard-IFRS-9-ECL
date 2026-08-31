"""
Model-validation toolkit — the diagnostics a model-risk / validation team expects
on top of headline discrimination metrics.

* **Gains / KS table** — decile view of rank-ordering (bad rate, lift, cumulative KS).
* **Rating masterscale** — score band → PD → *observed* default rate.
* **Calibration** — reliability table + Hosmer–Lemeshow goodness-of-fit test
  (calibrated PDs matter: IFRS 9 ECL multiplies them directly).
* **Bootstrap CIs** — sampling uncertainty around AUROC / Gini / KS.
* **Out-of-time validation** — train on early vintages, test on the latest, plus
  score-distribution PSI across vintages (early-warning for model decay).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config, metrics
from .scorecard import Scorecard
from .woe import WoEEncoder


# ---------------------------------------------------------------------------
# Rank-ordering
# ---------------------------------------------------------------------------
def gains_table(y_true, pd_hat, n_bands: int = 10) -> pd.DataFrame:
    """Decile gains/KS table, riskiest band first."""
    df = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_hat)})
    df = df.sort_values("pd", ascending=False).reset_index(drop=True)
    df["band"] = pd.qcut(df.index, n_bands, labels=range(1, n_bands + 1))

    total_bad = df["y"].sum()
    total_good = len(df) - total_bad
    g = df.groupby("band", observed=True)["y"].agg(n="count", bad="sum")
    g["good"] = g["n"] - g["bad"]
    g["bad_rate"] = g["bad"] / g["n"]
    g["cum_bad_rate"] = g["bad"].cumsum() / total_bad
    g["cum_good_rate"] = g["good"].cumsum() / total_good
    g["ks"] = (g["cum_bad_rate"] - g["cum_good_rate"]).abs()
    g["lift"] = g["bad_rate"] / (total_bad / len(df))
    return g.reset_index()


def rating_masterscale(rating, pd_hat, y_true) -> pd.DataFrame:
    """Per rating grade: exposure share, mean predicted PD, observed default rate."""
    df = pd.DataFrame({"grade": np.asarray(rating), "pd": np.asarray(pd_hat), "y": np.asarray(y_true)})
    g = df.groupby("grade").agg(n=("y", "size"), predicted_pd=("pd", "mean"), observed_dr=("y", "mean"))
    g["population_pct"] = g["n"] / g["n"].sum()
    order = [b[0] for b in config.DEFAULT_ECL.rating_bands]
    return g.reindex([o for o in order if o in g.index]).reset_index()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_table(y_true, pd_hat, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted PD vs observed default rate per predicted-PD decile."""
    df = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_hat)})
    df["bucket"] = pd.qcut(df["pd"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
    g = df.groupby("bucket", observed=True).agg(
        n=("y", "size"), predicted=("pd", "mean"), observed=("y", "mean")
    )
    return g.reset_index()


def hosmer_lemeshow(y_true, pd_hat, n_bins: int = 10) -> dict:
    """Hosmer–Lemeshow goodness-of-fit test. Large p-value ⇒ well-calibrated."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "pd": np.asarray(pd_hat, dtype=float)})
    df["bucket"] = pd.qcut(df["pd"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bucket").agg(obs=("y", "sum"), exp=("pd", "sum"), n=("y", "size"))
    # HL statistic sums (O-E)^2 / (E(1-E/n)) across deciles.
    e_rate = g["exp"] / g["n"]
    denom = g["exp"] * (1 - e_rate)
    stat = float((((g["obs"] - g["exp"]) ** 2) / denom.replace(0, np.nan)).sum())
    dof = max(n_bins - 2, 1)
    p_value = float(stats.chi2.sf(stat, dof))
    return {"statistic": round(stat, 3), "dof": dof, "p_value": round(p_value, 4)}


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def bootstrap_metric_ci(y_true, score, metric="gini", n_boot: int = 500,
                        alpha: float = 0.05, random_state: int = config.RANDOM_STATE) -> dict:
    """Bootstrap percentile CI for a discrimination metric."""
    fn = {"auroc": metrics.auroc, "gini": metrics.gini, "ks": metrics.ks_statistic}[metric]
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    stats_ = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() in (0, len(idx)):  # need both classes
            continue
        stats_.append(fn(y_true[idx], score[idx]))
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"metric": metric, "point": round(fn(y_true, score), 4),
            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}


# ---------------------------------------------------------------------------
# Stability & out-of-time
# ---------------------------------------------------------------------------
def psi_by_vintage(scores, vintages, base_vintage=None) -> pd.DataFrame:
    """Score-distribution PSI of each vintage vs a base vintage."""
    df = pd.DataFrame({"score": np.asarray(scores), "vintage": np.asarray(vintages)}).dropna()
    years = sorted(df["vintage"].unique())
    base_vintage = base_vintage if base_vintage is not None else years[0]
    base = df.loc[df["vintage"] == base_vintage, "score"].to_numpy()
    rows = []
    for y in years:
        cur = df.loc[df["vintage"] == y, "score"].to_numpy()
        rows.append({"vintage": int(y), "n": len(cur),
                     "psi_vs_base": round(metrics.psi(base, cur), 4)})
    return pd.DataFrame(rows)


def out_of_time_validation(df: pd.DataFrame, spec: config.FeatureSpec,
                           cutoff_year=None) -> dict | None:
    """Refit on early vintages, evaluate on the latest — a true OOT test.

    Returns ``None`` when the data lacks usable vintages.
    """
    if "vintage" not in df.columns or df["vintage"].dropna().nunique() < 2:
        return None

    d = df.dropna(subset=["vintage"]).copy()
    years = sorted(d["vintage"].unique())
    cutoff_year = cutoff_year if cutoff_year is not None else years[-1]

    train = d[d["vintage"] < cutoff_year]
    test = d[d["vintage"] == cutoff_year]
    if len(train) < 500 or len(test) < 200:
        return None

    encoder = WoEEncoder(spec.numeric, spec.categorical)
    Xtr = encoder.fit_transform(train[spec.model_features], train[spec.target])
    card = Scorecard(encoder).fit(Xtr, train[spec.target])

    pd_test = card.predict_pd(encoder.transform(test[spec.model_features]))
    pd_train = card.predict_pd(Xtr)
    m = metrics.evaluate(test[spec.target], pd_test)
    return {
        "train_years": [int(y) for y in years if y < cutoff_year],
        "test_year": int(cutoff_year),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "auroc": m.auroc, "gini": round(m.gini, 4), "ks": round(m.ks, 4),
        "score_psi": round(metrics.psi(card.score(Xtr), card.score(encoder.transform(test[spec.model_features]))), 4),
    }
