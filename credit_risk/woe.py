"""
Weight-of-Evidence (WoE) binning and Information Value (IV).

WoE recodes each predictor into the log-odds of *good* vs *bad* within a bin,
which linearises the relationship with the log-odds of default — exactly what a
logistic scorecard wants. IV summarises each feature's univariate predictive
power.

Conventions
-----------
* ``event`` = default (target == 1, "bad"); ``non-event`` = "good".
* ``WoE = ln( P(good | bin) / P(bad | bin) )`` — higher WoE ⇒ lower risk.
* A 0.5 count adjustment prevents division-by-zero in sparse bins.
* IV interpretation (Siddiqi): <0.02 unpredictive · 0.02–0.1 weak ·
  0.1–0.3 medium · 0.3–0.5 strong · >0.5 suspiciously strong (check leakage).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_SMOOTHING = 0.5


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious"


@dataclass
class FeatureBinning:
    """WoE mapping for a single feature."""

    feature: str
    kind: str                     # "numeric" | "categorical"
    edges: np.ndarray | None      # bin edges for numeric features
    labels: list                  # bin labels (intervals or category values)
    woe: dict                     # label -> WoE
    iv: float
    table: pd.DataFrame           # per-bin diagnostics

    def apply(self, series: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            binned = pd.cut(series, bins=self.edges, include_lowest=True)
            keys = binned.astype(str)
        else:
            keys = series.astype(str)
        default_woe = 0.0
        return keys.map(self.woe).fillna(default_woe).astype(float)


def _woe_table(labels: pd.Series, target: pd.Series) -> tuple[pd.DataFrame, dict, float]:
    """Compute the WoE table, WoE map and IV for an already-binned feature.

    Grouping is done on the *original* labels so that an ordered categorical from
    ``pd.cut`` keeps its interval order (stringifying first would sort bins
    lexicographically, e.g. "(8.7, 12.6]" after "(12.6, 18.1]"). Labels are cast
    to ``str`` only afterwards, to match how :meth:`FeatureBinning.apply` keys them.
    """
    grouped = pd.DataFrame({"bin": labels, "target": target.to_numpy()})
    agg = grouped.groupby("bin", observed=True)["target"].agg(["count", "sum"])
    agg.index = agg.index.astype(str)
    agg = agg.rename(columns={"count": "n", "sum": "bad"})
    agg["good"] = agg["n"] - agg["bad"]

    total_good = agg["good"].sum()
    total_bad = agg["bad"].sum()
    dist_good = (agg["good"] + _SMOOTHING) / (total_good + _SMOOTHING * len(agg))
    dist_bad = (agg["bad"] + _SMOOTHING) / (total_bad + _SMOOTHING * len(agg))

    agg["event_rate"] = agg["bad"] / agg["n"]
    agg["woe"] = np.log(dist_good / dist_bad)
    agg["iv"] = (dist_good - dist_bad) * agg["woe"]

    iv = float(agg["iv"].sum())
    woe_map = agg["woe"].to_dict()
    return agg.reset_index(), woe_map, iv


def _bin_event_rates(series: pd.Series, y: pd.Series, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (event_rate, count) per bin defined by *edges*, in bin order."""
    binned = pd.cut(series, bins=edges, include_lowest=True)
    grouped = pd.DataFrame({"bin": binned, "y": y.to_numpy()}).groupby("bin", observed=False)["y"]
    counts = grouped.count().to_numpy(dtype=float)
    rates = grouped.mean().to_numpy(dtype=float)
    return rates, counts


def monotonic_edges(series: pd.Series, y: pd.Series, edges: np.ndarray, min_frac: float = 0.05) -> np.ndarray:
    """Merge adjacent quantile bins until the event rate is monotonic and every
    bin holds at least *min_frac* of the population.

    A monotonic WoE trend is a standard scorecard requirement: it keeps the
    relationship interpretable and the per-attribute points economically sensible.
    """
    edges = list(edges)
    s = series.dropna()
    y = y.loc[s.index]
    rates, counts = _bin_event_rates(s, y, np.array(edges))
    # Direction of the trend (rising or falling event rate across bins).
    idx = np.arange(len(rates))
    direction = 1.0 if np.corrcoef(idx, np.nan_to_num(rates))[0, 1] >= 0 else -1.0

    while len(edges) > 2:
        rates, counts = _bin_event_rates(s, y, np.array(edges))
        total = counts.sum()

        # 1) merge under-populated bins.
        small = np.where(counts / total < min_frac)[0]
        if len(small):
            i = int(small[0])
            drop = i + 1 if i == 0 else i  # remove the edge that fuses it with a neighbour
            edges.pop(drop)
            continue

        # 2) merge the first pair that breaks monotonicity.
        diffs = np.diff(rates) * direction
        violations = np.where(diffs < 0)[0]
        if len(violations):
            edges.pop(int(violations[0]) + 1)
            continue
        break

    return np.array(edges)


class WoEEncoder:
    """Fit WoE bins on training data and transform any frame to WoE features."""

    def __init__(self, numeric_features, categorical_features, n_bins: int = 8,
                 monotonic: bool = True, min_bin_frac: float = 0.05):
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.n_bins = n_bins
        self.monotonic = monotonic
        self.min_bin_frac = min_bin_frac
        self.bins_: dict[str, FeatureBinning] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        y = pd.Series(np.asarray(y), index=X.index)
        for feat in self.numeric_features:
            self.bins_[feat] = self._fit_numeric(X[feat], y)
        for feat in self.categorical_features:
            self.bins_[feat] = self._fit_categorical(X[feat], y)
        return self

    def _fit_numeric(self, series: pd.Series, y: pd.Series) -> FeatureBinning:
        # Quantile edges, de-duplicated so low-cardinality columns don't explode.
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        edges = np.unique(np.quantile(series.dropna(), quantiles))
        edges[0], edges[-1] = -np.inf, np.inf
        if self.monotonic and len(edges) > 3:
            edges = monotonic_edges(series, y, edges, self.min_bin_frac)
        binned = pd.cut(series, bins=edges, include_lowest=True)
        table, woe_map, iv = _woe_table(binned, y)
        return FeatureBinning(
            feature=series.name, kind="numeric", edges=edges,
            labels=list(woe_map), woe=woe_map, iv=iv, table=table,
        )

    def _fit_categorical(self, series: pd.Series, y: pd.Series) -> FeatureBinning:
        table, woe_map, iv = _woe_table(series.astype(str), y)
        return FeatureBinning(
            feature=series.name, kind="categorical", edges=None,
            labels=list(woe_map), woe=woe_map, iv=iv, table=table,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {f"{feat}_woe": binning.apply(X[feat]) for feat, binning in self.bins_.items()}
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def iv_summary(self) -> pd.DataFrame:
        rows = [
            {"feature": f, "iv": b.iv, "strength": iv_strength(b.iv)}
            for f, b in self.bins_.items()
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    @property
    def woe_feature_names(self) -> list[str]:
        return [f"{feat}_woe" for feat in self.bins_]
