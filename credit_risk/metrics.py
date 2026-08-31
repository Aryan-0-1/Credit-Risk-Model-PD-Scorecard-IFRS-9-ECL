"""
Discrimination and stability metrics for credit models.

* **AUROC** – area under the ROC curve.
* **Gini** – ``2 * AUROC - 1``; the industry-standard rank-ordering measure.
* **KS**   – Kolmogorov–Smirnov: the maximum gap between the cumulative good and
  bad score distributions (higher ⇒ better separation).
* **PSI**  – Population Stability Index: distribution drift between two samples
  (e.g. train vs out-of-time), used to monitor model decay.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class ScoreMetrics:
    auroc: float
    gini: float
    ks: float

    def as_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def auroc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def gini(y_true, y_score) -> float:
    return 2.0 * auroc(y_true, y_score) - 1.0


def ks_statistic(y_true, y_score) -> float:
    """Maximum separation between cumulative non-event and event distributions."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_sorted = y_true[order]

    n_bad = y_sorted.sum()
    n_good = len(y_sorted) - n_bad
    if n_bad == 0 or n_good == 0:
        return 0.0

    cum_bad = np.cumsum(y_sorted) / n_bad
    cum_good = np.cumsum(1 - y_sorted) / n_good
    return float(np.max(np.abs(cum_good - cum_bad)))


def evaluate(y_true, y_score) -> ScoreMetrics:
    """Bundle the three discrimination metrics."""
    a = auroc(y_true, y_score)
    return ScoreMetrics(auroc=a, gini=2 * a - 1, ks=ks_statistic(y_true, y_score))


def roc_points(y_true, y_score, max_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """ROC curve (fpr, tpr), downsampled to at most *max_points* points.

    ``roc_curve`` returns a point per unique score, which on a large book can
    be hundreds of thousands of points — far more than a line chart needs to
    look identical on screen. The curve is monotonic in both axes, so evenly
    spaced downsampling preserves its shape; only the (already-precise)
    ``auroc``/``gini``/``ks`` functions above need the full-precision data.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return fpr, tpr


def ks_points(y_true, y_score, max_points: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative (population_share, cum_good, cum_bad) for a KS chart, downsampled."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    x = np.linspace(0, 1, len(y_sorted))
    if len(x) > max_points:
        idx = np.unique(np.linspace(0, len(x) - 1, max_points).astype(int))
        x, cum_good, cum_bad = x[idx], cum_good[idx], cum_bad[idx]
    return x, cum_good, cum_bad


def psi(expected, actual, n_bins: int = 10) -> float:
    """Population Stability Index between two score/probability samples."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    eps = 1e-6
    exp_pct = np.histogram(expected, bins=edges)[0] / len(expected) + eps
    act_pct = np.histogram(actual, bins=edges)[0] / len(actual) + eps
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# ---------------------------------------------------------------------------
# Plotting (matplotlib) — used by the training script to save report figures.
# ---------------------------------------------------------------------------
def plot_roc(ax, curves: dict[str, tuple], title: str = "ROC curve") -> None:
    """*curves* maps a label to ``(y_true, y_score)``."""
    for label, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auroc(y_true, y_score):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")


def plot_ks(ax, y_true, y_score, title: str = "KS separation") -> None:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    x = np.linspace(0, 1, len(y_sorted))
    ks_idx = int(np.argmax(np.abs(cum_good - cum_bad)))

    ax.plot(x, cum_good, label="Cumulative good")
    ax.plot(x, cum_bad, label="Cumulative bad")
    ax.vlines(x[ks_idx], cum_bad[ks_idx], cum_good[ks_idx], color="red",
              label=f"KS={abs(cum_good[ks_idx] - cum_bad[ks_idx]):.3f}")
    ax.set_xlabel("Population sorted by score")
    ax.set_ylabel("Cumulative share")
    ax.set_title(title)
    ax.legend(loc="upper left")
