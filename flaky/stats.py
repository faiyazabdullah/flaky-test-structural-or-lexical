"""Small statistics helpers used by 05 and 07.

Kept separate so the numbers in the report trace to code that can be read on
its own.
"""
from __future__ import annotations

import math

import numpy as np


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_test(k: int, n: int, p: float = 0.5, alternative: str = "two-sided") -> float:
    if n == 0:
        return float("nan")
    from scipy.stats import binomtest

    return float(binomtest(k, n, p, alternative=alternative).pvalue)


def wilcoxon_one_sided(a, b, alternative: str = "greater") -> dict:
    """Paired one-sided Wilcoxon signed-rank of ``a`` against ``b``.

    Five folds is very little power -- the smallest attainable one-sided p is
    1/32 = 0.031 -- so the fold-level differences are returned alongside and
    the report prints them in full rather than only the verdict.
    """
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    out = {
        "n": int(len(d)),
        "a": a.tolist(),
        "b": b.tolist(),
        "diff": d.tolist(),
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "n_positive": int((d > 0).sum()),
        "alternative": alternative,
        "min_attainable_p": float(2.0 ** -len(d)) if len(d) else float("nan"),
    }
    if len(d) == 0 or np.allclose(d, 0):
        out.update({"statistic": float("nan"), "p": 1.0,
                    "note": "all paired differences are zero"})
        return out
    try:
        st = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
        out["statistic"] = float(st.statistic)
        out["p"] = float(st.pvalue)
    except ValueError as exc:
        out.update({"statistic": float("nan"), "p": float("nan"), "note": str(exc)})
    # a distribution-free effect size that survives n=5
    out["cliffs_delta"] = float(np.mean(np.sign(d)))
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    out["cohens_dz"] = float(d.mean() / sd) if sd > 0 else float("nan")
    return out
