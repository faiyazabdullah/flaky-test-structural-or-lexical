"""Cross-validation regimes, feature builders and the fit/score loop.

Two regimes, both 5-fold, stratified, seed 0:

``within``  ``StratifiedKFold`` -- tests from one project appear on both sides.
            Measures interpolation.
``cross``   ``StratifiedGroupKFold`` grouped by project.  No project spans the
            split.  Measures generalisation.

Every vectoriser is fitted **inside** the fold.  ``C`` is tuned on an inner
split of the training rows, never on a test fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Java identifiers, case preserved -- camelCase carries signal and lowercasing
# destroys it.
JAVA_TOKEN_PATTERN = r"[A-Za-z_$][A-Za-z0-9_$]*"

C_GRID = (0.01, 0.1, 1.0, 10.0)
N_SPLITS = 5


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def make_folds(y, groups, regime: str, n_splits: int = N_SPLITS, seed: int = 0):
    """Fold indices for one regime.  ``groups`` is the project of each row."""
    y = np.asarray(y)
    if regime == "within":
        sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in sp.split(np.zeros(len(y)), y)]
    if regime == "cross":
        sp = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in sp.split(np.zeros(len(y)), y, groups=np.asarray(groups))]
    raise ValueError(f"unknown regime {regime!r}")


def inner_folds(y_tr, groups_tr, regime: str, n_splits: int = 3, seed: int = 0):
    """Inner split used only for tuning ``C``.  Mirrors the outer regime so the
    tuned value is not optimistic about project leakage."""
    y_tr = np.asarray(y_tr)
    n_pos = int(y_tr.sum())
    k = max(2, min(n_splits, n_pos))
    if regime == "cross":
        n_groups = len(set(map(str, groups_tr)))
        k = max(2, min(k, n_groups))
        sp = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(sp.split(np.zeros(len(y_tr)), y_tr, groups=np.asarray(groups_tr)))
    sp = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(sp.split(np.zeros(len(y_tr)), y_tr))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def score(y_true, s) -> dict:
    y_true = np.asarray(y_true)
    s = np.asarray(s, dtype=np.float64)
    out = {"ap": float(average_precision_score(y_true, s)),
           "prior": float(y_true.mean()),
           "n": int(len(y_true)),
           "n_pos": int(y_true.sum())}
    out["auroc"] = float(roc_auc_score(y_true, s)) if 0 < out["n_pos"] < out["n"] else float("nan")
    return out


# --------------------------------------------------------------------------
# feature builders
# --------------------------------------------------------------------------


class LengthFeatures:
    """`n_chars`, `n_lines`, `n_tokens`, standardised on the training fold."""

    name = "length"

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, rows, y=None):
        self.scaler.fit(self._raw(rows))
        return self

    def transform(self, rows):
        return self.scaler.transform(self._raw(rows))

    @staticmethod
    def _raw(rows):
        return np.asarray([[r["n_chars"], r["n_lines"], r["n_tokens"]] for r in rows],
                          dtype=np.float64)


class BowFeatures:
    """TF-IDF over Java identifiers.  ``drop`` removes cue columns after
    fitting -- intervention A is feature-space only, the code is untouched."""

    def __init__(self, name="bow", drop: set[str] | None = None, min_df=3,
                 analyzer="word", ngram_range=(1, 1)):
        self.name = name
        self.drop = set(drop or ())
        self.vec = TfidfVectorizer(
            lowercase=False,
            token_pattern=JAVA_TOKEN_PATTERN if analyzer == "word" else None,
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )
        self.keep = None

    def fit(self, rows, y=None):
        self.vec.fit([r["text"] for r in rows])
        feats = self.vec.get_feature_names_out()
        if self.drop:
            self.keep = np.asarray([f not in self.drop for f in feats])
            self.n_dropped = int((~self.keep).sum())
        else:
            self.keep = None
            self.n_dropped = 0
        return self

    def transform(self, rows):
        X = self.vec.transform([r["text"] for r in rows])
        return X[:, self.keep] if self.keep is not None else X


class HStack:
    """Sparse bag-of-tokens concatenated with the standardised length block."""

    def __init__(self, name, parts):
        self.name = name
        self.parts = parts

    def fit(self, rows, y=None):
        for p in self.parts:
            p.fit(rows, y)
        return self

    def transform(self, rows):
        import scipy.sparse as sp

        mats = [p.transform(rows) for p in self.parts]
        if any(sp.issparse(m) for m in mats):
            return sp.hstack([sp.csr_matrix(m) for m in mats], format="csr")
        return np.hstack(mats)


def build_featurizer(method: str, cue_drop: set[str] | None = None):
    if method == "length":
        return LengthFeatures()
    if method == "bow":
        return BowFeatures("bow")
    if method == "bow_ablated":
        return BowFeatures("bow_ablated", drop=cue_drop or set())
    if method == "char_ngram":
        return BowFeatures("char_ngram", analyzer="char", ngram_range=(3, 5), min_df=3)
    if method == "bow_plus_length":
        return HStack("bow_plus_length", [BowFeatures("bow"), LengthFeatures()])
    raise ValueError(f"unknown method {method!r}")


BASELINE_METHODS = ("length", "bow", "bow_ablated", "char_ngram", "bow_plus_length")
# S1 is evaluated against the best of these -- the *word*-based baselines.
WORD_BASELINE_METHODS = ("bow", "char_ngram", "bow_plus_length")


# --------------------------------------------------------------------------
# classifier
# --------------------------------------------------------------------------


def make_clf(C: float, seed: int = 0, max_iter: int = 2000) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        solver="liblinear" if C <= 0 else "lbfgs",
        random_state=seed,
        n_jobs=None,
    )


def tune_C(Xtr, ytr, groups_tr, regime: str, seed: int = 0,
           grid=C_GRID, max_iter: int = 2000) -> tuple[float, dict]:
    """Pick ``C`` by mean AP over an inner split of the training rows."""
    folds = inner_folds(ytr, groups_tr, regime, seed=seed)
    best, best_ap, table = grid[0], -1.0, {}
    for C in grid:
        aps = []
        for itr, ite in folds:
            if len(set(np.asarray(ytr)[itr])) < 2 or len(set(np.asarray(ytr)[ite])) < 2:
                continue
            clf = make_clf(C, seed=seed, max_iter=max_iter)
            clf.fit(Xtr[itr], np.asarray(ytr)[itr])
            aps.append(average_precision_score(np.asarray(ytr)[ite],
                                               clf.decision_function(Xtr[ite])))
        m = float(np.mean(aps)) if aps else float("nan")
        table[str(C)] = m
        if aps and m > best_ap:
            best, best_ap = C, m
    return best, table


def fit_and_score(Xtr, ytr, groups_tr, Xte, yte, regime: str, seed: int = 0,
                  grid=C_GRID, max_iter: int = 2000, extra_eval: dict | None = None):
    """Tune, refit on the full training fold, score on the test fold.

    ``extra_eval`` maps a name to an already-transformed test matrix (used for
    the renaming transfer evaluation), scored with the same fitted model.
    """
    C, table = tune_C(Xtr, ytr, groups_tr, regime, seed=seed, grid=grid, max_iter=max_iter)
    clf = make_clf(C, seed=seed, max_iter=max_iter)
    clf.fit(Xtr, ytr)
    res = {"C": C, "C_table": table, **score(yte, clf.decision_function(Xte))}
    for k, Xk in (extra_eval or {}).items():
        res[k] = score(yte, clf.decision_function(Xk))
    return res, clf
