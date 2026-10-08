"""Model inspection tools.

These answer questions a LAIR-Net user actually has and a generic explainer
cannot: which features the model uses, how much the anchor is contributing,
whether depth is helping, and how much the depths disagree.

Everything here returns arrays and dataclasses. Nothing draws. The functions in
:mod:`lairnet.plotting` consume these results, which keeps matplotlib out of the
fitting path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "ImportanceResult",
    "DepthCurve",
    "AlignSensitivity",
    "LayerAgreement",
    "permutation_importance",
    "depth_curve",
    "align_sensitivity",
    "layer_agreement",
    "anchor_contribution",
]


def _default_scorer(estimator):
    return accuracy_score if is_classifier(estimator) else r2_score


def _check_scorer(scorer):
    """Reject scikit-learn scorer objects with a message that says what to do.

    This module's scorers take ``(y_true, y_pred)``, the ``sklearn.metrics``
    signature. A scorer built by ``sklearn.metrics.get_scorer`` or
    ``make_scorer`` takes ``(estimator, X, y)`` instead, and passing one here
    otherwise fails deep inside with an error about array shapes that says
    nothing about the real mistake.
    """
    if scorer is None:
        return None
    if hasattr(scorer, "_score_func") or type(scorer).__name__.endswith("Scorer"):
        raise TypeError(
            "scorer must take (y_true, y_pred), like sklearn.metrics.r2_score. "
            f"Got {type(scorer).__name__}, which is a scikit-learn scorer "
            "object taking (estimator, X, y). Pass the metric itself, for "
            "example `sklearn.metrics.r2_score`, or "
            "`sklearn.metrics.get_scorer('r2')._score_func`.")
    return scorer


def _score(estimator, X, y, scorer) -> float:
    return float(scorer(y, estimator.predict(X)))


@dataclass
class ImportanceResult:
    """Permutation importance, per feature.

    Fields are documented inline below rather than in an ``Attributes`` section,
    because autodoc renders a dataclass field and a matching docstring entry as
    two separate entries for the same thing.
    """

    #: Mean drop in score over repeats, one per feature.
    importances_mean: np.ndarray
    #: Standard deviation of the drop over repeats.
    importances_std: np.ndarray
    #: Every repeat, shape ``(n_features, n_repeats)``. Kept so a caller can see
    #: the spread rather than trust a mean over a handful of shuffles.
    importances: np.ndarray
    #: Score before any shuffling.
    baseline_score: float
    #: Feature names when the estimator was fitted with them, else ``None``.
    feature_names: Optional[List[str]] = None

    def ranking(self) -> np.ndarray:
        """Feature indices, most important first."""
        return np.argsort(self.importances_mean)[::-1]

    def top(self, k: int = 10):
        """``(name, mean, std)`` for the ``k`` most important features."""
        out = []
        for i in self.ranking()[:k]:
            name = (self.feature_names[i] if self.feature_names
                    else f"feature {i}")
            out.append((name, float(self.importances_mean[i]),
                        float(self.importances_std[i])))
        return out


def permutation_importance(
    estimator,
    X,
    y,
    *,
    n_repeats: int = 10,
    scorer: Optional[Callable] = None,
    random_state: Optional[int] = None,
) -> ImportanceResult:
    """Drop in score when each feature is shuffled.

    Model-agnostic and applied to the fitted estimator, so it measures what
    this model uses rather than what the data contains. A feature that is
    important but duplicated elsewhere will look unimportant here, which is a
    property of permutation importance and not of LAIR-Net.

    Parameters
    ----------
    estimator : fitted LAIRNetRegressor or LAIRNetClassifier
    X, y : array-like
        Held-out data. Running this on the training set measures what the model
        memorised.
    n_repeats : int, default=10
        Shuffles per feature.
    scorer : callable, optional
        ``scorer(y_true, y_pred) -> float``. Defaults to accuracy for
        classifiers and R² for regressors.
    """
    check_is_fitted(estimator)
    scorer = _check_scorer(scorer) or _default_scorer(estimator)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    baseline = _score(estimator, X, y, scorer)
    drops = np.empty((X.shape[1], n_repeats))
    work = X.copy()
    for j in range(X.shape[1]):
        original = work[:, j].copy()
        for r in range(n_repeats):
            rng.shuffle(work[:, j])
            drops[j, r] = baseline - _score(estimator, work, y, scorer)
        work[:, j] = original

    names = getattr(estimator, "feature_names_in_", None)
    return ImportanceResult(
        importances_mean=drops.mean(axis=1),
        importances_std=drops.std(axis=1),
        importances=drops,
        baseline_score=baseline,
        feature_names=list(names) if names is not None else None,
    )


@dataclass
class DepthCurve:
    """Score as a function of how many depths are aggregated."""

    #: Depths evaluated, ``1 .. n_layers``.
    depths: np.ndarray
    #: Score from aggregating the first ``k`` depths.
    scores: np.ndarray
    #: Score from depth ``k`` on its own.
    per_layer_scores: np.ndarray

    @property
    def best_depth(self) -> int:
        """Depth at which the aggregated score peaks.

        If this is well below ``n_layers``, the extra depth is not paying for
        itself on this problem.
        """
        return int(self.depths[int(np.argmax(self.scores))])


def depth_curve(estimator, X, y, *, scorer: Optional[Callable] = None) -> DepthCurve:
    """Aggregate the first ``k`` depths, for every ``k``, and score each.

    Answers whether depth is helping. The layerwise predictions are computed
    once and re-aggregated, so this costs one forward pass rather than
    ``n_layers`` refits.
    """
    check_is_fitted(estimator)
    scorer = _check_scorer(scorer) or _default_scorer(estimator)
    from ._core import AGGREGATORS

    P = estimator.layer_predictions(X)
    agg = AGGREGATORS[estimator.aggregator]
    y = np.asarray(y)

    def finish(pred):
        if is_classifier(estimator):
            return estimator.classes_[pred.argmax(axis=1)]
        return pred.ravel() if pred.shape[1] == 1 else pred

    scores, per_layer = [], []
    for k in range(1, P.shape[0] + 1):
        scores.append(float(scorer(y, finish(agg(P[:k])))))
        per_layer.append(float(scorer(y, finish(P[k - 1]))))
    return DepthCurve(np.arange(1, P.shape[0] + 1), np.array(scores),
                      np.array(per_layer))


@dataclass
class AlignSensitivity:
    """Score against the alignment coefficient."""

    #: Alignment coefficients evaluated.
    align_values: np.ndarray
    #: Score at each coefficient.
    scores: np.ndarray
    #: The coefficient the estimator under inspection was fitted with.
    fitted_align: float

    @property
    def best_align(self) -> float:
        return float(self.align_values[int(np.argmax(self.scores))])

    @property
    def anchor_gain(self) -> float:
        """Score at the best alignment minus the score at ``align=0``.

        ``align=0`` removes the anchor entirely, so this is what the
        target-aware component is worth on this data. A value near zero means
        the model is working as a plain leaky randomized stack.
        """
        zero = self.align_values == 0.0
        if not zero.any():
            return float("nan")
        return float(self.scores.max() - self.scores[zero][0])


def align_sensitivity(
    estimator,
    X,
    y,
    *,
    align_values: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9),
    X_valid=None,
    y_valid=None,
    scorer: Optional[Callable] = None,
) -> AlignSensitivity:
    """Refit across alignment coefficients and score each.

    The alignment coefficient is the parameter that decides how much the model
    relies on the anchor, and its useful value depends on the data. This traces
    that dependence directly.

    Unlike the other functions here this one **refits**, once per value, so it
    costs what it looks like it costs.
    """
    scorer = _check_scorer(scorer) or _default_scorer(estimator)
    Xv = X if X_valid is None else X_valid
    yv = y if y_valid is None else y_valid
    scores = []
    for a in align_values:
        est = clone(estimator).set_params(align=float(a)).fit(X, y)
        scores.append(_score(est, Xv, yv, scorer))
    return AlignSensitivity(np.asarray(align_values, dtype=float),
                            np.asarray(scores), float(estimator.align))


@dataclass
class LayerAgreement:
    """How much the depths disagree, per sample and overall."""

    #: Standard deviation across depths, one per sample.
    per_sample_std: np.ndarray
    #: Mean of ``per_sample_std``.
    mean_std: float
    #: Correlation between the depths' predictions, ``(n_layers, n_layers)``.
    correlation: np.ndarray

    @property
    def mean_correlation(self) -> float:
        """Mean off-diagonal correlation between depths.

        Near one means the depths are near-duplicates and aggregation is
        buying little. This is the quantity the paper's aggregation
        proposition is about: the variance reduction available from averaging
        depends on them being uncorrelated, and here they are not.
        """
        k = self.correlation.shape[0]
        if k < 2:
            return float("nan")
        off = ~np.eye(k, dtype=bool)
        return float(self.correlation[off].mean())


def layer_agreement(estimator, X) -> LayerAgreement:
    """Spread and correlation of the layerwise predictions."""
    P = estimator.layer_predictions(X)
    flat = P.reshape(P.shape[0], -1)
    return LayerAgreement(
        per_sample_std=P.std(axis=0),
        mean_std=float(P.std(axis=0).mean()),
        correlation=np.corrcoef(flat),
    )


def anchor_contribution(
    estimator,
    X_train,
    y_train,
    X_eval,
    y_eval,
    *,
    scorer: Optional[Callable] = None,
):
    """Score with the anchor and with it switched off.

    Refits the same architecture at ``align=0``, which removes the target-aware
    component entirely, and scores both on the same held-out data. The
    difference is what the anchor is worth here.

    Training and evaluation data are separate **required** arguments, and
    deliberately so. An earlier signature took one ``(X, y)`` pair, refitted the
    ablation on it and scored both arms on it. Handed a test set that produces
    an in-sample score for the ablation and an out-of-sample score for the
    original, and the comparison silently reverses: in testing it reported the
    anchor *hurting* by 0.09 when a proper split showed it helping by 0.15.

    Parameters
    ----------
    estimator : fitted LAIRNetRegressor or LAIRNetClassifier
    X_train, y_train : array-like
        The data the estimator was fitted on. The ablation is refitted on it so
        the two arms differ only in ``align``.
    X_eval, y_eval : array-like
        Held-out data. Both arms are scored on this.

    Returns
    -------
    dict with ``with_anchor``, ``without_anchor`` and ``gain``.
    """
    scorer = _check_scorer(scorer) or _default_scorer(estimator)
    base = _score(estimator, X_eval, y_eval, scorer)
    off = clone(estimator).set_params(align=0.0).fit(X_train, y_train)
    without = _score(off, X_eval, y_eval, scorer)
    return {"with_anchor": base, "without_anchor": without,
            "gain": base - without}
