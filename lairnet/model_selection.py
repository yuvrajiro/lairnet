"""Hyperparameter search for LAIR-Net.

:class:`LAIRNetCV` is a thin wrapper over scikit-learn's search estimators. It
adds two things a plain ``GridSearchCV`` does not:

* a default grid built around ``align``, because that is the parameter whose
  useful value depends on the data and the one worth tuning first;
* delegation of the LAIR-specific methods -- ``layer_predictions``,
  ``prediction_band``, and the layer and anchor diagnostics -- to the selected
  model, so the search result is usable for interpretation without reaching
  into ``best_estimator_``.

Everything else is scikit-learn's. Splitters, scorers, ``n_jobs``, ``cv_results_``
and the refit behaviour all work as they do anywhere else, and nothing here
reimplements them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted

from .estimators import LAIRNetRegressor

__all__ = ["LAIRNetCV", "default_grid"]


def default_grid(*, wide: bool = False) -> dict:
    """A starting grid, centred on the parameters that matter.

    ``align`` first, because it decides how much the model relies on the
    anchor and its best value is data-dependent. ``ridge`` second, because the
    readouts are what actually fit. ``anchor_alpha`` third, because an anchor
    that fits noise propagates it into every layer.

    Depth is deliberately absent from the narrow grid. It is expensive to
    search and :func:`lairnet.inspection.depth_curve` answers the same question
    from a single fit.

    Parameters
    ----------
    wide : bool, default=False
        Add ``n_layers`` and ``aggregator``. Multiplies the grid by six, so it
        is opt-in.
    """
    grid = {
        "align": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9],
        "ridge": [1e-4, 1e-3, 1e-2, 1e-1],
        "anchor_alpha": [1e-4, 1e-2, 1.0],
    }
    if wide:
        grid["n_layers"] = [5, 10, 20]
        grid["aggregator"] = ["median", "mean"]
    return grid


class LAIRNetCV(MetaEstimatorMixin, BaseEstimator):
    """Search over LAIR-Net hyperparameters and keep the best model.

    Parameters
    ----------
    estimator : LAIRNetRegressor or LAIRNetClassifier, optional
        The estimator to tune. Defaults to a :class:`LAIRNetRegressor` with
        default parameters.
    param_grid : dict or list of dict, optional
        Passed to the underlying search. Defaults to :func:`default_grid`.
    search : {'grid', 'random'}, default='grid'
        Exhaustive or randomised. Use ``'random'`` with ``n_iter`` for wide
        grids.
    n_iter : int, default=30
        Candidates drawn when ``search='random'``.
    cv : int or splitter, default=5
    scoring : str or callable, optional
        Any scikit-learn scoring specification. Note that this is the
        *scikit-learn* convention, unlike the ``scorer(y_true, y_pred)``
        callables taken by :mod:`lairnet.inspection`; this class delegates to
        scikit-learn, so it uses scikit-learn's.
    n_jobs : int, optional
    refit : bool, default=True
        Refit the best candidate on the whole dataset. Required for the
        delegated methods to work.
    verbose : int, default=0
    error_score : 'raise' or numeric, default=numpy.nan
        What to record when a candidate fails to fit. ``numpy.nan`` keeps the
        search going and leaves the failure visible in ``cv_results_``.
    return_train_score : bool, default=False
        Add training scores to ``cv_results_``. Off by default because it costs
        an extra scoring pass per split.
    random_state : int, optional
        Seed for ``search='random'``.

    Attributes
    ----------
    best_estimator_, best_params_, best_score_, best_index_, cv_results_,
    n_splits_, scorer_ :
        As on the underlying scikit-learn search.

    Examples
    --------
    >>> from lairnet.model_selection import LAIRNetCV
    >>> from sklearn.datasets import make_friedman1
    >>> X, y = make_friedman1(n_samples=200, random_state=0)
    >>> search = LAIRNetCV(param_grid={'align': [0.0, 0.5]}, cv=3).fit(X, y)
    >>> search.best_params_['align'] in (0.0, 0.5)
    True
    """

    def __init__(
        self,
        estimator=None,
        param_grid: Mapping | Sequence[Mapping] | None = None,
        *,
        search: str = "grid",
        n_iter: int = 30,
        cv=5,
        scoring=None,
        n_jobs: int | None = None,
        refit: bool = True,
        verbose: int = 0,
        error_score=np.nan,
        return_train_score: bool = False,
        random_state: int | None = None,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.search = search
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.random_state = random_state

    # ------------------------------------------------------------- internals
    def _base_estimator(self):
        if self.estimator is not None:
            return clone(self.estimator)
        return LAIRNetRegressor()

    def _build(self):
        base = self._base_estimator()
        grid = self.param_grid if self.param_grid is not None else default_grid()
        common = {"cv": self.cv, "scoring": self.scoring, "n_jobs": self.n_jobs,
                      "refit": self.refit, "verbose": self.verbose,
                      "error_score": self.error_score,
                      "return_train_score": self.return_train_score}
        if self.search == "grid":
            return GridSearchCV(base, grid, **common)
        if self.search == "random":
            return RandomizedSearchCV(base, grid, n_iter=self.n_iter,
                                      random_state=self.random_state, **common)
        raise ValueError(
            f"search must be 'grid' or 'random', got {self.search!r}")

    # ------------------------------------------------------------------ API
    def fit(self, X, y):
        search = self._build().fit(X, y)
        self.search_ = search
        self.cv_results_ = search.cv_results_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        self.best_index_ = search.best_index_
        self.n_splits_ = search.n_splits_
        self.scorer_ = search.scorer_
        if self.refit:
            self.best_estimator_ = search.best_estimator_
            self.n_features_in_ = self.best_estimator_.n_features_in_
            names = getattr(self.best_estimator_, "feature_names_in_", None)
            if names is not None:
                self.feature_names_in_ = names
            if is_classifier(self.best_estimator_):
                self.classes_ = self.best_estimator_.classes_
        return self

    def _best(self):
        """The refitted model, or a message saying why there isn't one.

        Without this, ``refit=False`` reaches ``check_is_fitted`` and reports
        that the object is not fitted -- which is false and sends the reader
        looking for a missing ``fit`` call instead of at ``refit``.
        """
        if not self.refit and hasattr(self, "cv_results_"):
            raise AttributeError(
                "this LAIRNetCV was fitted with refit=False, so there is no "
                "selected model to delegate to. best_params_, best_score_ and "
                "cv_results_ are available; set refit=True to get predict, "
                "layer_predictions, prediction_band and explain.")
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_

    def predict(self, X):
        return self._best().predict(X)

    def score(self, X, y):
        return self._best().score(X, y)

    def predict_proba(self, X):
        return self._best().predict_proba(X)

    def predict_log_proba(self, X):
        return self._best().predict_log_proba(X)

    def decision_function(self, X):
        return self._best().decision_function(X)

    def layer_predictions(self, X):
        """Delegated to the selected model."""
        return self._best().layer_predictions(X)

    def prediction_band(self, X, coverage: float = 0.9):
        """Delegated to the selected model. Read its warning about what the
        band is and is not."""
        return self._best().prediction_band(X, coverage=coverage)

    def explain(self, X_train, y_train, X_eval=None, y_eval=None, **kwargs):
        """Run :func:`lairnet.explain.explain_model` on the selected model."""
        from .explain import explain_model

        return explain_model(self._best(), X_train, y_train, X_eval, y_eval,
                             **kwargs)

    def align_profile(self):
        """Best score available at each alignment, after tuning the rest.

        A by-product of the search that would otherwise need a second sweep.

        Where other parameters were searched too, several candidates share each
        ``align`` value, and this reports the **best** of them rather than
        their average -- so the curve answers "how well can the model do at
        this alignment", not "how well does it do on average there". The two
        differ whenever the best ridge or anchor penalty depends on ``align``,
        which it generally does.

        Returns
        -------
        (align_values, best_scores) or None
            ``None`` when ``align`` was not among the searched parameters.
        """
        check_is_fitted(self, "cv_results_")
        key = "param_align"
        if key not in self.cv_results_:
            return None
        values = np.array(self.cv_results_[key], dtype=float)
        scores = np.asarray(self.cv_results_["mean_test_score"], dtype=float)
        unique = np.unique(values)
        return unique, np.array([np.nanmax(scores[values == v]) for v in unique])

    def __getattr__(self, name):
        """Forward the fitted diagnostics to the selected model.

        ``layer_conditions_``, ``anchor_converged_`` and the rest live on the
        estimator, and a caller who has a search result should not have to know
        that. Only trailing-underscore names are forwarded, so a typo in a
        parameter name still raises rather than silently resolving.
        """
        if name.startswith("_") or not name.endswith("_"):
            raise AttributeError(name)
        best = self.__dict__.get("best_estimator_")
        if best is None:
            raise AttributeError(name)
        return getattr(best, name)
