"""scikit-learn estimators for LAIR-Net.

:class:`LAIRNetRegressor` and :class:`LAIRNetClassifier` are the user-facing
API. Both wrap :class:`lairnet._core.LAIRCore`, follow the scikit-learn
estimator contract, and work inside :class:`~sklearn.pipeline.Pipeline`,
:func:`~sklearn.model_selection.cross_val_score` and the search estimators.

Beyond ``fit``/``predict``, both expose what the stack learned: the prediction
from every depth, the conditioning of every readout, how far each layer moved
the hidden state, and the anchor optimiser's own verdict on its fit. Those are
the inputs to :mod:`lairnet.explain` and :mod:`lairnet.plotting`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y

from ._compat import validate_data

from ._core import AGGREGATORS, LAIRCore

__all__ = ["LAIRNetRegressor", "LAIRNetClassifier"]

_PARAM_DOC = """
    n_hidden : int, default=100
        Width of the randomized layers and of the anchor. The two must match,
        because the anchor is mixed into the hidden state directly.
    n_layers : int, default=20
        Depth of the randomized stack. One readout is fitted per depth.
    leak : float, default=0.5
        Leaky transition coefficient in ``[0, 1]``. How much of the new random
        response replaces the previous state; small values change the state
        slowly with depth.
    align : float, default=0.5
        Alignment coefficient in ``[0, 1]``. How strongly every layer is pulled
        back towards the anchor. At ``0`` the model is a leaky randomized stack
        with no target-aware component; at ``1`` every layer collapses onto the
        anchor and the random variation disappears. This is the parameter worth
        tuning first, and :class:`lairnet.model_selection.LAIRNetCV` selects it.
    ridge : float, default=1e-3
        Ridge penalty for the layerwise readouts. Must be positive.
    input_scale : float, default=1.0
        Half-width of the uniform distribution the random weights are drawn
        from.
    activation : {'relu', 'tanh', 'logistic', 'identity'}, default='relu'
        Nonlinearity of the randomized layers.
    aggregator : {'median', 'mean', 'trimmed'}, default='median'
        How the layerwise predictions are combined. ``'median'`` limits the
        influence of an unreliable minority of depths; ``'mean'`` is more
        efficient when every depth is sound; ``'trimmed'`` sits between them.
    anchor_alpha : float, default=1e-4
        L2 penalty on the anchor network. **This is the regularisation knob for
        the anchor.** An anchor that fits noise propagates that noise into every
        layer, so this matters more than its size suggests.
    anchor_activation : {'tanh', 'relu', 'logistic', 'identity'}, default='tanh'
        Nonlinearity of the anchor network.
    anchor_backend : {'auto', 'numpy', 'numba', 'torch', 'sklearn'}, default='auto'
        Optimiser used to fit the anchor. ``'auto'`` chooses by sample size.
        ``'sklearn'`` is a reference implementation, not a fast path.
    anchor_max_iter : int, default=5000
        Iteration ceiling for the anchor fit. A capped fit is not a converged
        one and ``fit`` warns when the cap binds, because an iteration budget
        that stops the fit early is an unlabelled regularisation choice.
    anchor_tol : float, default=1e-8
        Convergence tolerance for the anchor fit. Deliberately tight; see
        ``anchor_alpha`` for the knob that is meant to regularise.
    device : str, optional
        Torch device for the ``'torch'`` backend, e.g. ``'cuda'``. Measured to
        be slower than CPU at small sample sizes, where transfers dominate.
    dtype : dtype, default=numpy.float64
        Precision of the randomized stack buffers only. The anchor fit and the
        readout solves are always float64, because that is where conditioning
        bites. The package is float64-first; ``dtype`` is a memory-traffic knob
        for the projection, not a general precision setting.
    random_state : int, optional
        Seed for the random weights and the anchor initialisation.
"""

_ATTR_DOC = """
    n_features_in_ : int
        Features seen during ``fit``.
    feature_names_in_ : ndarray
        Feature names, when ``fit`` was given a DataFrame.
    layer_conditions_ : ndarray of shape (n_layers,)
        Condition number of each layerwise readout. Large values mean the
        solution at that depth is sensitive to the data.
    layer_solvers_ : list of str
        Which solver each depth used, ``'cholesky'`` or ``'eigh'``. Any
        ``'eigh'`` means the Cholesky path was not trusted there.
    layer_state_deltas_ : ndarray of shape (n_layers,)
        Mean absolute change in the hidden state introduced by each layer.
    layer_anchor_distances_ : ndarray of shape (n_layers,)
        Mean absolute distance from the anchor after each layer.
    layer_scores_ : ndarray of shape (n_layers,)
        Training score of each depth on its own.
    anchor_converged_ : bool
        Whether the anchor fit converged.
    anchor_status_, anchor_message_ :
        The anchor optimiser's status code and message. Backends that report no
        status say so in the message rather than presenting a guess as a fact.
    anchor_n_iter_, anchor_loss_, anchor_backend_ :
        Iterations taken, final objective, and which backend ran.
"""


class _LAIRNetBase(BaseEstimator):
    """Parameters and fitted diagnostics shared by both estimators."""

    def __init__(
        self,
        n_hidden: int = 100,
        n_layers: int = 20,
        leak: float = 0.5,
        align: float = 0.5,
        ridge: float = 1e-3,
        input_scale: float = 1.0,
        activation: str = "relu",
        aggregator: str = "median",
        anchor_alpha: float = 1e-4,
        anchor_activation: str = "tanh",
        anchor_backend: str = "auto",
        anchor_max_iter: int = 5000,
        anchor_tol: float = 1e-8,
        device: Optional[str] = None,
        dtype: type = np.float64,
        random_state: Optional[int] = None,
    ):
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.leak = leak
        self.align = align
        self.ridge = ridge
        self.input_scale = input_scale
        self.activation = activation
        self.aggregator = aggregator
        self.anchor_alpha = anchor_alpha
        self.anchor_activation = anchor_activation
        self.anchor_backend = anchor_backend
        self.anchor_max_iter = anchor_max_iter
        self.anchor_tol = anchor_tol
        self.device = device
        self.dtype = dtype
        self.random_state = random_state

    # ------------------------------------------------------------- internals
    def _make_core(self) -> LAIRCore:
        return LAIRCore(
            n_hidden=self.n_hidden, n_layers=self.n_layers, leak=self.leak,
            align=self.align, ridge=self.ridge, input_scale=self.input_scale,
            activation=self.activation, aggregator=self.aggregator,
            anchor_alpha=self.anchor_alpha,
            anchor_activation=self.anchor_activation,
            anchor_backend=self.anchor_backend,
            anchor_max_iter=self.anchor_max_iter, anchor_tol=self.anchor_tol,
            device=self.device, dtype=self.dtype,
            random_state=self.random_state,
        )

    def _check_validation(self, validation):
        """Coerce held-out data the same way ``fit`` coerced the training data.

        Without this the training ``X`` goes through validation and the
        held-out ``Xv`` does not, so a DataFrame or a mixed array-like reaches
        the core in a different form from the data the model was fitted on.
        Caught by RA2 in review.
        """
        if validation is None:
            return None
        try:
            Xv, yv = validation
        except (TypeError, ValueError):
            raise ValueError(
                "validation must be a (X, y) tuple, got "
                f"{type(validation).__name__}") from None
        Xv = validate_data(self, Xv, reset=False)
        yv = np.asarray(yv, dtype=np.float64)
        if Xv.shape[0] != yv.shape[0]:
            raise ValueError(
                f"validation X has {Xv.shape[0]} rows but y has {yv.shape[0]}")
        return Xv, yv

    def _record(self, core: LAIRCore) -> None:
        self.core_ = core
        self.layer_conditions_ = np.array([l.condition for l in core.layers_])
        self.layer_solvers_ = [l.solver for l in core.layers_]
        self.layer_state_deltas_ = np.array(
            [l.state_delta for l in core.layers_])
        self.layer_anchor_distances_ = np.array(
            [l.anchor_distance for l in core.layers_])
        self.layer_scores_ = np.array([l.train_score for l in core.layers_])
        a = core.anchor_
        self.anchor_converged_ = a.converged_
        self.anchor_status_ = a.status_
        self.anchor_message_ = a.message_
        self.anchor_n_iter_ = a.n_iter_
        self.anchor_n_func_evals_ = a.n_func_evals_
        self.anchor_loss_ = a.loss_
        self.anchor_backend_ = a.backend_
        self.anchor_hit_max_iter_ = a.hit_max_iter_

    # ------------------------------------------------------------------ API
    def layer_predictions(self, X) -> np.ndarray:
        """Prediction from every depth, ``(n_layers, n_samples, n_outputs)``.

        The raw material for interpretation. How much these disagree is what
        the median aggregator is protecting against, and looking at it is the
        quickest way to see whether depth is helping or thrashing.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.core_.layer_predictions(X)

    def prediction_band(self, X, coverage: float = 0.9):
        """Spread of the layerwise predictions.

        .. warning::

           This is a **depth-disagreement band, not a statistical confidence
           interval**. It describes how much the depths of this one fitted
           model disagree about each sample. It is not calibrated, carries no
           coverage guarantee, and will not widen for a sample far outside the
           training data unless the depths happen to disagree there.

        Returns
        -------
        lower, upper : ndarray
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.core_.predict_interval(X, coverage=coverage)

    def predict_interval(self, X, coverage: float = 0.9):
        """Alias of :meth:`prediction_band`. See its warning about what this
        is and is not."""
        return self.prediction_band(X, coverage=coverage)

    def save(self, filepath: str) -> None:
        """Serialise the fitted estimator with joblib."""
        import joblib

        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load an estimator written by :meth:`save`."""
        import joblib

        return joblib.load(filepath)


class LAIRNetRegressor(RegressorMixin, _LAIRNetBase):
    __doc__ = """Leaky Alignment-Impulse Residual Network for regression.

    A shallow network is fitted to the target and its hidden activations become
    an anchor. A stack of fixed random layers is then run, with every layer
    pulled part-way back towards that anchor, and a ridge readout is fitted at
    every depth. Only the anchor and the readouts are trained; the stack is
    never backpropagated through.

    Parameters
    ----------""" + _PARAM_DOC + """
    Attributes
    ----------""" + _ATTR_DOC + """
    Examples
    --------
    >>> from lairnet import LAIRNetRegressor
    >>> from sklearn.datasets import make_friedman1
    >>> X, y = make_friedman1(n_samples=300, random_state=0)
    >>> model = LAIRNetRegressor(n_layers=10, random_state=0).fit(X, y)
    >>> model.predict(X[:3]).shape
    (3,)
    """

    def fit(self, X, y, validation=None):
        """Fit the anchor, the randomized stack and one readout per depth.

        Parameters
        ----------
        X, y : array-like
        validation : tuple of (X, y), optional
            Held-out data used only to record a per-depth validation score in
            ``layer_valid_scores_``. It does not change the fit.
        """
        X, y = validate_data(self, X, y, y_numeric=True, multi_output=True)
        self._y_ndim = np.ndim(y)
        core = self._make_core().fit(
            X, y, validation=self._check_validation(validation),
            score_fn=r2_score)
        self._record(core)
        self.layer_valid_scores_ = np.array(
            [l.valid_score for l in core.layers_])
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        out = self.core_.predict(X)
        return out.ravel() if self._y_ndim == 1 else out


class LAIRNetClassifier(ClassifierMixin, _LAIRNetBase):
    __doc__ = """Leaky Alignment-Impulse Residual Network for classification.

    The classes are one-hot encoded and the same regression machinery is fitted
    to the indicator columns, which is the standard route for randomized
    networks with a closed-form readout. ``predict`` takes the arg-max of the
    aggregated scores, and ``predict_proba`` passes them through a softmax.

    .. note::

       The softmax is a monotone squashing of scores that were fitted by least
       squares, not a likelihood. The ordering is meaningful; treat the
       magnitudes as scores rather than as calibrated probabilities, and
       calibrate explicitly if you need them to be.

    Parameters
    ----------""" + _PARAM_DOC + """
    Attributes
    ----------
    classes_ : ndarray
        The class labels seen during ``fit``.""" + _ATTR_DOC

    def fit(self, X, y, validation=None):
        """Fit the model to one-hot encoded class indicators."""
        X, y = check_X_y(X, y, multi_output=False)
        X = validate_data(self, X, reset=True)
        self.classes_ = unique_labels(y)
        if len(self.classes_) < 2:
            raise ValueError(
                f"classifier needs at least 2 classes, got {len(self.classes_)}")
        self._binarizer = LabelBinarizer().fit(y)
        Y = self._binarizer.transform(y).astype(np.float64)
        if Y.shape[1] == 1:  # binary: LabelBinarizer gives one column
            Y = np.hstack([1.0 - Y, Y])

        def score_fn(true, pred):
            return accuracy_score(true.argmax(axis=1), pred.argmax(axis=1))

        core = self._make_core().fit(
            X, Y, validation=self._check_validation_labels(validation),
            score_fn=score_fn)
        self._record(core)
        self.layer_valid_scores_ = np.array(
            [l.valid_score for l in core.layers_])
        return self

    def _check_validation_labels(self, validation):
        """As ``_check_validation``, then one-hot encode the held-out labels.

        The core is fitted on indicator columns, so held-out labels have to be
        encoded identically; otherwise the per-depth validation score compares
        a label vector against an indicator matrix.
        """
        checked = self._check_validation(validation)
        if checked is None:
            return None
        Xv, yv = checked
        Yv = self._binarizer.transform(yv).astype(np.float64)
        if Yv.shape[1] == 1:
            Yv = np.hstack([1.0 - Yv, Yv])
        return Xv, Yv

    def decision_function(self, X):
        """Aggregated per-class scores, before the arg-max."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.core_.predict(X)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def predict_proba(self, X):
        """Softmax of the aggregated scores. See the note in the class docs
        about what these are and are not."""
        scores = self.decision_function(X)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-300, None))
