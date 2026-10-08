"""The LAIR-Net forward pass and readout stack.

One shallow network is fitted to the target and its hidden activations become
the *anchor*. A stack of fixed random layers is then run, and at every depth the
state is pulled part-way back towards that anchor:

    Z = [1, H_prev, X]                 layer input, keeps the raw features
    Htilde = g(Z W)                    fixed random projection
    U = leak * Htilde + (1 - leak) * H_prev
    H = (1 - align) * U + align * S

A ridge readout is fitted on ``[1, X, H]`` at every depth and the layerwise
predictions are aggregated.

This module is the numerical core and holds no sklearn API. The estimators in
``lairnet.estimators`` wrap it.

Speed notes, since that is the point of this package:

* the layer input is never materialised. ``Z W`` is computed as three pieces --
  a bias row, ``H_prev @ Wh`` and ``X @ Wx`` -- straight into a preallocated
  buffer, which avoids building an ``(n, 1 + m + d)`` copy at every depth;
* the ``[1, X]`` block of the readout design is fixed for the whole stack, so
  its Gram block is formed once (see :class:`lairnet._linalg.GramCache`);
* all layer buffers are allocated once and reused, so a depth-20 stack does 20
  matrix products and no allocations;
* ``dtype`` may be ``float32``, which roughly halves the memory traffic in the
  projection. The readout solve is always done in float64 because that is where
  conditioning actually bites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._anchor import ACTIVATIONS, AnchorNet
from ._linalg import GramCache, RidgeSolution, solve_ridge

__all__ = ["LAIRCore", "LayerState", "AGGREGATORS"]


def _median(P: np.ndarray) -> np.ndarray:
    return np.median(P, axis=0)


def _mean(P: np.ndarray) -> np.ndarray:
    return P.mean(axis=0)


def _trimmed(P: np.ndarray, frac: float = 0.2) -> np.ndarray:
    """Mean of the middle ``1 - 2*frac`` of the layerwise predictions.

    Sits between the mean, which is efficient when every depth is sound, and
    the median, which ignores all but the middle one. Useful when a handful of
    depths are unreliable rather than exactly half.
    """
    k = P.shape[0]
    cut = int(np.floor(frac * k))
    if cut == 0:
        return P.mean(axis=0)
    S = np.sort(P, axis=0)
    return S[cut:k - cut].mean(axis=0)


#: Aggregation rule name -> callable over an ``(n_layers, n_samples, ...)`` array.
AGGREGATORS = {
    "median": _median,
    "mean": _mean,
    "trimmed": _trimmed,
}


@dataclass
class LayerState:
    """Everything one depth produced, kept for interpretation."""

    index: int
    coef: np.ndarray
    ridge: float
    condition: float
    solver: str
    train_score: float = float("nan")
    valid_score: float = float("nan")
    #: Mean absolute change in the hidden state introduced by this layer.
    state_delta: float = float("nan")
    #: Mean absolute distance from the anchor after this layer.
    anchor_distance: float = float("nan")


@dataclass
class LAIRCore:
    """Fixed random stack with a target-aware anchor and layerwise readouts.

    Parameters mirror the estimator; see
    :class:`lairnet.estimators.LAIRNetRegressor` for the user-facing docs.
    """

    n_hidden: int = 100
    n_layers: int = 20
    leak: float = 0.5
    align: float = 0.5
    ridge: float = 1e-3
    input_scale: float = 1.0
    activation: str = "relu"
    aggregator: str = "median"
    anchor_alpha: float = 1e-4
    anchor_activation: str = "tanh"
    anchor_backend: str = "auto"
    anchor_max_iter: int = 5000
    anchor_tol: float = 1e-8
    device: Optional[str] = None
    dtype: type = np.float64
    random_state: Optional[int] = None

    # -- fitted state -------------------------------------------------------
    anchor_: Optional[AnchorNet] = field(default=None, init=False)
    weights_: List[np.ndarray] = field(default_factory=list, init=False)
    layers_: List[LayerState] = field(default_factory=list, init=False)

    # ------------------------------------------------------------- helpers
    def _check(self) -> None:
        if not 0.0 <= self.leak <= 1.0:
            raise ValueError(f"leak must be in [0, 1], got {self.leak}")
        if not 0.0 <= self.align <= 1.0:
            raise ValueError(f"align must be in [0, 1], got {self.align}")
        if self.ridge <= 0:
            raise ValueError(f"ridge must be positive, got {self.ridge}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1, got {self.n_hidden}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(ACTIVATIONS)}, "
                f"got {self.activation!r}")
        if self.aggregator not in AGGREGATORS:
            raise ValueError(
                f"aggregator must be one of {sorted(AGGREGATORS)}, "
                f"got {self.aggregator!r}")

    def _draw_weights(self, n_features: int, rng) -> None:
        """Draw the random projection of every layer once.

        Stored split into the bias row, the state block and the input block,
        because that is how the forward pass consumes them and it saves a slice
        per layer per call.
        """
        m, d = self.n_hidden, n_features
        s = self.input_scale
        self.weights_ = []
        for _ in range(self.n_layers):
            W = rng.uniform(-s, s, size=(1 + m + d, m)).astype(self.dtype)
            self.weights_.append(W)

    # -------------------------------------------------------------- forward
    def _states(self, X: np.ndarray, S: np.ndarray) -> List[np.ndarray]:
        """Run the stack, returning the hidden state after every layer.

        Allocates two buffers and reuses them; the returned list holds copies,
        because callers need every depth at once for the readouts.
        """
        n, d = X.shape
        m = self.n_hidden
        act = ACTIVATIONS[self.activation]
        Xc = np.ascontiguousarray(X, dtype=self.dtype)
        Sc = np.ascontiguousarray(S, dtype=self.dtype)

        H = np.zeros((n, m), dtype=self.dtype)
        buf = np.empty((n, m), dtype=self.dtype)
        out: List[np.ndarray] = []

        for W in self.weights_:
            w0 = W[0]              # bias row
            Wh = W[1:1 + m]        # acts on the previous state
            Wx = W[1 + m:]         # acts on the raw inputs
            # buf = X @ Wx + H @ Wh + w0, without materialising [1, H, X].
            np.matmul(Xc, Wx, out=buf)
            buf += H @ Wh
            buf += w0
            Htilde, _ = act(buf)
            # U = leak * Htilde + (1 - leak) * H  then
            # H = (1 - align) * U + align * S, fused into one pass.
            a, g = self.align, self.leak
            H = (1.0 - a) * (g * Htilde + (1.0 - g) * H) + a * Sc
            out.append(H.copy())
        return out

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        validation: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        score_fn=None,
    ) -> "LAIRCore":
        """Fit the anchor, the stack and one readout per depth."""
        self._check()
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.ascontiguousarray(y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        self.anchor_ = AnchorNet(
            n_hidden=self.n_hidden, activation=self.anchor_activation,
            alpha=self.anchor_alpha, max_iter=self.anchor_max_iter,
            tol=self.anchor_tol, backend=self.anchor_backend,
            device=self.device, random_state=self.random_state,
        ).fit(X, Y)
        S = self.anchor_.transform(X)

        self._draw_weights(d, rng)
        states = self._states(X, S)

        A = np.hstack([np.ones((n, 1)), X])
        cache = GramCache(A, Y)
        self.layers_ = []
        prev = np.zeros_like(states[0])
        for i, H in enumerate(states):
            gram, rhs = cache.build(np.asarray(H, dtype=np.float64))
            sol: RidgeSolution = solve_ridge(gram, rhs, self.ridge)
            layer = LayerState(
                index=i, coef=sol.coef, ridge=sol.ridge,
                condition=sol.condition, solver=sol.method,
                state_delta=float(np.abs(H - prev).mean()),
                anchor_distance=float(np.abs(H - S).mean()),
            )
            if score_fn is not None:
                D = np.hstack([A, np.asarray(H, dtype=np.float64)])
                layer.train_score = float(score_fn(Y, D @ sol.coef))
            self.layers_.append(layer)
            prev = H

        if validation is not None and score_fn is not None:
            Xv, yv = validation
            Pv = self.layer_predictions(Xv)
            Yv = yv[:, None] if np.ndim(yv) == 1 else yv
            for i, layer in enumerate(self.layers_):
                layer.valid_score = float(score_fn(Yv, Pv[i]))
        return self

    # -------------------------------------------------------------- predict
    def layer_predictions(self, X: np.ndarray) -> np.ndarray:
        """Prediction from every depth, shape ``(n_layers, n_samples, n_targets)``.

        This is the object the interpretability tools consume. Aggregating it is
        one line; what it shows is how much the depths actually disagree, which
        is the quantity the median is protecting against.
        """
        if not self.layers_:
            raise RuntimeError("LAIRCore is not fitted")
        X = np.ascontiguousarray(X, dtype=np.float64)
        n = X.shape[0]
        S = self.anchor_.transform(X)
        states = self._states(X, S)
        A = np.hstack([np.ones((n, 1)), X])
        out = np.empty((len(states), n, self.layers_[0].coef.shape[1]))
        for i, (H, layer) in enumerate(zip(states, self.layers_)):
            D = np.hstack([A, np.asarray(H, dtype=np.float64)])
            out[i] = D @ layer.coef
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = self.layer_predictions(X)
        return AGGREGATORS[self.aggregator](P)

    def predict_interval(
        self, X: np.ndarray, coverage: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Empirical interval across depths.

        The stack produces ``n_layers`` predictions per sample, so a spread is
        available for free. It is a measure of how much the depths disagree, not
        a calibrated predictive interval, and the estimator documentation says
        so rather than letting a user assume otherwise.
        """
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        P = self.layer_predictions(X)
        lo = np.quantile(P, (1.0 - coverage) / 2.0, axis=0)
        hi = np.quantile(P, 1.0 - (1.0 - coverage) / 2.0, axis=0)
        return lo, hi
