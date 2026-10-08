"""The shallow alignment network that produces the anchor.

LAIR-Net needs one target-aware representation, computed once, before any
random layer exists. It is a single-hidden-layer network fitted to the target;
the anchor is its hidden activations, not its predictions.

The reference implementation used ``sklearn.neural_network.MLPRegressor``, which
is slow for this shape and was capped at a fixed iteration count -- so the
"anchor" was whatever L-BFGS happened to reach, and the fitted model depended on
the cap. Here the objective and its gradient are written out, handed to a real
L-BFGS, and run to a convergence tolerance. Backends:

``numpy``   analytic gradient + ``scipy.optimize.minimize``. No dependency
            beyond scipy, which is already required. Chosen by ``auto`` below
            the large-sample threshold.
``numba``   the same gradient, JIT compiled. Chosen by ``auto`` at or above the
            threshold, when importable. Slower than numpy on small samples.
``torch``   autograd + ``torch.optim.LBFGS``, and the only backend that can use
            a GPU. Optional, and never chosen automatically: it was the fastest
            measured at 5000 samples but it is a heavy dependency, so selecting
            it is the caller's decision.
``sklearn`` the reference implementation, kept so the package can reproduce the
            paper's behaviour on request rather than only claim to.

Every backend minimises the same objective. That does **not** make them
interchangeable in their output: the problem is non-convex and they do not
share an initialisation, so they reach different local optima. What is asserted
in ``tests/test_backends.py`` is the weaker and true statement -- that from the
same start the numpy and numba paths agree to optimiser tolerance, and that
every backend reaches a comparable objective value.

See ``benchmarks/RESULTS.md`` for the measurements the ``auto`` policy rests on.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin


class ConvergenceWarning(UserWarning):
    """The anchor fit stopped on its iteration cap rather than on ``tol``."""


__all__ = ["AnchorNet", "available_backends", "ACTIVATIONS",
           "ConvergenceWarning"]


def _tanh(z):
    h = np.tanh(z)
    return h, 1.0 - h * h


def _relu(z):
    h = np.maximum(z, 0.0)
    return h, (z > 0.0).astype(z.dtype)


def _logistic(z):
    h = 0.5 * (1.0 + np.tanh(0.5 * z))
    return h, h * (1.0 - h)


def _identity(z):
    return z, np.ones_like(z)


#: Activation name -> ``f(z) -> (value, derivative)``.
ACTIVATIONS: dict = {
    "tanh": _tanh,
    "relu": _relu,
    "logistic": _logistic,
    "identity": _identity,
}


def available_backends() -> Tuple[str, ...]:
    """Backends importable in this environment.

    The order is discovery order and carries no meaning. Which backend is
    fastest depends on the sample size and reverses between the sizes measured;
    which one is used is decided by ``backend="auto"``, not by this list.
    """
    out = ["numpy"]
    for name, module in (("numba", "numba"), ("torch", "torch"),
                         ("sklearn", "sklearn")):
        try:
            __import__(module)
            out.append(name)
        except ImportError:
            pass
    return tuple(out)


# --------------------------------------------------------------- parameters
def _pack(W, b, V, c):
    return np.concatenate([W.ravel(), b, V.ravel(), c.ravel()])


def _unpack(theta, n_features, n_hidden, n_targets):
    i = 0
    W = theta[i:i + n_features * n_hidden].reshape(n_features, n_hidden)
    i += n_features * n_hidden
    b = theta[i:i + n_hidden]
    i += n_hidden
    V = theta[i:i + n_hidden * n_targets].reshape(n_hidden, n_targets)
    i += n_hidden * n_targets
    c = theta[i:i + n_targets]
    return W, b, V, c


def _init(n_features, n_hidden, n_targets, rng):
    # He scaling on the hidden layer, small on the readout: the anchor is a
    # representation, so the hidden units matter more than the output head.
    W = rng.standard_normal((n_features, n_hidden)) * np.sqrt(2.0 / n_features)
    b = np.zeros(n_hidden)
    V = rng.standard_normal((n_hidden, n_targets)) * np.sqrt(1.0 / n_hidden)
    c = np.zeros(n_targets)
    return _pack(W, b, V, c)


def _objective(theta, X, y, n_features, n_hidden, n_targets, l2, act):
    n = X.shape[0]
    W, b, V, c = _unpack(theta, n_features, n_hidden, n_targets)
    Z = X @ W + b
    H, dH = act(Z)
    resid = H @ V + c - y
    loss = 0.5 * np.sum(resid * resid) / n + 0.5 * l2 * np.dot(theta, theta) / n
    r = resid / n
    gV = H.T @ r + l2 * V / n
    gc = r.sum(axis=0) + l2 * c / n
    dZ = (r @ V.T) * dH
    gW = X.T @ dZ + l2 * W / n
    gb = dZ.sum(axis=0) + l2 * b / n
    return loss, _pack(gW, gb, gV, gc)


class AnchorNet(TransformerMixin, BaseEstimator):
    """Single-hidden-layer network whose hidden activations are the anchor.

    A scikit-learn transformer: ``fit`` then ``transform`` gives the anchor, so
    it composes in a ``Pipeline`` and can be cloned. It is exported publicly
    because the anchor is useful on its own -- it is a compact target-aware
    representation, and looking at it is the quickest way to tell whether
    LAIR-Net has anything to align to on a given problem.

    Parameters
    ----------
    n_hidden : int, default=100
        Width of the hidden layer. Must equal the width of the randomized
        layers, since the anchor is mixed into the hidden state directly.
    activation : {'tanh', 'relu', 'logistic', 'identity'}, default='tanh'
        Hidden nonlinearity.
    alpha : float, default=1e-4
        L2 penalty on all parameters. This is the knob that decides how closely
        the anchor tracks the training targets, and therefore how much noise it
        can carry into every layer.
    max_iter : int, default=5000
        L-BFGS iteration budget. A ceiling, not the stopping rule. The default
        is set from measurement, not taste: at ``tol=1e-8`` convergence took
        2873 iterations on a 400-sample problem and 3126 on a 2000-sample one,
        so a cap of 2000 -- the previous default -- would have bound on both.
        A cap that binds silently turns into an implicit regulariser, which is
        the defect this class exists to avoid, so ``fit`` warns when it does.
    tol : float, default=1e-8
        Convergence tolerance, applied to both the gradient norm and the
        objective change. Deliberately tight, because a loose tolerance stops
        the fit early and so acts as an implicit regulariser -- the same defect
        as an iteration cap, wearing a different name. Regularisation belongs
        to ``alpha`` where a user can see and tune it.

        Tightening further buys little: going from ``1e-8`` to ``1e-9`` cut the
        objective by roughly a quarter for two to three times the iterations on
        both problems measured.
    backend : {'numpy', 'numba', 'torch', 'sklearn'} or 'auto', default='auto'
        Optimiser implementation. ``'auto'`` picks ``numba`` when the sample is
        large enough for the JIT to pay for itself and ``numpy`` otherwise.
    device : str, optional
        Torch device, e.g. ``'cuda'``. Ignored by other backends.
    random_state : int, optional
        Seed for the parameter initialisation.
    callback : callable, optional
        Called as ``callback(loss)`` at every objective evaluation. A
        diagnostic and benchmarking hook, not a modelling parameter. It exists
        so that ``benchmarks/bench_anchor.py`` can trace **this** code rather
        than a parallel implementation of it -- the benchmark previously
        reimplemented the torch path, its copy was correct, and the shipped one
        had a bug in how it read the final loss.

        It is a constructor parameter, so it appears in ``get_params`` and is
        carried through ``sklearn.base.clone``. A bound method survives that
        intact, so a clone still writes to the object the callback was bound
        to. Note that ``clone(est).callback is est.callback`` is nonetheless
        ``False`` for a bound method, because CPython builds a fresh one on each
        attribute access; identity is not the thing to check.

        The ``sklearn`` backend provides no per-iteration hook, so it never
        calls this. It emits nothing rather than emitting one point and calling
        it a trajectory.

    Attributes
    ----------
    converged_ : bool
        Whether the fit converged. For the scipy-backed paths this is the
        optimiser's own ``success`` flag. For torch and sklearn, neither of
        which reports one, it is inferred from the iteration cap alone, and
        ``message_`` says so rather than letting the two look equivalent.
    success_, status_, message_
        The optimiser's verdict, code and text. ``status_`` is ``-1`` where the
        backend provides none.
    hit_max_iter_ : bool
        Whether the iteration cap was reached. Separate from ``converged_``,
        because a run can stop early for a reason other than convergence.
    n_iter_ : int
        L-BFGS iterations.
    n_func_evals_ : int
        Objective evaluations. Larger than ``n_iter_`` whenever a line search
        runs, and the two are not interchangeable across backends.
    loss_ : float
        Final objective value.
    """

    #: Sample size at or above which ``backend="auto"`` prefers numba.
    #:
    #: Set from the ``package`` benchmark profile, which runs at this class's
    #: own defaults, so the evidence describes the regime users are actually in.
    #: numba is 0.58x of numpy at n=1000 and 1.92x of it at n=5000, with
    #: clearly separated spreads in both directions.
    #:
    #: 5000 is the smallest *measured* size that clears the rule, not the
    #: crossover: the ``quick`` profile suggests numba is already ahead by
    #: n=2000, but that profile is a sanity check and is documented as not
    #: being threshold evidence. Tightening this needs a package-profile run at
    #: n=2000. See benchmarks/RESULTS.md. A heuristic, not a guarantee.
    _NUMBA_THRESHOLD = 5000

    def __init__(
        self,
        n_hidden: int = 100,
        activation: str = "tanh",
        alpha: float = 1e-4,
        max_iter: int = 5000,
        tol: float = 1e-8,
        backend: str = "auto",
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        callback: Optional[Callable] = None,
    ):
        self.n_hidden = n_hidden
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.backend = backend
        self.device = device
        self.random_state = random_state
        self.callback = callback

    # ------------------------------------------------------------ internals
    def _resolve_backend(self, n_samples: int) -> str:
        if self.backend != "auto":
            if self.backend not in available_backends():
                raise ValueError(
                    f"backend {self.backend!r} is not available here; "
                    f"available: {available_backends()}")
            return self.backend
        if n_samples >= self._NUMBA_THRESHOLD and "numba" in available_backends():
            return "numba"
        return "numpy"

    def _act(self) -> Callable:
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(ACTIVATIONS)}, "
                f"got {self.activation!r}")
        return ACTIVATIONS[self.activation]

    # ----------------------------------------------------------------- API
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AnchorNet":
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]}")

        n, d = X.shape
        t = y.shape[1]
        self.n_features_in_ = d
        self.n_targets_ = t
        backend = self._resolve_backend(n)
        self.backend_ = backend
        rng = np.random.default_rng(self.random_state)
        theta0 = _init(d, self.n_hidden, t, rng)

        if backend == "sklearn":
            self._fit_sklearn(X, y)
            return self
        if backend == "torch":
            self._fit_torch(X, y, theta0)
            return self

        fun = self._objective_for(backend, X, y, d, t)
        if self.callback is not None:
            _inner = fun

            def fun(theta):
                loss, grad = _inner(theta)
                self.callback(float(loss))
                return loss, grad

        # Both tolerances, not just gtol. L-BFGS-B stops on whichever fires
        # first, and with only gtol set the default ftol governed instead --
        # so the advertised `tol` had no effect on where the fit stopped.
        res = minimize(fun, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter, "gtol": self.tol,
                                "ftol": self.tol})
        self._store(res.x, d, t)
        self.n_iter_ = int(res.nit)
        self.n_func_evals_ = int(res.nfev)
        self.loss_ = float(res.fun)
        # res.success is the optimiser's own verdict and is the source of
        # truth. `nit < max_iter` only detects one failure mode and calls a
        # line-search breakdown a success, which is worse than not reporting.
        self.success_ = bool(res.success)
        self.status_ = int(res.status)
        self.message_ = str(res.message)
        self.converged_ = self.success_
        self.hit_max_iter_ = bool(res.nit >= self.max_iter)
        self._warn_if_capped()
        return self

    def _objective_for(self, backend, X, y, d, t):
        act = self._act()
        if backend == "numpy":
            return lambda th: _objective(th, X, y, d, self.n_hidden, t,
                                         self.alpha, act)
        from ._anchor_numba import make_objective  # local import: optional dep
        return make_objective(X, y, d, self.n_hidden, t, self.alpha,
                              self.activation)

    def _fit_sklearn(self, X, y):
        from sklearn.neural_network import MLPRegressor
        mdl = MLPRegressor(hidden_layer_sizes=(self.n_hidden,),
                           activation=self.activation, solver="lbfgs",
                           alpha=self.alpha, max_iter=self.max_iter,
                           tol=self.tol, random_state=self.random_state)
        target = y.ravel() if y.shape[1] == 1 else y
        mdl.fit(X, target)
        self._sklearn_model = mdl
        self.W_, self.b_ = mdl.coefs_[0], mdl.intercepts_[0]
        self.V_, self.c_ = mdl.coefs_[1], mdl.intercepts_[1]
        self.n_iter_ = int(mdl.n_iter_)
        self.n_func_evals_ = int(mdl.n_iter_)
        self.loss_ = float(getattr(mdl, "loss_", np.nan))
        self.hit_max_iter_ = bool(mdl.n_iter_ >= self.max_iter)
        self.success_ = not self.hit_max_iter_
        self.status_ = -1
        self.message_ = ("MLPRegressor exposes no convergence status; "
                         "inferred from the iteration cap only")
        self.converged_ = self.success_

    def _fit_torch(self, X, y, theta0):
        import torch
        dev = torch.device(self.device or "cpu")
        torch.manual_seed(self.random_state or 0)
        d, t = X.shape[1], y.shape[1]
        Xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
        yt = torch.as_tensor(y, dtype=torch.float64, device=dev)
        W, b, V, c = _unpack(theta0, d, self.n_hidden, t)
        params = [torch.tensor(a, dtype=torch.float64, device=dev,
                               requires_grad=True) for a in (W, b, V, c)]
        act = {"tanh": torch.tanh, "relu": torch.relu,
               "logistic": torch.sigmoid,
               "identity": lambda z: z}[self.activation]
        opt = torch.optim.LBFGS(params, max_iter=self.max_iter,
                                tolerance_grad=self.tol, history_size=10,
                                line_search_fn="strong_wolfe")
        state = {"n": 0}

        def closure():
            opt.zero_grad()
            Wp, bp, Vp, cp = params
            H = act(Xt @ Wp + bp)
            resid = H @ Vp + cp - yt
            reg = sum((p * p).sum() for p in params)
            loss = 0.5 * (resid * resid).sum() / len(yt) \
                + 0.5 * self.alpha * reg / len(yt)
            loss.backward()
            state["n"] += 1
            if self.callback is not None:
                self.callback(float(loss.detach()))
            return loss

        opt.step(closure)
        # torch.optim.LBFGS.step returns the loss from the FIRST closure
        # evaluation, not the last. Recording its return value stored the
        # pre-fit objective as `loss_`, which made a working backend look
        # broken -- the optimisation ran, the parameters moved, and every
        # diagnostic reported the starting value. Recompute instead.
        with torch.no_grad():
            Wp, bp, Vp, cp = params
            H = act(Xt @ Wp + bp)
            resid = H @ Vp + cp - yt
            reg = sum((p * p).sum() for p in params)
            loss = (0.5 * (resid * resid).sum() / len(yt)
                    + 0.5 * self.alpha * reg / len(yt))
        self.W_, self.b_, self.V_, self.c_ = (
            p.detach().cpu().numpy() for p in params)
        # A closure call is a function evaluation. L-BFGS calls it several
        # times per iteration during the line search, so reporting it as
        # `n_iter_` would not mean the same thing as the scipy backend's
        # `n_iter_`. The real iteration count lives in the optimiser state.
        self.n_func_evals_ = state["n"]
        inner = opt.state_dict().get("state", {}).get(0, {})
        self.n_iter_ = int(inner.get("n_iter", state["n"]))
        self.loss_ = float(loss)
        self.hit_max_iter_ = bool(self.n_iter_ >= self.max_iter)
        # torch.optim.LBFGS reports no success flag, so the only signal
        # available is whether it stopped before the cap. Recorded as such
        # rather than dressed up to look like the scipy verdict.
        self.success_ = not self.hit_max_iter_
        self.status_ = -1
        self.message_ = ("torch.optim.LBFGS exposes no convergence status; "
                         "inferred from the iteration cap only")
        self.converged_ = self.success_

    def _warn_if_capped(self) -> None:
        """Say so when the iteration cap stopped the fit.

        A capped fit is not a converged one, and silently returning it is how
        an optimiser budget turns into an unlabelled regularisation choice.
        """
        if getattr(self, "hit_max_iter_", False):
            warnings.warn(
                f"AnchorNet stopped at max_iter={self.max_iter} without "
                f"converging (loss={self.loss_:.6g}). The anchor is whatever "
                f"L-BFGS had reached, so the iteration cap is acting as an "
                f"implicit regulariser. Raise max_iter, loosen tol, or set "
                f"alpha deliberately.",
                ConvergenceWarning, stacklevel=3)

    def _store(self, theta, d, t):
        self.W_, self.b_, self.V_, self.c_ = _unpack(theta, d, self.n_hidden, t)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the anchor: the hidden activations for ``X``."""
        if not hasattr(self, "W_"):
            raise RuntimeError("AnchorNet is not fitted")
        X = np.ascontiguousarray(X, dtype=np.float64)
        h, _ = self._act()(X @ self.W_ + self.b_)
        return h

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Scalar output of the module. Not used by LAIR-Net; exposed because
        it is the quantity the anchor was fitted to, and a caller checking
        whether the anchor is any good will want it."""
        out = self.transform(X) @ self.V_ + self.c_
        return out.ravel() if out.shape[1] == 1 else out
