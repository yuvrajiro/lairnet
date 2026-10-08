"""Numba-compiled objective for :class:`~lairnet._anchor.AnchorNet`.

Same mathematics as the NumPy path in ``_anchor.py``; the loops that NumPy has
to materialise as temporaries are fused here instead. Imported lazily, so numba
is never required to import the package.

The JIT costs roughly a second on first call, which is why ``AnchorNet``
switches to this backend only above a sample-size threshold. The threshold is
set from ``benchmarks/bench_anchor.py``, not guessed.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - guarded by available_backends
    raise ImportError(
        "the numba backend requires numba; install it or use "
        "backend='numpy'") from exc


@njit(cache=True, fastmath=True)
def _loss_grad(theta, X, y, n_features, n_hidden, n_targets, l2, act_code):
    n = X.shape[0]
    off = 0
    W = theta[off:off + n_features * n_hidden].reshape(n_features, n_hidden)
    off += n_features * n_hidden
    b = theta[off:off + n_hidden]
    off += n_hidden
    V = theta[off:off + n_hidden * n_targets].reshape(n_hidden, n_targets)
    off += n_hidden * n_targets
    c = theta[off:off + n_targets]

    Z = X @ W
    H = np.empty_like(Z)
    dH = np.empty_like(Z)
    for i in range(n):
        for j in range(n_hidden):
            z = Z[i, j] + b[j]
            if act_code == 0:      # tanh
                h = np.tanh(z)
                H[i, j] = h
                dH[i, j] = 1.0 - h * h
            elif act_code == 1:    # relu
                H[i, j] = z if z > 0.0 else 0.0
                dH[i, j] = 1.0 if z > 0.0 else 0.0
            elif act_code == 2:    # logistic
                h = 0.5 * (1.0 + np.tanh(0.5 * z))
                H[i, j] = h
                dH[i, j] = h * (1.0 - h)
            else:                  # identity
                H[i, j] = z
                dH[i, j] = 1.0

    resid = H @ V
    for i in range(n):
        for k in range(n_targets):
            resid[i, k] += c[k] - y[i, k]

    sq = 0.0
    for i in range(n):
        for k in range(n_targets):
            sq += resid[i, k] * resid[i, k]
    reg = 0.0
    for i in range(theta.shape[0]):
        reg += theta[i] * theta[i]
    loss = 0.5 * sq / n + 0.5 * l2 * reg / n

    r = resid / n
    gV = H.T @ r + l2 * V / n
    gc = np.zeros(n_targets)
    for k in range(n_targets):
        s = 0.0
        for i in range(n):
            s += r[i, k]
        gc[k] = s + l2 * c[k] / n

    dZ = r @ V.T
    for i in range(n):
        for j in range(n_hidden):
            dZ[i, j] *= dH[i, j]
    gW = X.T @ dZ + l2 * W / n
    gb = np.zeros(n_hidden)
    for j in range(n_hidden):
        s = 0.0
        for i in range(n):
            s += dZ[i, j]
        gb[j] = s + l2 * b[j] / n

    g = np.empty_like(theta)
    off = 0
    g[off:off + n_features * n_hidden] = gW.ravel()
    off += n_features * n_hidden
    g[off:off + n_hidden] = gb
    off += n_hidden
    g[off:off + n_hidden * n_targets] = gV.ravel()
    off += n_hidden * n_targets
    g[off:off + n_targets] = gc
    return loss, g


_ACT_CODE = {"tanh": 0, "relu": 1, "logistic": 2, "identity": 3}


def make_objective(X, y, n_features, n_hidden, n_targets, l2, activation):
    """Return a ``theta -> (loss, grad)`` closure, with the JIT already warm.

    Warming inside this factory keeps compilation out of the optimiser's first
    iteration, where it would otherwise be charged to the line search.
    """
    code = _ACT_CODE[activation]
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    warm = np.zeros(n_features * n_hidden + n_hidden
                    + n_hidden * n_targets + n_targets)
    _loss_grad(warm, X, y, n_features, n_hidden, n_targets, l2, code)

    def objective(theta):
        return _loss_grad(np.ascontiguousarray(theta), X, y, n_features,
                          n_hidden, n_targets, l2, code)

    return objective
