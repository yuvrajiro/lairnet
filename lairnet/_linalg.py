"""Readout solvers.

LAIR-Net fits one ridge readout per depth on a design matrix
``D = [A, H]``, where ``A = [1, X]`` is fixed for the whole stack and only the
hidden block ``H`` changes from layer to layer. That structure is what makes the
readout cheap, and exploiting it is most of the speed story:

* the Gram matrix is built once per layer by three block products rather than
  one dense ``D.T @ D``, and the ``A.T @ A`` block is computed once for the
  whole stack;
* the solve is a Cholesky factorisation, not an explicit inverse, with an
  automatic fall back to an eigendecomposition when the Gram matrix is too
  ill-conditioned for Cholesky to be trusted;
* a single factorisation is reused across several ridge values, so sweeping the
  regularisation path costs one triangular solve per value instead of one
  factorisation per value.

The paper's implementation inverted the matrix directly with a fixed ``1e-8``
floor on the ridge, which is the sort of thing that ties published numbers to a
LAPACK version. Here the conditioning is measured and reported instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

__all__ = [
    "GramCache",
    "RidgeSolution",
    "solve_ridge",
    "solve_ridge_path",
]

# Above this 2-norm condition number a Cholesky solve is not trusted and the
# symmetric eigendecomposition is used instead. float64 carries ~16 digits, so
# 1e12 leaves four to spare.
_COND_LIMIT = 1e12


@dataclass
class RidgeSolution:
    """Coefficients and the diagnostics a caller needs to judge them."""

    coef: np.ndarray
    ridge: float
    condition: float
    method: str  # "cholesky" or "eigh"

    @property
    def well_conditioned(self) -> bool:
        return self.condition < _COND_LIMIT


class GramCache:
    """Reusable blocks of ``D.T @ D`` and ``D.T @ y`` for ``D = [A, H]``.

    The ``A`` block is identical at every depth, so its Gram block and its
    right-hand side are computed once and reused for the whole stack. Only the
    cross block ``A.T @ H`` and the hidden block ``H.T @ H`` are recomputed per
    layer.

    Parameters
    ----------
    A : ndarray of shape (n_samples, n_direct)
        The fixed block of the design matrix, usually ``[1, X]``.
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Targets. Two-dimensional ``y`` is solved for all targets at once.
    """

    def __init__(self, A: np.ndarray, y: np.ndarray):
        self.A = np.ascontiguousarray(A, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        if self.A.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"A has {self.A.shape[0]} rows but y has {self.y.shape[0]}")
        self.n_samples, self.n_direct = self.A.shape
        self._AtA = self.A.T @ self.A
        self._Aty = self.A.T @ self.y

    def build(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(D.T @ D, D.T @ y)`` for ``D = [A, H]``."""
        H = np.ascontiguousarray(H, dtype=np.float64)
        if H.shape[0] != self.n_samples:
            raise ValueError(
                f"H has {H.shape[0]} rows but the cache was built on "
                f"{self.n_samples}")
        AtH = self.A.T @ H
        HtH = H.T @ H
        p = self.n_direct + H.shape[1]
        gram = np.empty((p, p), dtype=np.float64)
        d = self.n_direct
        gram[:d, :d] = self._AtA
        gram[:d, d:] = AtH
        gram[d:, :d] = AtH.T
        gram[d:, d:] = HtH
        rhs = np.vstack([self._Aty, H.T @ self.y])
        return gram, rhs


def _condition(gram: np.ndarray, ridge: float) -> float:
    w = np.linalg.eigvalsh(gram)
    lo = w[0] + ridge
    hi = w[-1] + ridge
    if lo <= 0:
        return np.inf
    return float(hi / lo)


def solve_ridge(
    gram: np.ndarray,
    rhs: np.ndarray,
    ridge: float,
    *,
    check_condition: bool = True,
) -> RidgeSolution:
    """Solve ``(gram + ridge * I) coef = rhs``.

    Uses a Cholesky factorisation, falling back to a symmetric
    eigendecomposition when the regularised matrix is too ill-conditioned for
    Cholesky to be reliable. The condition number is returned either way, so a
    caller can tell a comfortable solve from a marginal one instead of guessing.

    Parameters
    ----------
    gram : ndarray of shape (n_features, n_features)
        ``D.T @ D``, symmetric positive semidefinite.
    rhs : ndarray of shape (n_features,) or (n_features, n_targets)
        ``D.T @ y``.
    ridge : float
        Regularisation strength, added to the diagonal. Must be positive;
        a zero ridge makes the problem singular whenever ``D`` is rank
        deficient, which is the common case at width above sample size.
    check_condition : bool, default=True
        Compute the condition number. Costs one symmetric eigenvalue
        decomposition; turn it off in inner loops where it has already been
        established that the problem is well behaved.
    """
    if ridge <= 0:
        raise ValueError(f"ridge must be positive, got {ridge}")
    p = gram.shape[0]
    if gram.shape != (p, p):
        raise ValueError(f"gram must be square, got {gram.shape}")

    regularised = gram + ridge * np.eye(p)
    cond = _condition(gram, ridge) if check_condition else float("nan")

    if not check_condition or cond < _COND_LIMIT:
        try:
            c, low = cho_factor(regularised, check_finite=False)
            coef = cho_solve((c, low), rhs, check_finite=False)
            return RidgeSolution(coef, ridge, cond, "cholesky")
        except np.linalg.LinAlgError:
            pass  # fall through to the robust path

    w, V = eigh(regularised, check_finite=False)
    w = np.maximum(w, np.finfo(np.float64).eps * w[-1])
    coef = V @ ((V.T @ rhs) / w[:, None])
    return RidgeSolution(coef, ridge, cond, "eigh")


def solve_ridge_path(
    gram: np.ndarray,
    rhs: np.ndarray,
    ridges,
) -> list:
    """Solve for several ridge values from one eigendecomposition.

    Sweeping the regularisation path naively costs one factorisation per value.
    A single symmetric eigendecomposition of the Gram matrix serves every value,
    because adding ``ridge`` to the diagonal shifts the eigenvalues and leaves
    the eigenvectors alone. For a path of ``k`` values this replaces ``k``
    factorisations with one.

    Returns
    -------
    list of RidgeSolution
        In the order the ridges were given.
    """
    ridges = [float(r) for r in ridges]
    if any(r <= 0 for r in ridges):
        raise ValueError("all ridge values must be positive")
    w, V = eigh(gram, check_finite=False)
    Vt_rhs = V.T @ rhs
    out = []
    for r in ridges:
        shifted = w + r
        coef = V @ (Vt_rhs / shifted[:, None])
        cond = float(shifted[-1] / shifted[0]) if shifted[0] > 0 else np.inf
        out.append(RidgeSolution(coef, r, cond, "eigh-path"))
    return out
