"""Tests for the readout solvers.

The blocked Gram build and the path solver are optimisations, so every test here
compares them against the obvious slow computation. An optimisation that is not
checked against the thing it replaced is just a rewrite.
"""

import numpy as np
import pytest

from lairnet._linalg import GramCache, solve_ridge, solve_ridge_path


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    n, d, m = 200, 7, 25
    A = np.hstack([np.ones((n, 1)), rng.standard_normal((n, d))])
    H = np.tanh(rng.standard_normal((n, m)))
    y = A @ rng.standard_normal(A.shape[1]) + 0.1 * rng.standard_normal(n)
    return A, H, y


def test_gram_cache_matches_dense(problem):
    A, H, y = problem
    gram, rhs = GramCache(A, y).build(H)
    D = np.hstack([A, H])
    np.testing.assert_allclose(gram, D.T @ D, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(rhs.ravel(), D.T @ y, rtol=1e-10, atol=1e-10)


def test_gram_cache_reuse_across_layers(problem):
    """The cached A block must not drift as the hidden block changes."""
    A, H, y = problem
    cache = GramCache(A, y)
    rng = np.random.default_rng(1)
    for _ in range(4):
        Hk = np.tanh(rng.standard_normal(H.shape))
        gram, rhs = cache.build(Hk)
        D = np.hstack([A, Hk])
        np.testing.assert_allclose(gram, D.T @ D, rtol=1e-10, atol=1e-10)


def test_solve_ridge_matches_direct(problem):
    A, H, y = problem
    gram, rhs = GramCache(A, y).build(H)
    for ridge in (1e-6, 1e-2, 1.0, 100.0):
        got = solve_ridge(gram, rhs, ridge)
        want = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
        np.testing.assert_allclose(got.coef, want, rtol=1e-7, atol=1e-9)
        assert got.method == "cholesky"
        assert got.well_conditioned


def test_solve_ridge_path_matches_individual(problem):
    A, H, y = problem
    gram, rhs = GramCache(A, y).build(H)
    ridges = [1e-4, 1e-2, 1.0, 10.0]
    path = solve_ridge_path(gram, rhs, ridges)
    assert [s.ridge for s in path] == ridges
    for sol in path:
        one = solve_ridge(gram, rhs, sol.ridge)
        np.testing.assert_allclose(sol.coef, one.coef, rtol=1e-6, atol=1e-9)


def test_rank_deficient_still_solves():
    """Width above sample size makes the Gram matrix singular; the ridge is
    what keeps the problem well posed, and the solver must not need luck."""
    rng = np.random.default_rng(2)
    n, m = 20, 60
    A = np.ones((n, 1))
    H = rng.standard_normal((n, m))
    y = rng.standard_normal(n)
    gram, rhs = GramCache(A, y).build(H)
    assert np.linalg.matrix_rank(gram) < gram.shape[0]
    sol = solve_ridge(gram, rhs, 1e-3)
    assert np.all(np.isfinite(sol.coef))
    D = np.hstack([A, H])
    resid = D @ sol.coef.ravel() - y
    assert np.linalg.norm(resid) < np.linalg.norm(y)


def test_ill_conditioned_falls_back_to_eigh():
    """A Gram matrix spanning more than the condition limit must not be solved
    by Cholesky and claimed to be fine."""
    p = 30
    scales = np.logspace(0, -16, p)
    Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((p, p)))
    gram = Q @ np.diag(scales) @ Q.T
    gram = 0.5 * (gram + gram.T)
    rhs = np.ones((p, 1))
    sol = solve_ridge(gram, rhs, 1e-18)
    assert sol.method == "eigh"
    assert not sol.well_conditioned
    assert np.all(np.isfinite(sol.coef))


def test_multi_target(problem):
    A, H, y = problem
    Y = np.column_stack([y, 2.0 * y - 1.0])
    gram, rhs = GramCache(A, Y).build(H)
    sol = solve_ridge(gram, rhs, 1e-3)
    assert sol.coef.shape == (gram.shape[0], 2)
    want = np.linalg.solve(gram + 1e-3 * np.eye(gram.shape[0]), rhs)
    np.testing.assert_allclose(sol.coef, want, rtol=1e-7, atol=1e-9)


def test_rejects_bad_input(problem):
    A, H, y = problem
    with pytest.raises(ValueError, match="rows"):
        GramCache(A, y[:-1])
    cache = GramCache(A, y)
    with pytest.raises(ValueError, match="rows"):
        cache.build(H[:-1])
    gram, rhs = cache.build(H)
    with pytest.raises(ValueError, match="positive"):
        solve_ridge(gram, rhs, 0.0)
    with pytest.raises(ValueError, match="positive"):
        solve_ridge_path(gram, rhs, [1e-3, -1.0])
