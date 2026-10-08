"""What the anchor backends do and do not guarantee about each other.

The module docstring of ``_anchor.py`` used to claim that the backends are
"interchangeable up to optimiser tolerance", and cited a test file that did not
exist. Both halves were wrong: the file was never written, and the claim is
false. The objective is non-convex and the backends do not share an
initialisation -- sklearn seeds itself -- so they reach different local optima.
Measured, they land on final objectives that differ by 25% or more.

So this file asserts the weaker true statements instead:

* numpy and numba implement the same gradient and start from the same point, so
  they must agree closely;
* every backend must reach a *comparable* objective, which catches a backend
  that is optimising the wrong thing without pretending they converge to the
  same place;
* the ``auto`` policy must resolve as documented.
"""

import numpy as np
import pytest

from lairnet._anchor import AnchorNet, available_backends

BACKENDS = available_backends()


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 6))
    y = np.tanh(X @ rng.standard_normal(6)) + 0.1 * rng.standard_normal(400)
    return X, y


def _fit(data, backend, **kw):
    X, y = data
    return AnchorNet(n_hidden=30, backend=backend, random_state=0,
                     max_iter=800, **kw).fit(X, y)


@pytest.mark.skipif("numba" not in BACKENDS, reason="numba not installed")
def test_numba_kernel_matches_the_numpy_objective(data):
    """The real equivalence claim, made where it actually holds.

    Comparing the two backends' *final* losses does not test the kernel: the
    problem is non-convex and ``fastmath`` reassociates arithmetic, so identical
    gradients still walk to different optima. The first version of this test
    asserted final-loss agreement and failed for that reason, on correct code.

    The gradient at a fixed point is the thing that must match, and it is what a
    wrong kernel would break -- a wrong gradient still converges, just to
    somewhere else, which no end-to-end comparison would catch.
    """
    from lairnet._anchor import _init, _objective, ACTIVATIONS
    from lairnet._anchor_numba import make_objective

    X, y = data
    Y = y[:, None]
    d, m, t = X.shape[1], 30, 1
    theta = _init(d, m, t, np.random.default_rng(0))

    loss_np, grad_np = _objective(theta, X, Y, d, m, t, 1e-4,
                                  ACTIVATIONS["tanh"])
    loss_nb, grad_nb = make_objective(X, Y, d, m, t, 1e-4, "tanh")(theta)

    assert loss_nb == pytest.approx(loss_np, rel=1e-12)
    np.testing.assert_allclose(grad_nb, grad_np, rtol=1e-10, atol=1e-14)


@pytest.mark.skipif("numba" not in BACKENDS, reason="numba not installed")
def test_numpy_and_numba_land_close_enough_to_be_the_same_model(data):
    """Same objective and same start, so the optima should be near each other
    even though floating-point reassociation stops them being identical."""
    a = _fit(data, "numpy")
    b = _fit(data, "numba")
    assert b.loss_ == pytest.approx(a.loss_, rel=0.1)


def test_every_backend_reaches_a_comparable_objective(data):
    """A backend that optimises the wrong thing lands somewhere else entirely.

    The tolerance is deliberately loose. These are different optimisers from
    different starts on a non-convex problem, so requiring agreement would be
    requiring something untrue; requiring the same order of magnitude catches a
    genuinely broken backend.
    """
    losses = {}
    for backend in BACKENDS:
        model = _fit(data, backend)
        assert np.isfinite(model.loss_)
        losses[backend] = model.loss_
    best = min(losses.values())
    for backend, loss in losses.items():
        assert loss < 10 * best, f"{backend} reached {loss:g} against {best:g}"
    # This test found a real defect: torch.optim.LBFGS.step returns the loss
    # from the FIRST closure evaluation, so recording its return value stored
    # the pre-fit objective. The fit worked and every diagnostic reported the
    # starting value, which made a working backend look 388x worse than the
    # others. Nothing else in the suite would have noticed.


def test_backends_are_not_claimed_to_be_identical(data):
    """The honest counterpart to the test above.

    If this ever starts failing because every backend agrees exactly, the
    docstring in ``_anchor.py`` should be revisited -- but it will not, and
    asserting the difference exists keeps the documentation from drifting back
    to the stronger claim.
    """
    if "sklearn" not in BACKENDS:
        pytest.skip("sklearn not installed")
    ours = _fit(data, "numpy")
    theirs = _fit(data, "sklearn")
    assert ours.W_.shape == theirs.W_.shape
    # Same shapes, different optima. Not a defect; the reason the equivalence
    # claim above is stated the way it is.
    assert not np.allclose(ours.W_, theirs.W_)


def test_auto_resolves_as_documented(data):
    """The threshold in the docstring and the behaviour must be the same."""
    model = AnchorNet(n_hidden=10)
    threshold = AnchorNet._NUMBA_THRESHOLD
    assert model._resolve_backend(threshold - 1) == "numpy"
    expected = "numba" if "numba" in BACKENDS else "numpy"
    assert model._resolve_backend(threshold) == expected
    assert model._resolve_backend(10 * threshold) == expected


def test_auto_never_chooses_torch(data):
    """torch is 4x faster at 5000 samples and still must not be automatic.

    A default that depends on what else happens to be installed makes the same
    code behave differently in two environments.
    """
    model = AnchorNet(n_hidden=10)
    for n in (10, 1000, 5000, 100000):
        assert model._resolve_backend(n) != "torch"


def test_unavailable_backend_is_rejected_clearly(data):
    with pytest.raises(ValueError, match="not available"):
        _fit(data, "definitely-not-a-backend")


@pytest.mark.skipif("torch" not in BACKENDS, reason="torch not installed")
def test_torch_backend_runs_and_reports_its_limits(data):
    """torch reports no convergence status, and must say so rather than guess."""
    model = _fit(data, "torch")
    assert np.isfinite(model.loss_)
    assert model.status_ == -1
    assert "no convergence status" in model.message_
    assert model.n_func_evals_ >= model.n_iter_


# ------------------------------------------------------------------ callback
HOOKED = [b for b in BACKENDS if b != "sklearn"]


@pytest.mark.parametrize("backend", HOOKED)
def test_callback_traces_and_ends_where_loss_does(data, backend):
    """The benchmark's protection against measuring a parallel implementation.

    ``benchmarks/bench_anchor.py`` times the shipped fitter through this hook
    rather than reimplementing it. That only works if the hook sees the real
    trajectory and the last thing it sees is where the optimiser finished --
    which is exactly what the torch backend got wrong, reporting the *first*
    closure evaluation as the final loss. This asserts both halves.
    """
    X, y = data
    seen = []
    model = AnchorNet(n_hidden=20, backend=backend, max_iter=100,
                      random_state=0, callback=seen.append).fit(X, y)
    assert len(seen) > 1, f"{backend} produced no trajectory"
    assert seen[-1] == pytest.approx(model.loss_, rel=1e-9)
    assert seen[-1] < seen[0], "the fit did not reduce the objective"


@pytest.mark.skipif("sklearn" not in BACKENDS, reason="sklearn not installed")
def test_sklearn_has_no_hook_and_does_not_pretend_to(data):
    """MLPRegressor exposes no per-iteration hook. Silently emitting one point
    and calling it a trajectory would be worse than emitting nothing."""
    X, y = data
    seen = []
    AnchorNet(n_hidden=20, backend="sklearn", max_iter=100, random_state=0,
              callback=seen.append).fit(X, y)
    assert seen == []


def test_callback_still_reaches_the_caller_after_cloning(data):
    """A cloned estimator's callback still writes to the caller's object.

    Worth pinning because the obvious identity assertion is misleading:
    ``clone(est).callback is seen.append`` is **False**, since CPython builds a
    fresh bound-method object on every attribute access. I read that as clone
    having detached the hook and nearly documented a warning about it. It had
    not -- ``copy.deepcopy`` treats a bound method atomically and keeps
    ``__self__``, so the clone appends to the same list.

    So the test asserts the behaviour that matters, which is where the data
    ends up, rather than object identity, which says nothing here.
    """
    from sklearn.base import clone

    X, y = data
    seen = []
    original = AnchorNet(n_hidden=15, backend="numpy", max_iter=50,
                         random_state=0, callback=seen.append)
    clone(original).fit(X, y)
    assert len(seen) > 1, "clone detached the callback from the caller"


def test_no_callback_is_the_default_and_costs_nothing(data):
    X, y = data
    model = AnchorNet(n_hidden=15, backend="numpy", max_iter=50,
                      random_state=0).fit(X, y)
    assert model.callback is None
    assert np.isfinite(model.loss_)
