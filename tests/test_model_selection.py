"""Tests for LAIRNetCV."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_friedman1
from sklearn.model_selection import KFold, train_test_split

from lairnet import LAIRNetClassifier, LAIRNetRegressor
from lairnet.model_selection import LAIRNetCV, default_grid


@pytest.fixture(scope="module")
def data():
    X, y = make_friedman1(n_samples=250, noise=1.0, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


def test_search_selects_and_refits(data):
    Xtr, Xte, ytr, yte = data
    base = LAIRNetRegressor(n_hidden=20, n_layers=4, random_state=0)
    search = LAIRNetCV(base, {"align": [0.0, 0.5]}, cv=3).fit(Xtr, ytr)
    assert search.best_params_["align"] in (0.0, 0.5)
    assert search.best_estimator_.align == search.best_params_["align"]
    assert search.n_splits_ == 3
    assert 0 <= search.best_index_ < len(search.cv_results_["params"])
    assert np.isfinite(search.score(Xte, yte))


def test_delegates_lair_specific_methods(data):
    Xtr, Xte, ytr, _ = data
    base = LAIRNetRegressor(n_hidden=20, n_layers=5, random_state=0)
    search = LAIRNetCV(base, {"align": [0.25, 0.5]}, cv=3).fit(Xtr, ytr)
    assert search.layer_predictions(Xte).shape[0] == 5
    lo, hi = search.prediction_band(Xte)
    assert np.all(lo <= hi)
    # Fitted diagnostics forward from the selected model.
    assert search.layer_conditions_.shape == (5,)
    assert isinstance(search.anchor_converged_, bool)


def test_attribute_forwarding_does_not_swallow_typos(data):
    """Only trailing-underscore names forward. A mistyped parameter must still
    raise, or a typo silently resolves to something unrelated."""
    Xtr, _, ytr, _ = data
    search = LAIRNetCV(LAIRNetRegressor(n_hidden=15, n_layers=3,
                                        random_state=0),
                       {"align": [0.5]}, cv=3).fit(Xtr, ytr)
    with pytest.raises(AttributeError):
        _ = search.aligment          # typo, no trailing underscore
    with pytest.raises(AttributeError):
        _ = search.not_a_real_thing_  # trailing underscore, still absent


def test_align_profile_reads_off_the_search(data):
    Xtr, _, ytr, _ = data
    base = LAIRNetRegressor(n_hidden=20, n_layers=4, random_state=0)
    values = [0.0, 0.25, 0.5]
    search = LAIRNetCV(base, {"align": values}, cv=3).fit(Xtr, ytr)
    got, scores = search.align_profile()
    np.testing.assert_allclose(got, values)
    assert scores.shape == (3,)
    assert scores.max() == pytest.approx(search.best_score_)


def test_align_profile_is_none_when_align_not_searched(data):
    Xtr, _, ytr, _ = data
    base = LAIRNetRegressor(n_hidden=15, n_layers=3, random_state=0)
    search = LAIRNetCV(base, {"ridge": [1e-3, 1e-2]}, cv=3).fit(Xtr, ytr)
    assert search.align_profile() is None


def test_random_search_and_custom_splitter(data):
    Xtr, _, ytr, _ = data
    base = LAIRNetRegressor(n_hidden=15, n_layers=3, random_state=0)
    search = LAIRNetCV(base, {"align": [0.0, 0.25, 0.5, 0.75]},
                       search="random", n_iter=3,
                       cv=KFold(3, shuffle=True, random_state=0),
                       random_state=0).fit(Xtr, ytr)
    assert len(search.cv_results_["params"]) == 3


def test_rejects_unknown_search_kind(data):
    Xtr, _, ytr, _ = data
    with pytest.raises(ValueError, match="grid.*random"):
        LAIRNetCV(search="bogus", param_grid={"align": [0.5]}).fit(Xtr, ytr)


def test_classifier_search_keeps_classes():
    X, y = make_classification(n_samples=200, n_features=8, n_informative=4,
                               n_classes=3, random_state=0)
    base = LAIRNetClassifier(n_hidden=20, n_layers=3, random_state=0)
    search = LAIRNetCV(base, {"align": [0.25, 0.5]}, cv=3).fit(X, y)
    assert list(search.classes_) == [0, 1, 2]
    proba = search.predict_proba(X[:5])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    assert set(search.predict(X)) <= set(search.classes_)


def test_default_grid_shape():
    narrow = default_grid()
    assert set(narrow) == {"align", "ridge", "anchor_alpha"}
    assert 0.0 in narrow["align"]  # the no-anchor case must be reachable
    wide = default_grid(wide=True)
    assert "n_layers" in wide and "aggregator" in wide


def test_explain_through_the_search(data):
    Xtr, Xte, ytr, yte = data
    base = LAIRNetRegressor(n_hidden=20, n_layers=4, random_state=0)
    search = LAIRNetCV(base, {"align": [0.25, 0.5]}, cv=3).fit(Xtr, ytr)
    result = search.explain(Xtr, ytr, Xte, yte, n_repeats=3, random_state=0)
    assert "with_anchor" in result.anchor
    assert isinstance(result.summary(), str)


def test_refit_false_explains_itself(data):
    """``refit=False`` must not report "not fitted", which is false and points
    the reader at the wrong thing."""
    Xtr, Xte, ytr, _ = data
    base = LAIRNetRegressor(n_hidden=15, n_layers=3, random_state=0)
    search = LAIRNetCV(base, {"align": [0.0, 0.5]}, cv=3,
                       refit=False).fit(Xtr, ytr)
    assert search.best_params_["align"] in (0.0, 0.5)
    assert np.isfinite(search.best_score_)
    with pytest.raises(AttributeError, match="refit=True"):
        search.predict(Xte)
    with pytest.raises(AttributeError, match="refit=True"):
        search.layer_predictions(Xte)


def test_align_profile_reports_the_best_not_the_mean(data):
    """With ridge searched too, each align value has several candidates and
    the profile reports the best of them."""
    Xtr, _, ytr, _ = data
    base = LAIRNetRegressor(n_hidden=15, n_layers=3, random_state=0)
    search = LAIRNetCV(base, {"align": [0.0, 0.5],
                              "ridge": [1e-3, 1e-1]}, cv=3).fit(Xtr, ytr)
    values, scores = search.align_profile()
    results = search.cv_results_
    params = np.array([p["align"] for p in results["params"]], dtype=float)
    means = np.asarray(results["mean_test_score"], dtype=float)
    for v, s in zip(values, scores):
        assert s == pytest.approx(np.nanmax(means[params == v]))


def test_top_level_exports():
    import lairnet

    assert "LAIRNetCV" in lairnet.__all__
    assert "default_grid" in lairnet.__all__
    assert lairnet.LAIRNetCV is LAIRNetCV
