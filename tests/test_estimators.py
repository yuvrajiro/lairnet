"""Tests for the public estimators.

Includes regression tests for two defects found in review:

* held-out data passed to ``fit(validation=...)`` was not coerced the way the
  training data was, so a DataFrame reached the core in a different form;
* the classifier's held-out labels were not one-hot encoded, so its per-depth
  validation score compared a label vector against an indicator matrix and
  produced a plausible-looking wrong number rather than an error.
"""

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.datasets import load_wine, make_classification, make_friedman1
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lairnet import LAIRNetClassifier, LAIRNetRegressor
from lairnet._anchor import ConvergenceWarning

pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def regression():
    X, y = make_friedman1(n_samples=300, noise=1.0, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.fixture(scope="module")
def classification():
    X, y = make_classification(n_samples=300, n_features=10, n_informative=5,
                               n_classes=3, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# ---------------------------------------------------------------- contract
def test_sklearn_contract(regression):
    Xtr, Xte, ytr, yte = regression
    est = LAIRNetRegressor(n_hidden=25, n_layers=4, random_state=0)
    assert is_regressor(est) and not is_classifier(est)
    assert "align" in est.get_params()
    clone(est)  # must not raise
    est.fit(Xtr, ytr)
    assert est.n_features_in_ == Xtr.shape[1]
    assert est.predict(Xte).shape == yte.shape


def test_works_in_pipeline_and_search(regression):
    Xtr, _, ytr, _ = regression
    est = LAIRNetRegressor(n_hidden=20, n_layers=3, random_state=0)
    pipe = make_pipeline(StandardScaler(), est)
    assert np.isfinite(cross_val_score(pipe, Xtr, ytr, cv=3)).all()
    search = GridSearchCV(est, {"align": [0.0, 0.5]}, cv=3).fit(Xtr, ytr)
    assert search.best_params_["align"] in (0.0, 0.5)


def test_predictions_are_reproducible(regression):
    Xtr, Xte, ytr, _ = regression
    kw = {"n_hidden": 20, "n_layers": 4, "random_state": 7}
    a = LAIRNetRegressor(**kw).fit(Xtr, ytr).predict(Xte)
    b = LAIRNetRegressor(**kw).fit(Xtr, ytr).predict(Xte)
    np.testing.assert_allclose(a, b, rtol=1e-12)


# -------------------------------------------------------------- validation
def test_dataframe_validation_data_is_coerced(regression):
    """Regression test: held-out X must go through the same validation as the
    training X, or a DataFrame reaches the core in a different form."""
    Xtr, Xte, ytr, yte = regression
    cols = [f"f{i}" for i in range(Xtr.shape[1])]
    est = LAIRNetRegressor(n_hidden=20, n_layers=4, random_state=0).fit(
        pd.DataFrame(Xtr, columns=cols), ytr,
        validation=(pd.DataFrame(Xte, columns=cols), yte))
    assert list(est.feature_names_in_) == cols
    assert est.layer_valid_scores_.shape == (4,)
    assert np.isfinite(est.layer_valid_scores_).all()


def test_classifier_validation_labels_are_encoded(classification):
    """Regression test: held-out labels must be one-hot encoded like the
    training targets, or the per-depth score compares a label vector against
    an indicator matrix and still returns a number."""
    Xtr, Xte, ytr, yte = classification
    est = LAIRNetClassifier(n_hidden=25, n_layers=4, random_state=0).fit(
        Xtr, ytr, validation=(Xte, yte))
    scores = est.layer_valid_scores_
    assert scores.shape == (4,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    # A stratified split of a learnable problem should beat the majority class.
    assert scores.max() > 1.0 / len(est.classes_)


def test_malformed_validation_is_rejected(regression):
    Xtr, Xte, ytr, _ = regression
    est = LAIRNetRegressor(n_hidden=15, n_layers=3, random_state=0)
    with pytest.raises(ValueError, match="tuple"):
        est.fit(Xtr, ytr, validation=Xte)
    with pytest.raises(ValueError, match="rows"):
        est.fit(Xtr, ytr, validation=(Xte, ytr))


# ------------------------------------------------------------- diagnostics
def test_diagnostics_are_populated(regression):
    Xtr, _, ytr, _ = regression
    est = LAIRNetRegressor(n_hidden=25, n_layers=6, random_state=0).fit(Xtr, ytr)
    assert est.layer_conditions_.shape == (6,)
    assert len(est.layer_solvers_) == 6
    assert est.layer_state_deltas_.shape == (6,)
    assert est.layer_anchor_distances_.shape == (6,)
    assert isinstance(est.anchor_converged_, bool)
    assert isinstance(est.anchor_message_, str)
    assert est.anchor_n_func_evals_ >= est.anchor_n_iter_
    assert est.anchor_backend_ in {"numpy", "numba", "torch", "sklearn"}


def test_capped_anchor_warns(regression):
    """A capped fit is not a converged one and must not pass silently."""
    Xtr, _, ytr, _ = regression
    with pytest.warns(ConvergenceWarning, match="max_iter"):
        est = LAIRNetRegressor(n_hidden=20, n_layers=3, anchor_max_iter=5,
                               random_state=0).fit(Xtr, ytr)
    assert est.anchor_hit_max_iter_
    assert not est.anchor_converged_


def test_prediction_band_brackets_the_prediction(regression):
    Xtr, Xte, ytr, _ = regression
    est = LAIRNetRegressor(n_hidden=25, n_layers=8, random_state=0).fit(Xtr, ytr)
    lo, hi = est.prediction_band(Xte, coverage=0.8)
    pred = est.predict(Xte).reshape(lo.shape)
    assert np.all(lo <= hi)
    # The median of the depths lies inside their own central 80%.
    assert np.mean((pred >= lo) & (pred <= hi)) > 0.95


def test_layer_predictions_aggregate_to_predict(regression):
    Xtr, Xte, ytr, _ = regression
    est = LAIRNetRegressor(n_hidden=25, n_layers=6, aggregator="median",
                           random_state=0).fit(Xtr, ytr)
    P = est.layer_predictions(Xte)
    np.testing.assert_allclose(np.median(P, axis=0).ravel(), est.predict(Xte),
                               rtol=1e-10)


# --------------------------------------------------------------- classifier
def test_classifier_basics(classification):
    Xtr, Xte, ytr, yte = classification
    est = LAIRNetClassifier(n_hidden=25, n_layers=5, random_state=0).fit(Xtr, ytr)
    assert list(est.classes_) == sorted(set(ytr))
    proba = est.predict_proba(Xte)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    assert set(est.predict(Xte)) <= set(est.classes_)
    assert est.score(Xte, yte) > 1.0 / len(est.classes_)


def test_binary_classification():
    X, y = make_classification(n_samples=200, n_features=8, n_classes=2,
                               random_state=0)
    est = LAIRNetClassifier(n_hidden=20, n_layers=4, random_state=0).fit(X, y)
    assert est.predict_proba(X).shape == (200, 2)
    assert est.score(X, y) > 0.6


def test_classifier_rejects_single_class():
    X = np.random.default_rng(0).standard_normal((40, 4))
    with pytest.raises(ValueError, match="2 classes"):
        LAIRNetClassifier(n_hidden=10, n_layers=2).fit(X, np.zeros(40))


def test_wine_accuracy_is_reasonable():
    """A sanity floor on a dataset where any competent model does well."""
    X, y = load_wine(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)
    pipe = make_pipeline(StandardScaler(),
                         LAIRNetClassifier(n_hidden=40, n_layers=6,
                                           random_state=0)).fit(Xtr, ytr)
    assert pipe.score(Xte, yte) > 0.90


# ------------------------------------------------------------- persistence
def test_save_and_load_round_trip(regression, tmp_path):
    Xtr, Xte, ytr, _ = regression
    est = LAIRNetRegressor(n_hidden=20, n_layers=4, random_state=0).fit(Xtr, ytr)
    path = tmp_path / "model.joblib"
    est.save(str(path))
    np.testing.assert_allclose(LAIRNetRegressor.load(str(path)).predict(Xte),
                               est.predict(Xte), rtol=1e-12)


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("params, match", [
    ({"leak": 1.5}, "leak"),
    ({"align": -0.1}, "align"),
    ({"ridge": 0.0}, "ridge"),
    ({"n_layers": 0}, "n_layers"),
    ({"activation": "nope"}, "activation"),
    ({"aggregator": "nope"}, "aggregator"),
])
def test_bad_parameters_are_rejected(regression, params, match):
    Xtr, _, ytr, _ = regression
    kwargs = {"n_hidden": 10, "n_layers": 2, **params}
    with pytest.raises(ValueError, match=match):
        LAIRNetRegressor(**kwargs).fit(Xtr, ytr)
