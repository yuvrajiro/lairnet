"""Tests for the inspection tools.

The important one here is ``test_anchor_contribution_agrees_with_align_sweep``.
An earlier ``anchor_contribution(estimator, X, y)`` refitted the ablation on the
data it then scored on, so handed a test set it produced an in-sample score for
one arm and an out-of-sample score for the other. That did not add noise, it
**reversed the conclusion** -- the anchor read as hurting by 0.09 when a proper
split showed it helping by 0.15. The bug was found because two tools that had
to agree did not, so the test is written the same way.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_friedman1
from sklearn.metrics import get_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split

from lairnet import LAIRNetClassifier, LAIRNetRegressor
from lairnet.inspection import (
    align_sensitivity,
    anchor_contribution,
    depth_curve,
    layer_agreement,
    permutation_importance,
)


@pytest.fixture(scope="module")
def fitted():
    X, y = make_friedman1(n_samples=300, noise=1.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LAIRNetRegressor(n_hidden=40, n_layers=8,
                             random_state=0).fit(Xtr, ytr)
    return model, Xtr, ytr, Xte, yte


def test_permutation_importance_finds_the_informative_features(fitted):
    """Friedman1 uses x0..x4 and ignores x5..x9."""
    model, _, _, Xte, yte = fitted
    result = permutation_importance(model, Xte, yte, n_repeats=5,
                                    random_state=0)
    assert result.importances.shape == (Xte.shape[1], 5)
    informative = set(result.ranking()[:5].tolist())
    assert len(informative & {0, 1, 2, 3, 4}) >= 4
    # The unused features should not register as strongly as the used ones.
    assert result.importances_mean[:5].mean() > result.importances_mean[5:].mean()


def test_permutation_importance_rejects_sklearn_scorer(fitted):
    """A scikit-learn scorer takes (estimator, X, y), not (y_true, y_pred).

    Without the guard this fails deep inside with a shape error that says
    nothing about the real mistake.
    """
    model, _, _, Xte, yte = fitted
    with pytest.raises(TypeError, match="y_true, y_pred"):
        permutation_importance(model, Xte, yte, scorer=get_scorer("r2"),
                               n_repeats=2)


def test_custom_scorer_is_used(fitted):
    model, _, _, Xte, yte = fitted
    result = permutation_importance(
        model, Xte, yte, n_repeats=3, random_state=0,
        scorer=lambda a, b: -mean_absolute_error(a, b))
    assert np.isfinite(result.baseline_score)
    assert result.baseline_score < 0  # negated error


def test_anchor_contribution_agrees_with_align_sweep(fitted):
    """The regression test for the in-sample reversal bug.

    ``anchor_contribution`` and ``align_sensitivity`` compute the same two
    quantities by different routes. They must agree exactly.
    """
    model, Xtr, ytr, Xte, yte = fitted
    contribution = anchor_contribution(model, Xtr, ytr, Xte, yte)
    sweep = align_sensitivity(model, Xtr, ytr, X_valid=Xte, y_valid=yte,
                              align_values=(0.0, model.align))

    assert contribution["without_anchor"] == pytest.approx(sweep.scores[0])
    assert contribution["with_anchor"] == pytest.approx(sweep.scores[1])
    assert contribution["gain"] == pytest.approx(
        sweep.scores[1] - sweep.scores[0])


def test_anchor_contribution_requires_four_arguments(fitted):
    """The three-argument form is what produced the reversal. It must not
    silently work again."""
    model, Xtr, ytr, _, _ = fitted
    with pytest.raises(TypeError):
        anchor_contribution(model, Xtr, ytr)


def test_depth_curve_shapes_and_monotone_domain(fitted):
    model, _, _, Xte, yte = fitted
    curve = depth_curve(model, Xte, yte)
    assert curve.depths.tolist() == list(range(1, model.n_layers + 1))
    assert curve.scores.shape == curve.per_layer_scores.shape
    assert 1 <= curve.best_depth <= model.n_layers
    # The one-layer aggregate is that layer alone, by definition.
    assert curve.scores[0] == pytest.approx(curve.per_layer_scores[0])


def test_layer_agreement_correlation_is_well_formed(fitted):
    model, _, _, Xte, _ = fitted
    agreement = layer_agreement(model, Xte)
    C = agreement.correlation
    assert C.shape == (model.n_layers, model.n_layers)
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-9)
    np.testing.assert_allclose(C, C.T, atol=1e-9)
    assert -1.0 <= agreement.mean_correlation <= 1.0
    assert agreement.per_sample_std.shape[0] == Xte.shape[0]


def test_align_zero_removes_the_anchor(fitted):
    """At align=0 the anchor cannot influence the prediction, so changing the
    anchor's regularisation must change nothing."""
    _model, Xtr, ytr, Xte, _ = fitted
    a = LAIRNetRegressor(n_hidden=40, n_layers=8, align=0.0,
                         anchor_alpha=1e-6, random_state=0).fit(Xtr, ytr)
    b = LAIRNetRegressor(n_hidden=40, n_layers=8, align=0.0,
                         anchor_alpha=10.0, random_state=0).fit(Xtr, ytr)
    np.testing.assert_allclose(a.predict(Xte), b.predict(Xte), rtol=1e-10)


def test_inspection_works_for_classifiers():
    X, y = make_classification(n_samples=250, n_features=10, n_informative=5,
                               n_classes=3, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)
    model = LAIRNetClassifier(n_hidden=30, n_layers=5,
                              random_state=0).fit(Xtr, ytr)
    imp = permutation_importance(model, Xte, yte, n_repeats=3, random_state=0)
    assert 0.0 <= imp.baseline_score <= 1.0
    curve = depth_curve(model, Xte, yte)
    assert np.all((curve.scores >= 0.0) & (curve.scores <= 1.0))
