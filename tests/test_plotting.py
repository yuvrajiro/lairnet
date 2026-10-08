"""Smoke tests for the plotting functions.

These do not check what a figure looks like. They check that every function
builds axes without error, and -- the point RA2 raised -- that matplotlib is
never imported by fitting. A plotting dependency that creeps into the fit path
turns an optional extra into a hard requirement without anyone noticing.
"""

import subprocess
import sys

import matplotlib
import pytest
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lairnet import LAIRNetRegressor
from lairnet.inspection import (
    align_sensitivity,
    depth_curve,
    layer_agreement,
    permutation_importance,
)
from lairnet.plotting import (
    plot_anchor_alignment,
    plot_depth_path,
    plot_feature_importance,
    plot_layer_agreement,
    plot_layer_diagnostics,
    plot_prediction_band,
)


@pytest.fixture(scope="module")
def artefacts():
    X, y = make_friedman1(n_samples=200, noise=1.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LAIRNetRegressor(n_hidden=25, n_layers=5,
                             random_state=0).fit(Xtr, ytr)
    return {
        "model": model, "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
        "importance": permutation_importance(model, Xte, yte, n_repeats=3,
                                             random_state=0),
        "depth": depth_curve(model, Xte, yte),
        "agreement": layer_agreement(model, Xte),
        "alignment": align_sensitivity(model, Xtr, ytr, X_valid=Xte,
                                       y_valid=yte, align_values=(0.0, 0.5)),
    }


def _check(ax):
    assert ax is not None
    assert ax.figure is not None
    # House rule: figures carry no title, because the caption carries it and
    # two copies of the same sentence can disagree.
    assert ax.get_title() == ""
    plt.close(ax.figure)


def test_plot_feature_importance(artefacts):
    _check(plot_feature_importance(artefacts["importance"]))


def test_plot_depth_path(artefacts):
    _check(plot_depth_path(artefacts["depth"]))


def test_plot_layer_diagnostics(artefacts):
    _check(plot_layer_diagnostics(artefacts["model"]))


def test_plot_anchor_alignment(artefacts):
    _check(plot_anchor_alignment(artefacts["alignment"]))


def test_plot_layer_agreement(artefacts):
    _check(plot_layer_agreement(artefacts["agreement"]))


def test_plot_prediction_band(artefacts):
    _check(plot_prediction_band(artefacts["model"], artefacts["Xte"],
                                artefacts["yte"]))


def test_plots_accept_a_supplied_axis(artefacts):
    """Every function draws onto a given axis, so figures compose."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    a = plot_depth_path(artefacts["depth"], ax=axes[0])
    b = plot_feature_importance(artefacts["importance"], ax=axes[1])
    assert a is axes[0] and b is axes[1]
    plt.close(fig)


def test_fitting_does_not_import_matplotlib():
    """Fit in a fresh interpreter and assert matplotlib never loaded.

    Run as a subprocess because this test module imports matplotlib itself, so
    checking ``sys.modules`` in-process would always pass.
    """
    code = (
        "import sys\n"
        "from sklearn.datasets import make_friedman1\n"
        "from lairnet import LAIRNetRegressor\n"
        "X, y = make_friedman1(n_samples=80, random_state=0)\n"
        "LAIRNetRegressor(n_hidden=10, n_layers=3, random_state=0)"
        ".fit(X, y).predict(X)\n"
        "assert 'matplotlib' not in sys.modules, "
        "'fitting imported matplotlib'\n"
        "print('clean')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout
