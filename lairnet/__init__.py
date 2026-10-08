"""LAIR-Net: leaky alignment-impulse residual randomized networks.

A shallow network fitted to the target supplies an *anchor*; a stack of fixed
random layers is then run with every layer pulled part-way back towards it, and
a ridge readout is fitted at every depth. Only the anchor and the readouts are
trained, so there is no backpropagation through the stack.

    >>> from lairnet import LAIRNetRegressor
    >>> from sklearn.datasets import make_friedman1
    >>> X, y = make_friedman1(n_samples=300, random_state=0)
    >>> model = LAIRNetRegressor(n_layers=10, random_state=0).fit(X, y)
    >>> model.predict(X[:5]).shape
    (5,)

The estimators follow the scikit-learn contract and additionally expose what
the stack learned -- the prediction from every depth, the conditioning of every
readout, and the anchor optimiser's own verdict on its fit -- for the tools in
:mod:`lairnet.explain` and :mod:`lairnet.plotting`.
"""

from ._anchor import AnchorNet, ConvergenceWarning, available_backends
from ._core import AGGREGATORS, LAIRCore, LayerState
from ._linalg import GramCache, RidgeSolution, solve_ridge, solve_ridge_path
from .estimators import LAIRNetClassifier, LAIRNetRegressor
from .model_selection import LAIRNetCV, default_grid

__all__ = [
    # estimators
    "LAIRNetRegressor",
    "LAIRNetClassifier",
    # search
    "LAIRNetCV",
    "default_grid",
    # components
    "AnchorNet",
    "LAIRCore",
    "LayerState",
    "AGGREGATORS",
    # readout solvers
    "GramCache",
    "RidgeSolution",
    "solve_ridge",
    "solve_ridge_path",
    # utilities
    "available_backends",
    "ConvergenceWarning",
]

__version__ = "0.1.0"
