<div align="center">

<img src="https://raw.githubusercontent.com/yuvrajiro/lairnet/main/docs/_static/logo.svg" width="90" alt="LAIR-Net">

# LAIR-Net

**A randomized network you can steer, and then see inside.**

[![PyPI](https://img.shields.io/pypi/v/lairnet.svg)](https://pypi.org/project/lairnet/)
[![Python](https://img.shields.io/pypi/pyversions/lairnet.svg)](https://pypi.org/project/lairnet/)
[![Docs](https://img.shields.io/badge/docs-lairnet.statml.in-1F8A82.svg)](https://lairnet.statml.in/)
[![Tests](https://github.com/yuvrajiro/lairnet/actions/workflows/tests.yml/badge.svg)](https://github.com/yuvrajiro/lairnet/actions/workflows/tests.yml)
[![Docs build](https://github.com/yuvrajiro/lairnet/actions/workflows/docs.yml/badge.svg)](https://github.com/yuvrajiro/lairnet/actions/workflows/docs.yml)

[Documentation](https://lairnet.statml.in/) ·
[Gallery](https://lairnet.statml.in/gallery.html) ·
[Notebooks](https://lairnet.statml.in/usage.html) ·
[API](https://lairnet.statml.in/api.html)

</div>

---

A shallow network fitted to your target supplies an **anchor**. A stack of fixed
random layers is run with every layer pulled part-way back towards it, and a
ridge readout is fitted at every depth. Nothing backpropagates through the
stack — so it trains in seconds, and it tells you what it did.

```bash
pip install lairnet
```

```python
from lairnet import LAIRNetRegressor

model = LAIRNetRegressor().fit(X_train, y_train)
model.predict(X_test)

# and then, unlike most randomized networks:
model.anchor_converged_      # did the anchor fit converge, per the optimiser?
model.layer_conditions_      # how well conditioned is each readout?
model.layer_predictions(X)   # what did every depth predict?
```

This package is not a frozen reproduction of the paper code. It keeps the LAIR
idea and adds cleaner solvers, explicit convergence diagnostics, a measured
backend policy, model-selection helpers, and a much larger user API.

## What you get beyond a fitted model

<img src="https://raw.githubusercontent.com/yuvrajiro/lairnet/main/docs/_static/gallery/feature_importance.png" width="100%" alt="permutation importance: the five informative Friedman-1 features separate from the five noise features, with any bar whose error crosses zero greyed out">

Every figure in the [gallery](https://lairnet.statml.in/gallery.html)
is produced by `python docs/make_gallery.py`, which calls the same public
functions documented here. Nothing is drawn by hand.

| | |
|---|---|
| **Which features it uses** | `inspection.permutation_importance` on held-out data. Bars whose error bar crosses zero are greyed out and labelled, because an importance not distinguished from noise should not be ranked confidently. |
| **Whether depth is helping** | `inspection.depth_curve` gives the score from aggregating the first *k* depths, for every *k*, in one forward pass rather than `n_layers` refits. |
| **What the anchor is worth** | `inspection.align_sensitivity` and `inspection.anchor_contribution` refit at `align=0` and compare on the same held-out split. A flat curve means the target-aware component is doing nothing on your data. |
| **How much the depths disagree** | `inspection.layer_agreement`. Near-uniform high correlation is expected, not wrong — the alignment impulse *makes* the layers correlated, which is why median aggregation is a robustness choice and not a variance-reduction one. |
| **Whether the numerics held** | `layer_conditions_` and `layer_solvers_`: a per-depth condition number, and which depths fell back from Cholesky to an eigendecomposition. |
| **All of it at once** | `explain.explain_model(...).summary()` — a readable report someone can paste straight into a write-up. |

```python
from lairnet.explain import explain_model

print(explain_model(model, X_train, y_train, X_test, y_test).summary())
```

```text
score on the evaluation data: 0.8369

features the model uses
  feature 3                +0.6931 +/- 0.0397
  feature 0                +0.4900 +/- 0.0644
  ...

depth
  best at depth 12 of 12
  aggregating all depths is worth +0.1069 over one
  depths correlate 0.942 on average, so aggregation recovers less variance
  than independent errors would give

anchor
  with anchor    0.8369
  without anchor 0.7472
  the anchor is worth +0.0897 here
```

Called without held-out data it still works, and says so — every score above the
note is measured on the training data.

## Quick start

```python
from lairnet import LAIRNetRegressor
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = make_friedman1(n_samples=600, noise=1.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = LAIRNetRegressor(
    n_layers=12,
    n_hidden=128,
    ridge=1e-2,
    anchor_backend="auto",
    random_state=0,
).fit(X_train, y_train)

print(r2_score(y_test, model.predict(X_test)))
```

Classification uses the same stack with a one-hot target representation:

```python
from lairnet import LAIRNetClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score

X, y = load_wine(return_X_y=True)
clf = LAIRNetClassifier(n_layers=8, n_hidden=96, random_state=0)
print(cross_val_score(clf, X, y, cv=5).mean())
```

Hyperparameter search goes through `LAIRNetCV`, a thin wrapper around the
scikit-learn search classes with LAIR-specific defaults and delegation:

```python
from lairnet import LAIRNetCV, LAIRNetRegressor

search = LAIRNetCV(
    LAIRNetRegressor(n_hidden=96, random_state=0),
    param_grid={"align": [0.0, 0.5], "ridge": [1e-3, 1e-2]},
    cv=3,
).fit(X_train, y_train)

print(search.best_params_)
print(search.align_profile())
```

Four executed notebooks walk through the rest:
[getting started](https://lairnet.statml.in/examples/01_getting_started.html),
[interpreting a fitted model](https://lairnet.statml.in/examples/02_interpreting_a_model.html),
[tuning and model selection](https://lairnet.statml.in/examples/03_model_selection.html),
and [backends and speed](https://lairnet.statml.in/examples/04_backends_and_speed.html).
They are committed with their outputs, so the numbers on those pages are what
the code printed.

## Speed

`anchor_backend="auto"` uses NumPy below 5000 samples and numba at or above
5000 when numba is installed. The threshold is read off a benchmark that runs at
the package's own settings, not at a tolerance the package never uses. Torch is
faster still on large CPU problems and is **never** selected automatically,
because it is a heavy dependency to acquire silently.

```bash
pip install lairnet[speed]    # numba
pip install lairnet[torch]    # torch
pip install lairnet[plots]    # matplotlib, for lairnet.plotting
```

Readouts are closed form: one Cholesky per depth on a Gram matrix whose fixed
`[1, X]` block is built once and reused for the whole stack, with an
eigendecomposition fallback and a reported condition number.

## Fitted attributes

```
anchor_converged_   anchor_status_    anchor_message_       anchor_n_iter_
anchor_loss_        anchor_backend_   anchor_hit_max_iter_  anchor_n_func_evals_
layer_scores_       layer_conditions_ layer_solvers_        layer_state_deltas_
layer_anchor_distances_               feature_names_in_
```

`feature_names_in_` follows the scikit-learn convention and is only set when you
fit with a DataFrame. `anchor_status_` and `anchor_message_` come from the
optimiser; the torch and sklearn paths report `-1` and say so rather than
inventing a status.

`anchor_converged_` is the optimiser's own verdict, not an inference from the
iteration count, and a capped anchor fit raises a `ConvergenceWarning` — a
capped fit is an unlabelled regularisation choice, and returning one silently is
the failure this package is built to avoid.

`layer_predictions` returns the per-depth predictions. `prediction_band` returns
a **depth-disagreement band, not a calibrated interval**: it carries no coverage
guarantee and will not widen for a sample outside the training range unless the
depths happen to disagree there.

## Public API

| Module | Names |
|---|---|
| `lairnet` | `LAIRNetRegressor`, `LAIRNetClassifier`, `LAIRNetCV`, `default_grid`, `AnchorNet` |
| `lairnet.inspection` | `permutation_importance`, `depth_curve`, `align_sensitivity`, `layer_agreement`, `anchor_contribution` |
| `lairnet.explain` | `explain_model` |
| `lairnet.plotting` | `plot_feature_importance`, `plot_depth_path`, `plot_layer_diagnostics`, `plot_anchor_alignment`, `plot_layer_agreement`, `plot_prediction_band` |
| `lairnet` (solvers) | `solve_ridge`, `solve_ridge_path`, `GramCache` |

`AnchorNet` is public as an advanced diagnostic component and works as a
scikit-learn transformer on its own; most users should start with
`LAIRNetRegressor`, `LAIRNetClassifier` or `LAIRNetCV`.

Interpretability functions return structured results rather than drawing, so
fitting never imports matplotlib. The plotting functions import it lazily and
consume those results:

```python
from lairnet.plotting import plot_feature_importance, plot_depth_path

explanation = explain_model(model, X_train, y_train, X_test, y_test)
plot_feature_importance(explanation.importance)
plot_depth_path(explanation.depth)
```

Inspection scorers use the metric signature `scorer(y_true, y_pred)`, like
`sklearn.metrics.r2_score`. They do not accept scikit-learn scorer objects from
`get_scorer`, which use `scorer(estimator, X, y)`.

## Development

```bash
python -m pip install -e ".[dev,plots,docs]"
python -m pytest
```

```bash
python docs/check_styles.py          # a class named in a page with no CSS rule
python docs/check_numbers_agree.py   # pages quoting numbers the notebooks did not produce
sphinx-build -b html -W --keep-going docs docs/_build/html
```

The figures and notebook outputs are not produced by the Sphinx build. Run
`python docs/make_gallery.py` and `python docs/make_notebooks.py` with the same
interpreter — `make_notebooks.py` pins the kernel to `sys.executable` and aborts
if it does not get it, because a different scipy takes L-BFGS down a different
path and the published numbers stop agreeing.

Benchmark rules, learned the hard way: report repeated trials rather than
one-off timings, prevent overlapping runs, compare time to a common objective
value when optimisers converge at different rates, and do not document a speed
claim until the benchmark records support it.

```bash
python -m build
twine check dist/*
```

## Citation

If you use LAIR-Net in academic work, please cite the paper:

> Goswami, R., Bhambu, A. and Karmakar, B. *LAIR-Net: Leaky Alignment-Impulse
> Residual Networks for Tabular Regression.* arXiv:2610.11538, 2026.

```bibtex
@misc{goswami2026lairnet,
  title  = {LAIR-Net: Leaky Alignment-Impulse Residual Networks for Tabular Regression},
  author = {Rahul Goswami and Aryan Bhambu and Bittu Karmakar},
  year   = {2026},
  eprint = {2610.11538},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  url    = {https://arxiv.org/abs/2610.11538}
}
```

