#!/usr/bin/env python
"""Build the usage notebooks by executing them, then commit them with outputs.

``docs/conf.py`` sets ``nb_execution_mode = "off"``, so Sphinx renders stored
outputs rather than running anything at build time. That keeps the docs build
fast and offline, and it means a notebook in the repo shows what the code
actually did rather than what it is expected to do -- but only if it was
executed here first. This script is what does that.

    python docs/make_notebooks.py

Every cell is real: the numbers, the figures and the warnings in the rendered
pages come from running the package.

The cells run in *this* interpreter, not in whatever the ambient ``python3``
kernelspec points at. That is not a detail. The first run of this script used
the installed kernelspec, which on this machine was a different Python with
numpy 2.1 and scipy 1.13 rather than numpy 1.23 and scipy 1.11; L-BFGS took a
different path and the notebooks reported an R^2 of 0.8476 for the fit that
``make_gallery.py`` documents as 0.8369. Both numbers were honest and the docs
showed them side by side.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
OUT = HERE / "examples"
KERNEL = "lairnet-docs"


@contextmanager
def kernel_for_this_interpreter():
    """Yield the name of a throwaway kernelspec that runs ``sys.executable``.

    Written to a temporary directory and exposed through ``JUPYTER_PATH`` so
    nothing is installed into the user's Jupyter configuration.
    """
    with tempfile.TemporaryDirectory() as root:
        spec = Path(root) / "kernels" / KERNEL
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher",
                     "-f", "{connection_file}"],
            "display_name": "Python 3",
            "language": "python",
        }), encoding="utf-8")
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = root
        try:
            yield KERNEL
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous


def check_kernel_is_this_interpreter(kernel: str) -> None:
    """Run one cell and refuse to continue unless it is the right Python.

    The kernelspec above is only a request. If it is not picked up the fallback
    is silent -- notebooks execute fine against another environment and produce
    numbers that quietly disagree with the rest of the documentation.
    """
    probe = nbformat.v4.new_notebook()
    probe.cells = [nbformat.v4.new_code_cell(
        "import sys, numpy, scipy\n"
        "print(sys.executable)\nprint(numpy.__version__)\n"
        "print(scipy.__version__)")]
    NotebookClient(probe, kernel_name=kernel,
                   resources={"metadata": {"path": str(HERE.parent)}}).execute()
    text = "".join(o.get("text", "") for o in probe.cells[0].outputs)
    got = text.strip().splitlines()
    import numpy
    import scipy
    want = [sys.executable, numpy.__version__, scipy.__version__]
    if got != want:
        raise SystemExit(
            "the notebook kernel is not this interpreter, so the notebooks "
            "would disagree with the gallery and the tests:\n"
            f"    kernel : {got}\n"
            f"    wanted : {want}\n"
            "Install ipykernel into this interpreter "
            f"({sys.executable} -m pip install ipykernel).")
    print(f"  kernel is {got[0]}  (numpy {got[1]}, scipy {got[2]})")


def nb(*cells) -> nbformat.NotebookNode:
    book = nbformat.v4.new_notebook()
    book.cells = list(cells)
    book.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    return book


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


# ------------------------------------------------------------------ 01 basics
BASICS = nb(
    md("""
# Getting started

Fit a regressor, predict, and read the diagnostics the model exposes beyond its
predictions. Everything below runs on `make_friedman1`, whose answer is known:
the response uses `x0`–`x4` and ignores `x5`–`x9`.
"""),
    code("""
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from lairnet import LAIRNetRegressor

X, y = make_friedman1(n_samples=600, noise=1.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

model = LAIRNetRegressor(n_hidden=60, n_layers=12, random_state=0)
model.fit(X_train, y_train)

r2_score(y_test, model.predict(X_test))
"""),
    md("""
## What the fit tells you about itself

A randomized network usually gives you a number and nothing else. These
attributes are the reason the package exists.
"""),
    code("""
print("anchor backend :", model.anchor_backend_)
print("converged      :", model.anchor_converged_)
print("iterations     :", model.anchor_n_iter_)
print("objective      :", f"{model.anchor_loss_:.6g}")
print("optimiser says :", model.anchor_message_)
"""),
    md("""
`anchor_converged_` is the optimiser's own verdict, not a guess from the
iteration count. If the cap binds, `fit` raises a `ConvergenceWarning` — a
capped fit is an unlabelled regularisation choice, which is exactly what the
package is built to avoid.
"""),
    code("""
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    LAIRNetRegressor(n_hidden=30, n_layers=4, anchor_max_iter=20,
                     random_state=0).fit(X_train, y_train)
    print(caught[0].message)
"""),
    md("""
## Per-depth numerics

One readout is fitted at every depth. These say how well conditioned each one
was and which solver it needed.
"""),
    code("""
import numpy as np

print("condition numbers :", np.array2string(model.layer_conditions_,
                                             precision=1, max_line_width=90))
print("solvers           :", set(model.layer_solvers_))
"""),
    md("""
Any depth reporting `eigh` is the solver saying it did not trust the Cholesky
path there.

## Predictions from every depth

The stack produces one prediction per depth. `predict` aggregates them; the raw
set is what the interpretability tools consume.
"""),
    code("""
P = model.layer_predictions(X_test)
P.shape          # (n_layers, n_samples, n_outputs)
"""),
    code("""
lower, upper = model.prediction_band(X_test, coverage=0.9)
float(np.mean(upper - lower))
"""),
    md("""
```{warning}
That band is **depth disagreement, not calibrated uncertainty**. It says how
much the depths of this one model disagree, carries no coverage guarantee, and
will not widen for a sample outside the training range unless the depths happen
to disagree there.
```
"""),
)

# ------------------------------------------------------------ 02 interpreting
INTERPRET = nb(
    md("""
# Interpreting a fitted model

Which features it uses, whether depth is helping, and what the anchor is
actually worth on your data.
"""),
    code("""
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

from lairnet import LAIRNetRegressor
from lairnet import inspection as ins

X, y = make_friedman1(n_samples=600, noise=1.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
model = LAIRNetRegressor(n_hidden=60, n_layers=12, random_state=0)
model.fit(X_train, y_train)
"""),
    md("""
## Feature importance

Applied to the fitted model on held-out data, so it measures what this model
uses rather than what the data contains.
"""),
    code("""
importance = ins.permutation_importance(model, X_test, y_test,
                                        n_repeats=10, random_state=0)
for name, mean, std in importance.top(6):
    flag = "" if mean - std > 0 else "   (within noise)"
    print(f"{name:<12} {mean:+.4f} +/- {std:.4f}{flag}")
"""),
    md("""
`x0`–`x4` are the informative ones, which is the right answer for this problem.

## Is depth helping?

Costs one forward pass, not `n_layers` refits.
"""),
    code("""
curve = ins.depth_curve(model, X_test, y_test)
print("best depth :", curve.best_depth, "of", len(curve.depths))
print("one depth  :", f"{curve.scores[0]:.4f}")
print("all depths :", f"{curve.scores[-1]:.4f}")
"""),
    md("""
## How much do the depths disagree?
"""),
    code("""
agreement = ins.layer_agreement(model, X_test)
print("mean correlation between depths:", f"{agreement.mean_correlation:.3f}")
"""),
    md("""
Near-uniform high correlation is expected rather than wrong: the alignment
impulse pulls every layer towards the same anchor, so it *makes* them
correlated. That is why the median aggregator is a robustness choice and not a
variance-reduction one.

## What is the anchor worth?

Refits at `align=0`, which removes the target-aware component entirely, and
scores both on the same held-out data.
"""),
    code("""
ins.anchor_contribution(model, X_train, y_train, X_test, y_test)
"""),
    md("""
Training and evaluation data are separate required arguments. An earlier
signature took one pair and scored both arms on it, which gave the ablation an
in-sample score and **reversed the conclusion**.
"""),
    code("""
sweep = ins.align_sensitivity(model, X_train, y_train,
                              X_valid=X_test, y_valid=y_test)
for a, s in zip(sweep.align_values, sweep.scores):
    print(f"align={a:<5} {s:.4f}")
print("best:", sweep.best_align, " gain over no anchor:",
      f"{sweep.anchor_gain:+.4f}")
"""),
    md("""
## Everything at once
"""),
    code("""
from lairnet.explain import explain_model

report = explain_model(model, X_train, y_train, X_test, y_test,
                       n_repeats=10, align_values=(0.0, 0.25, 0.5, 0.75),
                       random_state=0)
print(report.summary())
"""),
)

# ------------------------------------------------------- 03 model selection
SELECTION = nb(
    md("""
# Tuning and model selection

`align` decides how much the model relies on the anchor, and its useful value
depends on the data. It is the parameter worth tuning first.
"""),
    code("""
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

from lairnet import LAIRNetRegressor
from lairnet.model_selection import LAIRNetCV, default_grid

X, y = make_friedman1(n_samples=500, noise=1.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

default_grid()
"""),
    code("""
search = LAIRNetCV(
    LAIRNetRegressor(n_hidden=40, n_layers=8, random_state=0),
    param_grid={"align": [0.0, 0.25, 0.5, 0.75], "ridge": [1e-3, 1e-1]},
    cv=3,
).fit(X_train, y_train)

print("best params :", search.best_params_)
print("cv score    :", f"{search.best_score_:.4f}")
print("test score  :", f"{search.score(X_test, y_test):.4f}")
"""),
    md("""
## The alignment profile falls out for free

The best score achievable at each `align`, after tuning the other searched
parameters. No second sweep needed.
"""),
    code("""
values, scores = search.align_profile()
for a, s in zip(values, scores):
    print(f"align={a:<5} {s:.4f}")
"""),
    md("""
## The search result is usable directly

Prediction and the LAIR-specific inspection methods delegate to the selected
model, so you do not have to reach into `best_estimator_`.
"""),
    code("""
print("layer predictions :", search.layer_predictions(X_test).shape)
print("anchor converged  :", search.anchor_converged_)
print("worst conditioning:", f"{search.layer_conditions_.max():.3g}")
"""),
    md("""
## Classification works the same way
"""),
    code("""
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lairnet import LAIRNetClassifier

Xc, yc = load_wine(return_X_y=True)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.3, random_state=0, stratify=yc)

clf = make_pipeline(
    StandardScaler(),
    LAIRNetClassifier(n_hidden=40, n_layers=6, random_state=0),
).fit(Xc_tr, yc_tr)

print("accuracy :", f"{clf.score(Xc_te, yc_te):.4f}")
print("classes  :", clf[-1].classes_)
"""),
    md("""
```{note}
`predict_proba` is a softmax over scores fitted by least squares, not a
likelihood. The ordering is meaningful; calibrate explicitly if you need the
magnitudes to be.
```
"""),
)

# ------------------------------------------------------------- 04 backends
BACKENDS = nb(
    md("""
# Backends and speed

The anchor is fitted by L-BFGS. Four implementations are available and
`backend="auto"` chooses between two of them on a measured threshold.
"""),
    code("""
from lairnet import available_backends

available_backends()
"""),
    md("""
## What `auto` does

NumPy below 5000 samples, numba at or above when it is importable. torch is
never chosen automatically.
"""),
    code("""
from lairnet import AnchorNet

probe = AnchorNet(n_hidden=10)
for n in (500, 4999, 5000, 50_000):
    print(f"n={n:<7} -> {probe._resolve_backend(n)}")
"""),
    md("""
The threshold is the smallest size where numba was **measured** to win in the
regime the package actually runs, not an interpolated crossover — see
`benchmarks/RESULTS.md`.

torch is faster still at larger sample sizes and deliberately not automatic: it
is a multi-gigabyte dependency, and a default that changes with whatever else is
installed is not a default. Ask for it explicitly if you want it.

```{warning}
Do not reach for `device="cuda"` on problems of this shape. CUDA was slower than
the CPU in every regime measured, because transfers dominate an optimisation
this small.
```
"""),
    code("""
import time

from sklearn.datasets import make_friedman1

X, y = make_friedman1(n_samples=1500, random_state=0)

for backend in ("numpy", "sklearn"):
    start = time.perf_counter()
    fit = AnchorNet(n_hidden=50, backend=backend, max_iter=400,
                    random_state=0).fit(X, y)
    print(f"{backend:<9} {time.perf_counter() - start:6.2f}s   "
          f"objective {fit.loss_:.6g}   converged {fit.converged_}")
"""),
    md("""
## The anchor on its own

`AnchorNet` is a scikit-learn transformer, so `fit_transform` gives you the
target-aware representation directly. It is the quickest way to see whether
LAIR-Net has anything to align to on your problem.
"""),
    code("""
anchor = AnchorNet(n_hidden=20, max_iter=400, random_state=0)
anchor.fit_transform(X, y).shape
"""),
    code("""
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

make_pipeline(
    AnchorNet(n_hidden=20, max_iter=400, random_state=0), Ridge()
).fit(X, y).score(X, y)
"""),
)


def build(name: str, book, kernel: str) -> None:
    path = OUT / name
    # Execute with the repository root as the working directory so the kernel
    # imports the package from the checkout: ipykernel puts the working
    # directory on sys.path. Without this the kernel starts in docs/examples and
    # every notebook dies on ``import lairnet`` unless the package happens to be
    # installed in the environment running the docs build.
    client = NotebookClient(book, timeout=900, kernel_name=kernel,
                            resources={"metadata": {"path": str(HERE.parent)}})
    client.execute()
    nbformat.write(book, path)
    print(f"  executed and wrote {path.relative_to(HERE.parent)}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with kernel_for_this_interpreter() as kernel:
        check_kernel_is_this_interpreter(kernel)
        for name, book in (
            ("01_getting_started.ipynb", BASICS),
            ("02_interpreting_a_model.ipynb", INTERPRET),
            ("03_model_selection.ipynb", SELECTION),
            ("04_backends_and_speed.ipynb", BACKENDS),
        ):
            build(name, book, kernel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
