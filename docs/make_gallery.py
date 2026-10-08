#!/usr/bin/env python
"""Render the documentation gallery from the package itself.

Every image in the docs is produced by running the code the docs describe, on
data the reader can regenerate in three lines. Nothing is drawn by hand and
nothing is stylised for the page, so a figure that stops looking like this is a
figure whose function changed.

    python docs/make_gallery.py

Writes PNGs into ``docs/_static/gallery/`` and prints the explain summary that
``gallery.rst`` quotes, so the prose and the output cannot drift apart.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import make_friedman1  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from lairnet import LAIRNetRegressor  # noqa: E402
from lairnet import inspection as ins  # noqa: E402
from lairnet import plotting as plo  # noqa: E402
from lairnet.explain import explain_model  # noqa: E402

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent / "_static" / "gallery"
DPI = 130


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    X, y = make_friedman1(n_samples=600, noise=1.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LAIRNetRegressor(n_hidden=60, n_layers=12,
                             random_state=0).fit(Xtr, ytr)

    importance = ins.permutation_importance(model, Xte, yte, n_repeats=10,
                                            random_state=0)
    depth = ins.depth_curve(model, Xte, yte)
    agreement = ins.layer_agreement(model, Xte)
    alignment = ins.align_sensitivity(
        model, Xtr, ytr, X_valid=Xte, y_valid=yte,
        align_values=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9))

    figures = {
        "feature_importance": lambda: plo.plot_feature_importance(importance),
        "depth_path": lambda: plo.plot_depth_path(depth),
        "layer_diagnostics": lambda: plo.plot_layer_diagnostics(model),
        "anchor_alignment": lambda: plo.plot_anchor_alignment(alignment),
        "layer_agreement": lambda: plo.plot_layer_agreement(agreement),
        "prediction_band": lambda: plo.plot_prediction_band(model, Xte, yte),
    }
    for name, build in figures.items():
        ax = build()
        path = OUT / f"{name}.png"
        # Opaque, not transparent. lairnet.plotting hardcodes light-background
        # ink -- #444444 ticks, #333333 rules, white marker faces -- and
        # layer_agreement.png is 100% dark ink, so a transparent PNG needs
        # something behind it. pydata-sphinx-theme does supply that, via
        # `html[data-theme="dark"] .bd-content img { background-color: white }`,
        # so the documentation was never broken. GitHub does not: the README
        # embeds these, and on GitHub's dark theme a transparent version is
        # near-black ink on a near-black page. Baking the background in makes
        # them correct in both places and independent of a theme rule this
        # package does not control.
        ax.figure.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.22,
                          facecolor="white", transparent=False)
        plt.close(ax.figure)
        print(f"  wrote {path.relative_to(OUT.parents[2])}")

    print("\n--- explain_model summary, quoted verbatim in gallery.rst ---")
    print(explain_model(model, Xtr, ytr, Xte, yte, n_repeats=10,
                        align_values=(0.0, 0.25, 0.5, 0.75),
                        random_state=0).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
