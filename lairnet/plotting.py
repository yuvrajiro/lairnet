"""Plots for the results produced by :mod:`lairnet.inspection`.

Every function takes a result object or a fitted estimator, draws onto an axis,
and returns that axis, so figures compose and nothing here decides your layout.
matplotlib is imported lazily, so it is an optional extra and never a
requirement for fitting.

    pip install lairnet[plots]

Style follows one rule: no titles. A title inside a figure is redundant with the
caption in whatever document the figure lands in, and duplicated text is text
that can disagree with itself.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "plot_feature_importance",
    "plot_depth_path",
    "plot_layer_diagnostics",
    "plot_anchor_alignment",
    "plot_layer_agreement",
    "plot_prediction_band",
]

#: Colour-blind safe, ordered so adjacent series stay distinguishable in print.
PALETTE = ("#440154", "#31688E", "#1F8A82", "#35B779", "#D97706", "#9B2226")


def _axes(ax, figsize):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "plotting requires matplotlib; install it with "
            "`pip install lairnet[plots]`") from exc
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#888888")
    ax.tick_params(colors="#444444", labelsize=9)
    ax.grid(True, color="#DDDDDD", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    return ax


def plot_feature_importance(result, *, top_k: int = 15, ax=None,
                            figsize=(7.0, 4.5)):
    """Horizontal bars of permutation importance, most important at the top.

    Bars carry the standard deviation across repeats, because an importance
    whose error bar crosses zero has not been distinguished from noise and the
    plot should say so rather than rank it confidently.
    """
    ax = _style(_axes(ax, figsize))
    order = result.ranking()[:top_k][::-1]
    names = [result.feature_names[i] if result.feature_names else f"x{i}"
             for i in order]
    means = result.importances_mean[order]
    errs = result.importances_std[order]
    colours = [PALETTE[1] if m - e > 0 else "#BBBBBB"
               for m, e in zip(means, errs)]
    ax.barh(np.arange(len(order)), means, xerr=errs, color=colours,
            edgecolor="white", linewidth=0.6,
            error_kw={"ecolor": "#666666", "elinewidth": 1.0})
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("drop in score when shuffled", fontsize=10)
    ax.text(0.99, 0.02, "grey: error bar crosses zero", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color="#666666")
    return ax


def plot_depth_path(curve, *, ax=None, figsize=(7.0, 4.2)):
    """Score against depth, aggregated and per layer.

    The gap between the two lines is what aggregation is buying. If the
    aggregated line stops rising well before the last depth, the extra layers
    are not paying for themselves.
    """
    ax = _style(_axes(ax, figsize))
    ax.plot(curve.depths, curve.scores, color=PALETTE[0], linewidth=2.0,
            marker="o", markersize=5, markerfacecolor="white",
            markeredgewidth=1.6, markeredgecolor=PALETTE[0],
            label="aggregated over the first k depths")
    ax.plot(curve.depths, curve.per_layer_scores, color=PALETTE[4],
            linewidth=1.4, linestyle=(0, (4, 3)), marker="s", markersize=4,
            label="that depth alone")
    best = curve.best_depth
    ax.axvline(best, color="#999999", linewidth=1.0, linestyle=":")
    ax.annotate(f"best at depth {best}", xy=(best, curve.scores.max()),
                xytext=(4, -12), textcoords="offset points", fontsize=9,
                color="#444444")
    ax.set_xlabel("depth", fontsize=10)
    ax.set_ylabel("score", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False, fontsize=9)
    return ax


def plot_layer_diagnostics(estimator, *, ax=None, figsize=(7.5, 4.4)):
    """Per-depth conditioning, state movement and anchor distance.

    Three quantities on two axes. A condition number climbing with depth means
    the readouts are becoming sensitive; a state delta collapsing to zero means
    the stack has stopped moving and the remaining depths are duplicates.
    """
    ax = _style(_axes(ax, figsize))
    depths = np.arange(1, len(estimator.layer_conditions_) + 1)
    ax.semilogy(depths, estimator.layer_conditions_, color=PALETTE[0],
                linewidth=2.0, marker="o", markersize=4,
                label="readout condition number")
    eigh = [i for i, s in enumerate(estimator.layer_solvers_) if s != "cholesky"]
    if eigh:
        ax.scatter(depths[eigh], estimator.layer_conditions_[eigh], s=70,
                   facecolor="none", edgecolor="#9B2226", linewidth=1.6,
                   zorder=5, label="fell back to eigh")
    ax.set_xlabel("depth", fontsize=10)
    ax.set_ylabel("condition number", fontsize=10)

    twin = ax.twinx()
    twin.spines["top"].set_visible(False)
    twin.plot(depths, estimator.layer_state_deltas_, color=PALETTE[3],
              linewidth=1.6, linestyle=(0, (4, 3)), label="state change")
    twin.plot(depths, estimator.layer_anchor_distances_, color=PALETTE[4],
              linewidth=1.6, linestyle=(0, (1, 2)), label="distance to anchor")
    twin.set_ylabel("mean absolute value", fontsize=10)
    twin.tick_params(colors="#444444", labelsize=9)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, frameon=False, fontsize=9)
    return ax


def plot_anchor_alignment(sensitivity, *, ax=None, figsize=(7.0, 4.2)):
    """Score against the alignment coefficient, with the fitted value marked.

    The height of the curve above its value at zero is what the anchor is
    worth. A flat curve means the target-aware component is doing nothing here.
    """
    ax = _style(_axes(ax, figsize))
    ax.plot(sensitivity.align_values, sensitivity.scores, color=PALETTE[2],
            linewidth=2.0, marker="o", markersize=5, markerfacecolor="white",
            markeredgewidth=1.6, markeredgecolor=PALETTE[2])
    if (sensitivity.align_values == 0.0).any():
        zero = sensitivity.scores[sensitivity.align_values == 0.0][0]
        ax.axhline(zero, color="#999999", linewidth=1.0, linestyle=":")
        ax.text(0.99, zero, " no anchor", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9, color="#666666")
    ax.axvline(sensitivity.fitted_align, color=PALETTE[5], linewidth=1.2,
               linestyle=(0, (4, 3)))
    ax.annotate("fitted", xy=(sensitivity.fitted_align, sensitivity.scores.min()),
                xytext=(4, 4), textcoords="offset points", fontsize=9,
                color=PALETTE[5])
    ax.set_xlabel(r"alignment coefficient", fontsize=10)
    ax.set_ylabel("score", fontsize=10)
    return ax


def plot_layer_agreement(agreement, *, ax=None, figsize=(5.4, 4.6)):
    """Correlation between depths, as a matrix.

    Near-uniform high correlation means the depths are near-duplicates, so
    aggregating them recovers much less variance than independence would give.
    """
    ax = _axes(ax, figsize)
    C = agreement.correlation
    im = ax.imshow(C, cmap="viridis", vmin=min(0.0, C.min()), vmax=1.0)
    ax.set_xlabel("depth", fontsize=10)
    ax.set_ylabel("depth", fontsize=10)
    ax.set_xticks(np.arange(C.shape[0]))
    ax.set_yticks(np.arange(C.shape[0]))
    ax.tick_params(labelsize=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("correlation between layer predictions", fontsize=9)
    ax.text(0.5, -0.18, f"mean off-diagonal {agreement.mean_correlation:.3f}",
            transform=ax.transAxes, ha="center", fontsize=9, color="#444444")
    return ax


def plot_prediction_band(estimator, X, y=None, *, coverage: float = 0.9,
                         max_points: int = 200, ax=None, figsize=(7.2, 4.4)):
    """Prediction with the spread across depths, sorted by prediction.

    .. warning::

       The band is **depth disagreement, not calibrated uncertainty**. It shows
       how much the depths of this one model disagree, carries no coverage
       guarantee, and will not widen for a sample far outside the training data
       unless the depths happen to disagree there.
    """
    ax = _style(_axes(ax, figsize))
    pred = np.asarray(estimator.predict(X)).ravel()
    lo, hi = estimator.prediction_band(X, coverage=coverage)
    lo, hi = np.asarray(lo).ravel(), np.asarray(hi).ravel()
    order = np.argsort(pred)[:max_points]
    idx = np.arange(len(order))
    ax.fill_between(idx, lo[order], hi[order], color=PALETTE[1], alpha=0.22,
                    linewidth=0, label=f"{int(coverage * 100)}% of depths")
    ax.plot(idx, pred[order], color=PALETTE[1], linewidth=1.8,
            label="aggregated prediction")
    if y is not None:
        ax.scatter(idx, np.asarray(y).ravel()[order], s=12, color="#333333",
                   alpha=0.6, zorder=4, label="observed")
    ax.set_xlabel("samples, ordered by prediction", fontsize=10)
    ax.set_ylabel("value", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              frameon=False, fontsize=9)
    ax.text(0.01, 0.97, "band is depth disagreement, not calibrated uncertainty",
            transform=ax.transAxes, va="top", fontsize=8.5, color="#9B2226")
    return ax
