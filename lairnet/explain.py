"""One call that runs the inspection suite and reports what it found.

:func:`explain_model` bundles the tools in :mod:`lairnet.inspection` into a
single result object and a readable summary. It is a convenience layer, not a
new method: everything it reports is available piecewise from
:mod:`lairnet.inspection`, and anything expensive is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .inspection import (
    AlignSensitivity,
    DepthCurve,
    ImportanceResult,
    LayerAgreement,
    align_sensitivity,
    anchor_contribution,
    depth_curve,
    layer_agreement,
    permutation_importance,
)

__all__ = ["ModelExplanation", "explain_model"]


@dataclass
class ModelExplanation:
    """Everything :func:`explain_model` computed."""

    score: float
    importance: ImportanceResult
    depth: DepthCurve
    agreement: LayerAgreement
    anchor: dict
    alignment: Optional[AlignSensitivity] = None
    layer_conditions: np.ndarray = field(default_factory=lambda: np.array([]))
    layer_solvers: list = field(default_factory=list)
    anchor_converged: bool = True
    anchor_message: str = ""
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        """A short report, written to be read rather than parsed."""
        lines = [f"score on the evaluation data: {self.score:.4f}", ""]

        lines.append("features the model uses")
        for name, mean, std in self.importance.top(5):
            marker = " " if mean - std > 0 else "  (within noise)"
            lines.append(f"  {name:<24} {mean:+.4f} +/- {std:.4f}{marker}")

        lines += ["", "depth"]
        lines.append(f"  best at depth {self.depth.best_depth} of "
                     f"{len(self.depth.depths)}")
        gain = self.depth.scores[-1] - self.depth.scores[0]
        lines.append(f"  aggregating all depths is worth {gain:+.4f} over one")
        lines.append(f"  depths correlate {self.agreement.mean_correlation:.3f} "
                     f"on average, so aggregation recovers less variance than "
                     f"independent errors would give")

        lines += ["", "anchor"]
        lines.append(f"  with anchor    {self.anchor['with_anchor']:.4f}")
        lines.append(f"  without anchor {self.anchor['without_anchor']:.4f}")
        lines.append(f"  the anchor is worth {self.anchor['gain']:+.4f} here")
        if self.alignment is not None:
            lines.append(f"  best alignment coefficient tested: "
                         f"{self.alignment.best_align:g} "
                         f"(fitted at {self.alignment.fitted_align:g})")
        if not self.anchor_converged:
            lines.append(f"  WARNING anchor did not converge: "
                         f"{self.anchor_message}")

        lines += ["", "numerics"]
        lines.append(f"  readout condition numbers "
                     f"{self.layer_conditions.min():.3g} to "
                     f"{self.layer_conditions.max():.3g}")
        fallbacks = sum(1 for s in self.layer_solvers if s != "cholesky")
        if fallbacks:
            lines.append(f"  WARNING {fallbacks} depth(s) fell back to eigh, "
                         f"so Cholesky was not trusted there")

        for note in self.notes:
            lines += ["", note]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


def explain_model(
    estimator,
    X_train,
    y_train,
    X_eval=None,
    y_eval=None,
    *,
    n_repeats: int = 10,
    align_values: Optional[Sequence[float]] = None,
    random_state: Optional[int] = None,
) -> ModelExplanation:
    """Run the inspection suite on a fitted estimator.

    Parameters
    ----------
    estimator : fitted LAIRNetRegressor or LAIRNetClassifier
    X_train, y_train : array-like
        The data the estimator was fitted on. Needed because the anchor
        ablation is refitted, and it must be refitted on the same data or the
        comparison is not between two models of the same problem.
    X_eval, y_eval : array-like, optional
        Held-out data for scoring. Defaults to the training data, with a note
        recorded in the result saying so, because every number here is then an
        in-sample one and should be read as such.
    n_repeats : int, default=10
        Shuffles per feature for the permutation importance.
    align_values : sequence of float, optional
        If given, the alignment coefficient is swept over these values. This
        **refits one model per value**, so it is off by default.
    """
    notes = []
    if X_eval is None or y_eval is None:
        X_eval, y_eval = X_train, y_train
        notes.append(
            "NOTE no held-out data was given, so every score above is measured "
            "on the training data and reads better than the model will "
            "generalise. Pass X_eval and y_eval for honest numbers.")

    importance = permutation_importance(
        estimator, X_eval, y_eval, n_repeats=n_repeats,
        random_state=random_state)
    depth = depth_curve(estimator, X_eval, y_eval)
    agreement = layer_agreement(estimator, X_eval)
    anchor = anchor_contribution(estimator, X_train, y_train, X_eval, y_eval)

    alignment = None
    if align_values is not None:
        alignment = align_sensitivity(
            estimator, X_train, y_train, align_values=align_values,
            X_valid=X_eval, y_valid=y_eval)

    return ModelExplanation(
        score=importance.baseline_score,
        importance=importance,
        depth=depth,
        agreement=agreement,
        anchor=anchor,
        alignment=alignment,
        layer_conditions=np.asarray(estimator.layer_conditions_),
        layer_solvers=list(estimator.layer_solvers_),
        anchor_converged=bool(estimator.anchor_converged_),
        anchor_message=str(estimator.anchor_message_),
        notes=notes,
    )
