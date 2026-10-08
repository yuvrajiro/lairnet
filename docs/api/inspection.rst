Inspection
==========

Tools that answer questions a generic explainer cannot: which features the
model uses, how much the anchor contributes, whether depth is helping, and how
much the depths disagree.

Everything here returns arrays and dataclasses. Nothing draws — the functions in
:doc:`plotting` consume these results, which keeps matplotlib out of the fitting
path.

.. note::

   Scorers here take ``(y_true, y_pred)``, like :func:`sklearn.metrics.r2_score`.
   They are **not** scikit-learn scorer objects, which take
   ``(estimator, X, y)``. Passing one raises a ``TypeError`` naming the fix.

.. currentmodule:: lairnet.inspection

.. autosummary::
   :nosignatures:

   permutation_importance
   depth_curve
   align_sensitivity
   layer_agreement
   anchor_contribution

.. autofunction:: permutation_importance
.. autofunction:: depth_curve
.. autofunction:: align_sensitivity
.. autofunction:: layer_agreement
.. autofunction:: anchor_contribution

Result objects
--------------

.. autoclass:: ImportanceResult
   :members:
.. autoclass:: DepthCurve
   :members:
.. autoclass:: AlignSensitivity
   :members:
.. autoclass:: LayerAgreement
   :members:
