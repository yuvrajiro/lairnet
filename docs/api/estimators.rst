Estimators
==========

The two models. Both follow the scikit-learn estimator contract, so they work
inside :class:`~sklearn.pipeline.Pipeline`, ``cross_val_score`` and the search
estimators, and both expose what the stack learned in addition to what it
predicts.

.. currentmodule:: lairnet

.. autosummary::
   :nosignatures:

   LAIRNetRegressor
   LAIRNetClassifier

.. autoclass:: LAIRNetRegressor
   :members:
   :inherited-members: BaseEstimator

.. autoclass:: LAIRNetClassifier
   :members:
   :inherited-members: BaseEstimator
