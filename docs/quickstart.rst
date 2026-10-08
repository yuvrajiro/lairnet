Quick Start
===========

Regression follows the scikit-learn estimator workflow:

.. code-block:: python

   from lairnet import LAIRNetRegressor
   from sklearn.datasets import make_friedman1
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import r2_score

   X, y = make_friedman1(n_samples=600, noise=1.0, random_state=0)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, random_state=0
   )

   model = LAIRNetRegressor(
       n_layers=12,
       n_hidden=128,
       ridge=1e-2,
       random_state=0,
   ).fit(X_train, y_train)

   y_pred = model.predict(X_test)
   print(r2_score(y_test, y_pred))

Classification uses the same randomized stack with one-hot targets:

.. code-block:: python

   from lairnet import LAIRNetClassifier
   from sklearn.datasets import load_wine
   from sklearn.model_selection import cross_val_score

   X, y = load_wine(return_X_y=True)
   clf = LAIRNetClassifier(n_layers=8, n_hidden=96, random_state=0)
   print(cross_val_score(clf, X, y, cv=5).mean())

Search uses scikit-learn under the hood, with LAIR-specific defaults and
delegation to the selected model:

.. code-block:: python

   from lairnet import LAIRNetCV, LAIRNetRegressor

   search = LAIRNetCV(
       LAIRNetRegressor(n_hidden=96, random_state=0),
       param_grid={"align": [0.0, 0.5], "ridge": [1e-3, 1e-2]},
       cv=3,
   ).fit(X_train, y_train)

   print(search.best_params_)
   print(search.align_profile())

``align_profile`` reports the best score available at each alignment after
tuning the other searched parameters. It is a by-product of the search, not a
separate one-dimensional sweep.

Fitted estimators expose diagnostics that help users understand the fit:

.. code-block:: python

   model.anchor_converged_
   model.anchor_status_
   model.anchor_message_
   model.anchor_n_iter_
   model.anchor_n_func_evals_
   model.anchor_backend_
   model.layer_conditions_
   model.layer_scores_

Layerwise predictions are available for inspection:

.. code-block:: python

   depth_predictions = model.layer_predictions(X_test)
   lower, upper = model.prediction_band(X_test, coverage=0.9)

``prediction_band`` is a depth-disagreement band, not a calibrated statistical
confidence interval.

Backend policy
--------------

``anchor_backend="auto"`` chooses a conservative benchmark-backed default:
NumPy below 5000 samples and numba at or above 5000 samples when numba is
installed. The threshold is the smallest package-profile sample size where
numba was measured to clear the benchmark rule against NumPy. It is a
heuristic, not a guarantee.

Torch is never selected automatically. It can be faster on large CPU problems
when installed, but it is a heavy optional dependency and CUDA was slower than
CPU in the measured benchmark regimes.

Advanced anchor inspection
--------------------------

``AnchorNet`` is a public scikit-learn transformer for advanced diagnostics. A
fitted anchor exposes the target-aware hidden representation that LAIR-Net
aligns toward:

.. code-block:: python

   from lairnet import AnchorNet

   anchor = AnchorNet(n_hidden=64, backend="auto", random_state=0)
   H = anchor.fit_transform(X_train, y_train)

Most workflows should still start with ``LAIRNetRegressor``,
``LAIRNetClassifier`` or ``LAIRNetCV``.

Interpretability functions return structured objects:

.. code-block:: python

   from lairnet.explain import explain_model

   explanation = explain_model(
       model,
       X_train,
       y_train,
       X_test,
       y_test,
       n_repeats=5,
       random_state=0,
   )
   print(explanation.summary())

Plotting functions consume those objects. matplotlib is imported lazily, so
fitting remains independent of plotting dependencies:

.. code-block:: python

   from lairnet.plotting import plot_feature_importance, plot_depth_path

   ax = plot_feature_importance(explanation.importance)
   ax = plot_depth_path(explanation.depth)

Inspection scorers use the metric signature ``scorer(y_true, y_pred)``, like
``sklearn.metrics.r2_score`` or ``sklearn.metrics.accuracy_score``. They do not
accept scikit-learn scorer objects from ``sklearn.metrics.get_scorer``, which
use the estimator signature ``scorer(estimator, X, y)``.
