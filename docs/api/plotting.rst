Plotting
========

Figures for the results produced by :doc:`inspection`. Every function takes a
result object or a fitted estimator, draws onto an axis, and returns that axis,
so figures compose.

matplotlib is imported lazily and is an optional extra:

.. code-block:: bash

   pip install lairnet[plots]

See the :doc:`../gallery` for every one of these rendered, with the code that
produced it.

.. currentmodule:: lairnet.plotting

.. autosummary::
   :nosignatures:

   plot_feature_importance
   plot_depth_path
   plot_layer_diagnostics
   plot_anchor_alignment
   plot_layer_agreement
   plot_prediction_band

.. autofunction:: plot_feature_importance
.. autofunction:: plot_depth_path
.. autofunction:: plot_layer_diagnostics
.. autofunction:: plot_anchor_alignment
.. autofunction:: plot_layer_agreement
.. autofunction:: plot_prediction_band
