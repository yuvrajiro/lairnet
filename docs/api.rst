API reference
=============

.. _metadata_routing:

.. note::

   Some inherited estimator methods mention scikit-learn metadata routing.
   LAIR-Net follows scikit-learn's estimator protocol for those hooks; most
   users never touch them.

.. toctree::
   :maxdepth: 1

   api/estimators
   api/model_selection
   api/inspection
   api/explain
   api/plotting
   api/anchor
   api/internals

.. grid:: 1 2 2 3
   :gutter: 2
   :class-container: model-cards

   .. grid-item-card:: Estimators
      :link: api/estimators
      :link-type: doc

      ``LAIRNetRegressor``, ``LAIRNetClassifier``

   .. grid-item-card:: Model selection
      :link: api/model_selection
      :link-type: doc

      ``LAIRNetCV``, ``default_grid``

   .. grid-item-card:: Inspection
      :link: api/inspection
      :link-type: doc

      importance, depth, alignment, agreement

   .. grid-item-card:: Explain
      :link: api/explain
      :link-type: doc

      ``explain_model``, ``ModelExplanation``

   .. grid-item-card:: Plotting
      :link: api/plotting
      :link-type: doc

      six figures, optional extra

   .. grid-item-card:: Anchor network
      :link: api/anchor
      :link-type: doc

      ``AnchorNet``, backends

   .. grid-item-card:: Internals
      :link: api/internals
      :link-type: doc

      solvers, forward pass
