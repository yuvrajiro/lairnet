Anchor network
==============

The shallow network whose hidden activations are the anchor.

.. warning::

   This is an advanced and diagnostic component. Most users should enter through
   :doc:`estimators` or :doc:`model_selection`. Reach for ``AnchorNet`` directly
   when you want to know *why* a fit went the way it did — the anchor on its own
   is the quickest way to see whether LAIR-Net has anything to align to on your
   problem.

It is a scikit-learn transformer, so ``fit_transform`` gives the anchor and it
composes in a ``Pipeline``.

.. currentmodule:: lairnet

.. autoclass:: AnchorNet
   :members:
   :inherited-members: BaseEstimator

.. autofunction:: available_backends

.. autoclass:: ConvergenceWarning
