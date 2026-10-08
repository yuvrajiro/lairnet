Model selection
===============

Searching over LAIR-Net hyperparameters. A thin wrapper over scikit-learn's
search estimators that adds a default grid centred on ``align`` and delegates
the LAIR-specific methods to the selected model.

.. currentmodule:: lairnet.model_selection

.. autosummary::
   :nosignatures:

   LAIRNetCV
   default_grid

.. autoclass:: LAIRNetCV
   :members:

.. autofunction:: default_grid
