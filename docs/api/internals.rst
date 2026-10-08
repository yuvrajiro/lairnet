Internals
=========

The numerical core. Documented because the diagnostics the estimators expose
come from here, not because most users need to call it.

Readout solvers
---------------

The design matrix is ``[A, H]`` where ``A = [1, X]`` is fixed for the whole
stack, so the ``A`` block's Gram is computed once and reused at every depth.
Solves are Cholesky with an automatic eigendecomposition fallback and a reported
condition number.

.. currentmodule:: lairnet

.. autoclass:: GramCache
   :members:
.. autoclass:: RidgeSolution
   :members:
.. autofunction:: solve_ridge
.. autofunction:: solve_ridge_path

Forward pass
------------

.. autoclass:: LAIRCore
   :members:
.. autoclass:: LayerState
   :members:

.. py:data:: AGGREGATORS

   Aggregation rule name to callable, mapping an array of shape
   ``(n_layers, n_samples, n_outputs)`` down to ``(n_samples, n_outputs)``.
   The keys are the values ``aggregator`` accepts: ``"median"`` (the default),
   ``"mean"`` and ``"trimmed"``.

   Written as an explicit ``py:data`` rather than ``autodata`` because autodoc
   falls back to :class:`dict`'s own docstring for a re-exported dictionary,
   which is both wrong and not valid reStructuredText.
