Installation
============

Install from a local checkout during development:

.. code-block:: bash

   python -m pip install -e ".[dev,plots]"

The core package requires Python 3.9 or newer, NumPy, SciPy, and
scikit-learn 1.3 or newer. Compatibility code supports both the older
``Estimator._validate_data`` validation path and the newer
``sklearn.utils.validation.validate_data`` path.

Optional acceleration extras are split so users do not need heavy dependencies
unless they ask for them:

.. code-block:: bash

   python -m pip install -e ".[speed]"
   python -m pip install -e ".[torch]"

The ``speed`` extra installs numba. With ``anchor_backend="auto"``, LAIR-Net
uses NumPy below 5000 samples and numba at or above 5000 samples when numba is
available. Torch is optional and explicit; it is never selected automatically,
even when installed.

Documentation dependencies are separate:

.. code-block:: bash

   python -m pip install -e ".[docs]"
