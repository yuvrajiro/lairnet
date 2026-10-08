"""scikit-learn version compatibility.

Input validation moved between scikit-learn versions. Up to 1.5 an estimator
called ``self._validate_data(...)``; from 1.6 that method is deprecated in
favour of the free function ``sklearn.utils.validation.validate_data(est, ...)``.

A published package has to work on both, so the choice is made once here rather
than repeated at every call site.
"""

from __future__ import annotations

__all__ = ["SKLEARN_HAS_VALIDATE_DATA", "validate_data"]

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data as _validate_data

    SKLEARN_HAS_VALIDATE_DATA = True

    def validate_data(estimator, X, y="no_validation", **kwargs):
        return _validate_data(estimator, X, y, **kwargs)

except ImportError:  # scikit-learn < 1.6
    SKLEARN_HAS_VALIDATE_DATA = False

    def validate_data(estimator, X, y="no_validation", **kwargs):
        """Delegate to the estimator's own validator.

        Signature-compatible with the 1.6+ free function, so calling code does
        not branch.
        """
        return estimator._validate_data(X, y, **kwargs)
