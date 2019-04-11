# -*- coding: utf-8 -*-
import numpy as np


class StateVector(np.ndarray):
    """State vector wrapper for :class:`numpy.ndarray`

    This class returns a view to a :class:`numpy.ndarray`, but ensures that
    its initialised at a *Nx1* vector. It's called same as to
    :func:`numpy.asarray`.
    """

    def __new__(cls, *args, **kwargs):
        array = np.asarray(*args, **kwargs)
        if not (array.ndim == 2 and array.shape[1] == 1):
            raise ValueError(
                "state vector shape should be Nx1 dimensions: got {}".format(
                    array.shape))
        return array.view(cls)

    def __array_wrap__(self, array):
        return StateVector._cast(np.asarray(array))

    def __matmul__(self, other):
        out = np.matmul(np.asfarray(self), np.asfarray(other))
        return out.view(type=type(self))

    def __rmatmul__(self, other):
        out = np.matmul(np.asfarray(other), np.asfarray(self))
        return out.view(type=type(other))

    @classmethod
    def _cast(cls, val):
        # This tries to cast the result as either a StateVector or
        # CovarianceMatrix if applicable.
        if val.ndim == 2:
            if val.shape[1] == 1:
                return cls(val)
            else:
                return CovarianceMatrix(val)
        else:
            return val

    def __sub__(self, other):
        tmp = super().__sub__(other)
        return StateVector._cast(tmp)

    def __rsub__(self, other):
        tmp = super().__rsub__(other)
        return StateVector._cast(tmp)

    def __add__(self, other):
        tmp = super().__add__(other)
        return StateVector._cast(tmp)

    def __radd__(self, other):
        tmp = super().__radd__(other)
        return StateVector._cast(tmp)


class CovarianceMatrix(np.ndarray):
    """Covariance matrix wrapper for :class:`numpy.ndarray`.

    This class returns a view to a :class:`numpy.ndarray`, but ensures that
    its initialised at a *NxN* matrix. It's called similar to
    :func:`numpy.asarray`.
    """

    def __new__(cls, *args, **kwargs):
        array = np.asarray(*args, **kwargs)
        if not array.ndim == 2:
            raise ValueError("Covariance should have ndim of 2: got {}"
                             "".format(array.ndim))
        return array.view(cls)

    def __array_wrap__(self, array):
        return StateVector._cast(np.asarray(array))

    def __matmul__(self, other):
        out = np.matmul(np.asfarray(self), np.asfarray(other))
        return out.view(type=type(self))

    def __rmatmul__(self, other):
        out = np.matmul(np.asfarray(other), np.asfarray(self))
        return out.view(type=type(other))

    def __sub__(self, other):
        return type(self)(super().__sub__(other))

    def __rsub__(self, other):
        return type(self)(super().__rsub__(other))

    def __add__(self, other):
        return type(self)(super().__add__(other))

    def __radd__(self, other):
        return type(self)(super().__radd__(other))
