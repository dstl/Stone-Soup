# -*- coding: utf-8 -*-
import numpy as np


class Array(np.ndarray):
    def __array_wrap__(self, array):
        return np.asarray(array)


class StateVector(Array):
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


class CovarianceMatrix(Array):
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
