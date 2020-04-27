# -*- coding: utf-8 -*-
import numpy as np


class Matrix(np.ndarray):
    """Matrix wrapper for :class:`numpy.ndarray`

    This class returns a view to a :class:`numpy.ndarray` It's called same as
    to :func:`numpy.asarray`.
    """

    def __new__(cls, *args, **kwargs):
        array = np.asarray(*args, **kwargs)
        return array.view(cls)

    def __array_wrap__(self, array):
        return Matrix._cast(np.asarray(array))

    @classmethod
    def _cast(cls, val):
        # This tries to cast the result as either a StateVector or
        # Matrix type if applicable.
        if isinstance(val, np.ndarray):
            if val.ndim == 2:
                if val.shape[1] == 1:
                    return val.view(StateVector)
                else:
                    return val.view(Matrix)
            else:
                return val.view(Matrix)
        else:
            return val

    def __matmul__(self, other):
        out = np.matmul(np.asfarray(self), np.asfarray(other))
        return self._cast(out)

    def __rmatmul__(self, other):
        out = np.matmul(np.asfarray(other), np.asfarray(self))
        return self._cast(out)


class StateVector(Matrix):
    r"""State vector wrapper for :class:`numpy.ndarray`

    This class returns a view to a :class:`numpy.ndarray`, but ensures that
    its initialised as an :math:`N \times 1` vector. It's called same as
    :func:`numpy.asarray`. The StateVector will attempt to convert the data
    given to a :math:`N \times 1` vector if it can easily be done. E.g.,
    ``StateVector([1., 2., 3.])``, ``StateVector ([[1., 2., 3.,]])``, and
    ``StateVector([[1.], [2.], [3.]])`` will all return the same 3x1 StateVector.

    .. note ::
        It is not recommended to use a StateVector for indexing another vector. Doing so will lead
        to unexpected effects. Use a :class:`tuple`, :class:`list` or :class:`np.ndarray` for this.
    """

    def __new__(cls, *args, **kwargs):
        array = np.asarray(*args, **kwargs)
        # For convenience handle shapes that can be easily converted in a
        # Nx1 shape
        if array.ndim == 1:
            array = array.reshape((array.shape[0], 1))
        elif array.ndim == 2 and array.shape[0] == 1:
            array = array.T

        if not (array.ndim == 2 and array.shape[1] == 1):
            raise ValueError(
                "state vector shape should be Nx1 dimensions: got {}".format(
                    array.shape))
        return array.view(cls)


class CovarianceMatrix(Matrix):
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
