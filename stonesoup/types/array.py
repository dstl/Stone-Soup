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

    It also overrides the behaviour of indexing such that my_state_vector[1] returns the second
    element (as `int`, `float` etc), rather than a StateVector of size (1, 1) as would be the case
    without this override. Behaviour of indexing with lists, slices or other indexing is
    unaffected (as you would expect those to return StateVectors). This override avoids the need
    for client to specifically index with zero as the second element (`my_state_vector[1, 0]`) to
    get a native numeric type. Iterating through the StateVector returns a sequence of numbers,
    rather than a sequence of 1x1 StateVectors. This makes the class behave as would be expected
    and avoids 'gotchas'.

    Note that code using the pattern `my_state_vector[1, 0]` will continue to work.

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

    def __getitem__(self, item):
        # If item has two elements, it is a tuple and should be left alone.
        # If item is a slice object, or an ndarray, we would expect a StateVector returned,
        #   so leave it alone.
        # If item is an int, we would expected a number returned, so we should append 0  to the
        #   item and extract the first (and only) column
        # Note that an ndarray of ints is an instance of int
        #   i.e. isinstance(np.array([1]), int) == True
        if isinstance(item, int):
            item = (item, 0)
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = (key, 0)
        return super().__setitem__(key, value)


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
