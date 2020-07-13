# -*- coding: utf-8 -*-
from collections.abc import Sequence

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
        return self._cast(array)

    @classmethod
    def _cast(cls, val):
        # This tries to cast the result as either a StateVector or Matrix type if applicable.
        if isinstance(val, np.ndarray):
            if val.ndim == 2 and val.shape[1] == 1:
                return val.view(StateVector)
            else:
                return val.view(Matrix)
        else:
            return val

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in (np.isfinite, np.matmul):
            # Custom types break here, so simply convert to floats.
            inputs = [np.asfarray(input_) if isinstance(input_, Matrix) else input_
                      for input_ in inputs]
        else:
            # Change to standard ndarray
            inputs = [np.asarray(input_) if isinstance(input_, Matrix) else input_
                      for input_ in inputs]
        if 'out' in kwargs:
            kwargs['out'] = tuple(np.asarray(out) if isinstance(out, Matrix) else out
                                  for out in kwargs['out'])

        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if result is NotImplemented:
            return NotImplemented
        else:
            return self._cast(result)


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

    When slicing would result in return of a invalid shape for a StateVector (i.e. not `(n, 1)`)
    then a :class:`~.Matrix` view will be returned.

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
        # Cast here, so StateVector isn't returned with invalid shape (e.g. (n, ))
        return self._cast(super().__getitem__(item))

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = (key, 0)
        return super().__setitem__(key, value)

    def flatten(self, *args, **kwargs):
        return self._cast(super().flatten(*args, **kwargs))

    def ravel(self, *args, **kwargs):
        return self._cast(super().ravel(*args, **kwargs))


class StateVectors(Matrix):
    """Wrapper for :class:`numpy.ndarray for multiple State Vectors`

    This class returns a view to a :class:`numpy.ndarray` that is in shape
    (num_dimensions, num_components), customising some numpy functions to ensure
    custom types are handled correctly. This can be initialised by a sequence
    type (list, tuple; not array) that contains :class:`StateVector`, otherwise
    it's called same as :func:`numpy.asarray`.
    """

    def __new__(cls, states, *args, **kwargs):
        if isinstance(states, Sequence) and not isinstance(states, np.ndarray):
            if isinstance(states[0], StateVector):
                return np.hstack(states).view(cls)
        array = np.asarray(states, *args, **kwargs)
        return array.view(cls)

    @classmethod
    def _cast(cls, val):
        out = super()._cast(val)
        if type(out) == Matrix:
            # Assume still set of State Vectors
            return out.view(StateVectors)
        else:
            return out

    def __array_function__(self, func, types, args, kwargs):
        if func is np.average:
            return self._average(*args, **kwargs)
        elif func is np.mean:
            return self._mean(*args, **kwargs)
        elif func is np.cov:
            return self._cov(*args, **kwargs)
        else:
            return super().__array_function__(func, types, args, kwargs)

    @staticmethod
    def _mean(state_vectors, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        if state_vectors.dtype != np.object_:
            # Can just use standard numpy mean if not using custom objects
            return np.mean(axis, dtype, out, keepdims)
        elif axis == 1 and out is None:
            state_vector = np.average(state_vectors, axis)
            if dtype:
                return state_vector.astype(dtype)
            else:
                return state_vector
        else:
            return NotImplemented

    @staticmethod
    def _average(state_vectors, axis=None, weights=None, returned=False):
        if state_vectors.dtype != np.object_:
            # Can just use standard numpy averaging if not using custom objects
            state_vector = np.average(np.asarray(state_vectors), axis=axis, weights=weights)
            # Convert type as may have type of weights
            state_vector = StateVector(state_vector.astype(np.float_, copy=False))
        elif axis == 1:  # Need to handle special cases of averaging potentially
            state_vector = StateVector(
                np.empty((state_vectors.shape[0], 1), dtype=state_vectors.dtype))
            for dim, row in enumerate(state_vectors):
                type_ = type(row[0])  # Assume all the same type
                if hasattr(type_, 'average'):
                    # Check if type has custom average method
                    state_vector[dim, 0] = type_.average(row, weights=weights)
                else:
                    # Else use numpy built in, converting to float array
                    state_vector[dim, 0] = type_(np.average(np.asfarray(row), weights=weights))
        else:
            return NotImplemented

        if returned:
            return state_vector, np.sum(weights)
        else:
            return state_vector

    @staticmethod
    def _cov(state_vectors, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
             aweights=None):

        if state_vectors.dtype != np.object_:
            # Can just use standard numpy averaging if not using custom objects
            cov = np.cov(np.asarray(state_vectors), y, rowvar, bias, ddof, fweights, aweights)
        elif y is None and rowvar and not bias and ddof == 0 and fweights is None:
            # Only really handle simple usage here
            avg, w_sum = np.average(state_vectors, axis=1, weights=aweights, returned=True)

            X = np.asfarray(state_vectors - avg)
            if aweights is None:
                X_T = X.T
            else:
                X_T = (X*np.asfarray(aweights)).T
            cov = X @ X_T.conj()
            cov *= np.true_divide(1, float(w_sum))
        else:
            return NotImplemented
        return CovarianceMatrix(np.atleast_2d(cov))


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
