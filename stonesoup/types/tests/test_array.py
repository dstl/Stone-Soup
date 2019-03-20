# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ..array import StateVector, CovarianceMatrix


def test_statevector():
    with pytest.raises(ValueError):
        StateVector(np.array([0]))

    state_vector_array = np.array([[1], [2], [3], [4]])
    state_vector = StateVector(state_vector_array)

    assert np.array_equal(state_vector, state_vector_array)


def test_covariancematrix():
    """ CovarianceMatrix Type test """

    with pytest.raises(ValueError):
        CovarianceMatrix(np.array([0]))

    covar_nparray = np.array([[2.2128, 0, 0, 0],
                              [0.0002, 2.2130, 0, 0],
                              [0.3897, -0.00004, 0.0128, 0],
                              [0, 0.3897, 0.0013, 0.0135]]) * 1e3

    covar_matrix = CovarianceMatrix(covar_nparray)
    assert(np.array_equal(covar_matrix, covar_nparray))


def test_multiplication():
    vector = np.array([[1, 1, 1, 1]]).T
    state_vector = StateVector(vector)
    array = np.array([[1., 2., 3., 4.],
                      [5., 6., 7., 8.]])
    covar = CovarianceMatrix(array)

    assert np.array_equal(covar@state_vector, array@vector)
    assert np.array_equal(covar@vector, array@vector)
    assert np.array_equal(array@state_vector, array@vector)
    assert np.array_equal(state_vector.T@covar.T, vector.T@array.T)
    assert np.array_equal(vector.T@covar.T, vector.T@array.T)
    assert np.array_equal(state_vector.T@array.T, vector.T@array.T)

    assert type(array@state_vector) == type(array)
    assert type(state_vector.T@array.T) == type(state_vector)
    assert type(covar@vector) == type(covar)
    assert type(vector.T@covar.T) == type(vector)
