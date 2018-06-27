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
