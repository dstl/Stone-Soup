# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ..array import Matrix, StateVector, CovarianceMatrix


def test_statevector():
    with pytest.raises(ValueError):
        StateVector([[0, 1], [1, 2]])

    with pytest.raises(ValueError):
        StateVector([[[0, 1], [1, 2]]])

    state_vector_array = np.array([[1], [2], [3], [4]])
    state_vector = StateVector(state_vector_array)

    assert np.array_equal(state_vector, state_vector_array)
    assert np.array_equal(StateVector([1, 2, 3, 4]), state_vector_array)
    assert np.array_equal(StateVector([[1, 2, 3, 4]]), state_vector_array)


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


def test_matrix():
    """ Matrix Type test """

    covar_nparray = np.array([[2.2128, 0, 0, 0],
                              [0.0002, 2.2130, 0, 0],
                              [0.3897, -0.00004, 0.0128, 0],
                              [0, 0.3897, 0.0013, 0.0135]]) * 1e3

    matrix = Matrix(covar_nparray)
    assert(np.array_equal(matrix, covar_nparray))


def test_multiplication():
    vector = np.array([[1, 1, 1, 1]]).T
    state_vector = StateVector(vector)
    array = np.array([[1., 2., 3., 4.],
                      [5., 6., 7., 8.]])
    covar = CovarianceMatrix(array)
    Mtype = Matrix
    Vtype = StateVector

    assert np.array_equal(covar@state_vector, array@vector)
    assert np.array_equal(covar@vector, array@vector)
    assert np.array_equal(array@state_vector, array@vector)
    assert np.array_equal(state_vector.T@covar.T, vector.T@array.T)
    assert np.array_equal(vector.T@covar.T, vector.T@array.T)
    assert np.array_equal(state_vector.T@array.T, vector.T@array.T)

    assert type(array@state_vector) == Vtype
    assert type(state_vector.T@array.T) == Mtype
    assert type(covar@vector) == Vtype
    assert type(vector.T@covar.T) == Mtype


def test_array_ops():
    vector = np.array([[1, 1, 1, 1]]).T
    vector2 = vector + 2.
    sv = StateVector(vector)
    array = np.array([[1., 2., 3., 4.], [2., 3., 4., 5.]]).T
    covar = CovarianceMatrix(array)
    Mtype = Matrix
    Vtype = type(sv)

    assert np.array_equal(covar - vector, array - vector)
    assert type(covar-vector) == Mtype
    assert np.array_equal(covar + vector, array + vector)
    assert type(covar+vector) == Mtype
    assert np.array_equal(vector - covar, vector - array)
    assert type(vector - covar) == Mtype
    assert np.array_equal(vector + covar, vector + array)
    assert type(vector + covar) == Mtype

    assert np.array_equal(vector2 - sv, vector2 - vector)
    assert type(vector2 - sv) == Vtype
    assert np.array_equal(sv - vector2, vector - vector2)
    assert type(sv - vector2) == Vtype
    assert np.array_equal(vector2 + sv, vector2 + vector)
    assert type(vector2 + sv) == Vtype
    assert np.array_equal(sv + vector2, vector + vector2)
    assert type(sv + vector2) == Vtype
    assert type(sv+2.) == Vtype
    assert type(sv*2.) == Vtype

    assert np.array_equal(array - sv, array - vector)
    assert type(array - sv) == Mtype
    assert np.array_equal(sv - array, vector - array)
    assert type(sv - array) == Mtype
    assert np.array_equal(array + sv, array + vector)
    assert type(array + sv) == Mtype
    assert np.array_equal(sv + array, vector + array)
    assert type(sv + array) == Mtype
    assert type(covar+2.) == Mtype
    assert type(covar*2.) == Mtype
