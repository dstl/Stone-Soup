import numpy as np
import pytest

from ..array import Matrix, StateVector, StateVectors, CovarianceMatrix, PrecisionMatrix


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
    assert np.array_equal(StateVector(state_vector_array), state_vector)


def test_statevectors():
    vec1 = np.array([[1.], [2.], [3.]])
    vec2 = np.array([[2.], [3.], [4.]])

    sv1 = StateVector(vec1)
    sv2 = StateVector(vec2)

    vecs1 = np.concatenate((vec1, vec2), axis=1)
    svs1 = StateVectors([sv1, sv2])
    svs2 = StateVectors(vecs1)
    svs3 = StateVectors([vec1, vec2])  # Creates 3dim array
    assert np.array_equal(svs1, vecs1)
    assert np.array_equal(svs2, vecs1)
    assert svs3.shape != vecs1.shape

    for sv in svs2:
        assert isinstance(sv, StateVector)


def test_statevectors_mean():
    svs = StateVectors([[1., 2., 3.], [4., 5., 6.]])
    mean = StateVector([[2., 5.]])

    assert np.allclose(np.average(svs, axis=1), mean)
    assert np.allclose(np.mean(svs, axis=1, keepdims=True), mean)


def test_standard_statevector_indexing():
    state_vector_array = np.array([[1], [2], [3], [4]])
    state_vector = StateVector(state_vector_array)

    # test standard indexing
    assert state_vector[2, 0] == 3
    assert not isinstance(state_vector[2, 0], StateVector)

    # test Slicing
    assert state_vector[1:2, 0] == 2
    assert isinstance(state_vector[1:2, 0], Matrix)  # (n,)
    assert isinstance(state_vector[1:2, :], StateVector)  # (n, 1)
    assert np.array_equal(state_vector[:], state_vector)
    assert isinstance(state_vector[:, 0], Matrix)  # (n,)
    assert isinstance(state_vector[:, :], StateVector)  # (n, 1)
    assert np.array_equal(state_vector[0:], state_vector)
    assert isinstance(state_vector[0:, 0], Matrix)  # (n,)
    assert isinstance(state_vector[0:, :], StateVector)  # (n, 1)

    # test list indices
    assert np.array_equal(state_vector[[1, 3]], StateVector([2, 4]))
    assert isinstance(state_vector[[1, 3], 0], Matrix)

    # test int indexing
    assert state_vector[2] == 3
    assert not isinstance(state_vector[2], StateVector)

    # test behaviour of ravel and flatten functions
    state_vector_ravel = state_vector.ravel()
    state_vector_flatten = state_vector.flatten()
    assert isinstance(state_vector_ravel, Matrix)
    assert isinstance(state_vector_flatten, Matrix)
    assert state_vector_flatten[0] == 1
    assert state_vector_ravel[0] == 1


def test_setting():
    state_vector_array = np.array([[1], [2], [3], [4]])
    state_vector = StateVector(state_vector_array.copy())

    state_vector[2, 0] = 4
    assert np.array_equal(state_vector, StateVector([1, 2, 4, 4]))

    state_vector[2] = 5
    assert np.array_equal(state_vector, StateVector([1, 2, 5, 4]))

    state_vector[:] = state_vector_array[:]
    assert np.array_equal(state_vector, StateVector([1, 2, 3, 4]))

    state_vector[1:3] = StateVector([5, 6])
    assert np.array_equal(state_vector, StateVector([1, 5, 6, 4]))


def test_covariancematrix():
    """ CovarianceMatrix Type test """

    with pytest.raises(ValueError):
        CovarianceMatrix(np.array([0]))

    covar_nparray = np.array([[2.2128, 0, 0, 0],
                              [0.0002, 2.2130, 0, 0],
                              [0.3897, -0.00004, 0.0128, 0],
                              [0, 0.3897, 0.0013, 0.0135]]) * 1e3

    covar_matrix = CovarianceMatrix(covar_nparray)
    assert np.array_equal(covar_matrix, covar_nparray)


def test_precisionmatrix():
    """ CovarianceMatrix Type test """

    with pytest.raises(ValueError):
        PrecisionMatrix(np.array([0]))

    prec_nparray = np.array([[7, 1, 0.5, 0],
                             [1, 4, 2, 0.4],
                             [0.5, 2, 6, 0.3],
                             [0, 0.4, 0.3, 5]])

    prec_matrix = PrecisionMatrix(prec_nparray)
    assert np.array_equal(prec_matrix, prec_nparray)


def test_matrix():
    """ Matrix Type test """

    covar_nparray = np.array([[2.2128, 0, 0, 0],
                              [0.0002, 2.2130, 0, 0],
                              [0.3897, -0.00004, 0.0128, 0],
                              [0, 0.3897, 0.0013, 0.0135]]) * 1e3

    matrix = Matrix(covar_nparray)
    assert np.array_equal(matrix, covar_nparray)


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

    assert type(array@state_vector) == Vtype  # noqa: E721
    assert type(state_vector.T@array.T) == Mtype  # noqa: E721
    assert type(covar@vector) == Vtype  # noqa: E721
    assert type(vector.T@covar.T) == Mtype  # noqa: E721


def test_array_ops():
    vector = np.array([[1, 1, 1, 1]]).T
    vector2 = vector + 2.
    sv = StateVector(vector)
    array = np.array([[1., 2., 3., 4.], [2., 3., 4., 5.]]).T
    covar = CovarianceMatrix(array)
    Mtype = Matrix
    Vtype = type(sv)

    assert np.array_equal(covar - vector, array - vector)
    assert type(covar-vector) == Mtype  # noqa: E721
    assert np.array_equal(covar + vector, array + vector)
    assert type(covar+vector) == Mtype  # noqa: E721
    assert np.array_equal(vector - covar, vector - array)
    assert type(vector - covar) == Mtype  # noqa: E721
    assert np.array_equal(vector + covar, vector + array)
    assert type(vector + covar) == Mtype  # noqa: E721

    assert np.array_equal(vector2 - sv, vector2 - vector)
    assert type(vector2 - sv) == Vtype  # noqa: E721
    assert np.array_equal(sv - vector2, vector - vector2)
    assert type(sv - vector2) == Vtype  # noqa: E721
    assert np.array_equal(vector2 + sv, vector2 + vector)
    assert type(vector2 + sv) == Vtype  # noqa: E721
    assert np.array_equal(sv + vector2, vector + vector2)
    assert type(sv + vector2) == Vtype  # noqa: E721
    assert type(sv+2.) == Vtype  # noqa: E721
    assert type(sv*2.) == Vtype  # noqa: E721

    assert np.array_equal(array - sv, array - vector)
    assert type(array - sv) == Mtype  # noqa: E721
    assert np.array_equal(sv - array, vector - array)
    assert type(sv - array) == Mtype  # noqa: E721
    assert np.array_equal(array + sv, array + vector)
    assert type(array + sv) == Mtype  # noqa: E721
    assert np.array_equal(sv + array, vector + array)
    assert type(sv + array) == Mtype  # noqa: E721
    assert type(covar+2.) == Mtype  # noqa: E721
    assert type(covar*2.) == Mtype  # noqa: E721
