# -*- coding: utf-8 -*-
import datetime

import numpy as np
from scipy.spatial import distance

from .. import measures

from ..types.array import StateVector, CovarianceMatrix
from ..types.state import GaussianState

# Create a time stamp to use for both states
t = datetime.datetime.now()

# Set target ground truth prior
u = StateVector([[10], [1], [10], [1]])
ui = CovarianceMatrix(np.diag([100, 10, 100, 10]))

state_u = GaussianState(u, ui, timestamp=t)

v = StateVector([[11], [10], [100], [2]])
vi = CovarianceMatrix(np.diag([20, 3, 7, 10]))

state_v = GaussianState(v, vi, timestamp=t)


def test_euclidean():

    measure = measures.Euclidean()
    assert measure(state_u, state_v) == distance.euclidean(u, v)


def test_euclideanweighted():
    weight = np.array([[1], [2], [3], [1]])
    measure = measures.EuclideanWeighted(weight)
    assert measure(state_u, state_v) == distance.euclidean(u, v, weight)


def test_mahalanobis():
    measure = measures.Mahalanobis()
    assert measure(state_u, state_v) == distance.mahalanobis(u,
                                                             v,
                                                             np.linalg.inv(ui))


def test_hellinger():
    measure = measures.GaussianHellinger()
    # Distance value obtained from MATLAB
    assert np.isclose(measure(state_u, state_v), 0.665, atol=1e-3)


def test_zero_hellinger():
    measure = measures.GaussianHellinger()
    # Set target ground truth prior
    u = StateVector([[10], [1], [10], [1]])
    ui = CovarianceMatrix(np.diag([0, 0, 0, 0]))
    state_u = GaussianState(u, ui, timestamp=t)

    v = StateVector([[11], [10], [100], [2]])
    vi = CovarianceMatrix(np.diag([0, 0, 0, 0]))
    state_v = GaussianState(v, vi, timestamp=t)
    print(measure(state_u, state_v))
    # Distance value obtained from MATLAB
    assert np.isclose(measure(state_u, state_v), 1, atol=1e-3)


def test_squared_hellinger():
    measure = measures.SquaredGaussianHellinger()
    # Distance value obtained from MATLAB
    assert np.isclose(measure(state_u, state_v), 0.444, atol=1e-3)


def test_hellinger_full_mapping():
    mapping = np.arange(len(u))
    measure = measures.GaussianHellinger(mapping=mapping)
    assert np.isclose(measure(state_u, state_v), 0.665, atol=1e-3)


def test_hellinger_partial_mapping():
    # Distance value obtained from MATLAB
    mapping = np.array([0, 1])
    measure = measures.GaussianHellinger(mapping=mapping)
    assert np.isclose(measure(state_u, state_v), 0.4555, atol=1e-3)
    mapping = np.array([0, 3])
    measure = measures.GaussianHellinger(mapping=mapping)
    assert np.isclose(measure(state_u, state_v), 0.3701, atol=1e-3)


def test_mahalanobis_full_mapping():
    mapping = np.arange(len(u))
    measure = measures.Mahalanobis(mapping=mapping)
    assert measure(state_u, state_v) == distance.mahalanobis(u,
                                                             v,
                                                             np.linalg.inv(ui))


def test_mahalanobis_partial_mapping():
    mapping = np.array([0, 1])
    measure = measures.Mahalanobis(mapping=mapping)
    reduced_ui = CovarianceMatrix(np.diag([100, 10]))
    assert measure(state_u, state_v) == \
        distance.mahalanobis([[10], [1]],
                             [[11], [10]], np.linalg.inv(reduced_ui))
    mapping = np.array([0, 3])
    reduced_ui = CovarianceMatrix(np.diag([100, 10]))
    measure = measures.Mahalanobis(mapping=mapping)
    assert measure(state_u, state_v) == \
        distance.mahalanobis([[10], [1]],
                             [[11], [2]], np.linalg.inv(reduced_ui))


def test_euclidean_full_mapping():
    mapping = np.arange(len(u))
    measure = measures.Euclidean(mapping=mapping)
    assert measure(state_u, state_v) == distance.euclidean(u, v)


def test_euclidean_partial_mapping():
    mapping = np.array([0, 1])
    measure = measures.Euclidean(mapping=mapping)
    assert measure(state_u, state_v) == \
        distance.euclidean([[10], [1]], [[11], [10]])
    mapping = np.array([0, 3])
    measure = measures.Euclidean(mapping=mapping)
    assert measure(state_u, state_v) == \
        distance.euclidean([[10], [1]], [[11], [2]])


def test_euclideanweighted_full_mapping():
    weight = np.array([[1], [2], [3], [1]])
    mapping = np.arange(len(u))
    measure = measures.EuclideanWeighted(weight, mapping=mapping)
    assert measure(state_u, state_v) == distance.euclidean(u, v, weight)


def test_euclideanweighted_partial_mapping():
    weight = np.array([[1], [2]])
    mapping = np.array([0, 1])
    measure = measures.EuclideanWeighted(weight, mapping=mapping)
    assert measure(state_u, state_v) == \
        distance.euclidean([[10], [1]], [[11], [10]], weight)
    mapping = np.array([0, 3])
    measure = measures.EuclideanWeighted(weight, mapping=mapping)
    assert measure(state_u, state_v) == \
        distance.euclidean([[10], [1]], [[11], [2]], weight)
