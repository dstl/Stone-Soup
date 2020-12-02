# coding: utf-8
import pytest
from pytest import approx
import numpy as np

from ....types.angle import Bearing
from ....types.array import StateVector, CovarianceMatrix
from ....types.detection import Detection
from ....types.state import State
from ..linear import LinearGaussian
from ..nonlinear import (
    CombinedReversibleGaussianMeasurementModel, CartesianToBearingRange)


@pytest.fixture(scope="module")
def model():
    return CombinedReversibleGaussianMeasurementModel([
        CartesianToBearingRange(5, [0, 1], np.diag([1, 10])),
        CartesianToBearingRange(5, [3, 4], np.diag([2, 20])),
    ])


def test_non_linear(model):
    assert model.ndim_meas == 4
    assert model.ndim_state == 5

    meas_vector = model.function(
        Detection(StateVector([[0], [10], [10], [0], [-10]])))

    assert isinstance(meas_vector[0, 0], Bearing)
    assert not isinstance(meas_vector[1, 0], Bearing)
    assert isinstance(meas_vector[2, 0], Bearing)
    assert not isinstance(meas_vector[3, 0], Bearing)
    assert isinstance(meas_vector, StateVector)

    assert np.array_equal(meas_vector,
                          np.array([[np.pi/2], [10], [-np.pi/2], [10]]))


def test_jacobian(model):
    state = State(StateVector([[10.0], [10.0], [0.0], [10.0], [0.0]]))
    jacobian = model.jacobian(state)
    assert jacobian == approx(np.array([[-0.05,      0.05,       0, 0, 0],
                                        [0.70710678, 0.70710678, 0, 0, 0],
                                        [0,          0,          0, 0, 0.1],
                                        [0,          0,          0, 1, 0]]))


def test_covar(model):
    covar = model.covar()
    assert covar == approx(np.diag([1, 10, 2, 20]))
    assert isinstance(covar, CovarianceMatrix)


def test_inverse(model):
    state = State(StateVector([[0.1], [10], [0], [0.2], [20]]))
    meas_state = model.function(state)

    assert model.inverse_function(State(meas_state)) == approx(state.state_vector)


def test_rvs(model):
    rvs_state = model.rvs()
    assert isinstance(rvs_state[0, 0], Bearing)
    assert not isinstance(rvs_state[1, 0], Bearing)
    assert isinstance(rvs_state[2, 0], Bearing)
    assert not isinstance(rvs_state[3, 0], Bearing)
    assert rvs_state.shape == (4, 1)
    assert isinstance(rvs_state, StateVector)

    rvs_state = model.rvs(10)
    assert rvs_state.shape == (4, 10)
    assert all(isinstance(state, Bearing) for state in rvs_state[0])
    assert all(not isinstance(state, Bearing) for state in rvs_state[1])
    assert all(isinstance(state, Bearing) for state in rvs_state[2])
    assert all(not isinstance(state, Bearing) for state in rvs_state[3])


def test_pdf(model):
    pdf = model.pdf(State(StateVector([[0], [10], [0], [10]])),
                    State(StateVector([[10], [0], [0], [10], [0]])))
    assert float(pdf) == approx(0.0012665, rel=1e-3)


def test_non_linear_and_linear():
    model = CombinedReversibleGaussianMeasurementModel([
        CartesianToBearingRange(3, [0, 1], np.diag([1, 10])),
        LinearGaussian(3, [2], np.array([[20]])),
    ])

    state = State(StateVector([[0], [10], [20]]))
    meas_vector = model.function(state)
    assert isinstance(meas_vector[0, 0], Bearing)
    assert not isinstance(meas_vector[1, 0], Bearing)
    assert not isinstance(meas_vector[2, 0], Bearing)
    assert isinstance(meas_vector, StateVector)
    assert np.array_equal(meas_vector, np.array([[np.pi/2], [10], [20]]))

    assert model.inverse_function(State(meas_vector)) == approx(state.state_vector)

    assert model.covar() == approx(np.diag([1, 10, 20]))


def test_mismatch_ndim_state():
    with pytest.raises(ValueError):
        CombinedReversibleGaussianMeasurementModel([
            CartesianToBearingRange(3, [0, 1], np.diag([1, 10])),
            CartesianToBearingRange(4, [0, 1], np.diag([1, 10])),
        ])


def test_none_covar():
    with pytest.raises(ValueError, match="Gaussian models must have defined covariances"):
        CombinedReversibleGaussianMeasurementModel([
            CartesianToBearingRange(3, [0, 1], None),
            CartesianToBearingRange(4, [0, 1], np.diag([1, 10]))
        ])
