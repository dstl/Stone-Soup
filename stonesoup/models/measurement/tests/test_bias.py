import datetime

import pytest
import numpy as np

from stonesoup.models.measurement.bias import (
    TimeBiasModelWrapper, TranslationBiasModelWrapper,
    OrientationBiasModelWrapper, OrientationTranslationBiasModelWrapper)
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRangeRate
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.transition.base import CombinedGaussianTransitionModel
from stonesoup.types.state import State, StateVector


@pytest.fixture(params=[
    np.array([[0.], [0.], [0.]]),
    np.array([[0.1], [0.2], [0.3]]),
    np.array([[-0.5], [0.], [np.pi/4]])
])
def orientation_bias(request):
    return request.param


@pytest.fixture(params=[
    np.array([[0.], [0.], [0.]]),
    np.array([[1.], [-2.], [3.]]),
    np.array([[-5.], [0.], [2.5]])
])
def translation_bias(request):
    return request.param


def test_orientation_translation_bias_wrapper(orientation_bias, translation_bias):
    # Setup measurement model
    model = CartesianToElevationBearingRangeRate(
        mapping=[0, 1, 2],
        ndim_state=4,
        noise_covar=np.eye(4),
        translation_offset=StateVector([[0.], [0.], [0.]]),
        rotation_offset=StateVector([[0.], [0.], [0.]]),
    )
    # Wrap with orientation+translation bias
    wrapper = OrientationTranslationBiasModelWrapper(
        measurement_model=model,
        state_mapping=[0, 1, 2, 3, 4, 5],
        bias_mapping=(6, 7, 8, 9, 10, 11),
        ndim_state=12
    )
    # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, tx, ty, tz]
    state_vec = np.vstack((
        [1.], [2.], [3.], [0.1], [0.2], [0.3],
        orientation_bias.reshape(3, 1),
        translation_bias.reshape(3, 1)
    ))
    state = State(StateVector(state_vec))
    # The wrapper should subtract the bias from the model's rotation_offset and translation_offset
    result = wrapper.function(state)
    # Check that the model's offsets were not changed
    assert np.allclose(wrapper.measurement_model.rotation_offset, StateVector([[0.], [0.], [0.]]))
    assert np.allclose(
        wrapper.measurement_model.translation_offset, StateVector([[0.], [0.], [0.]]))
    bias_model = CartesianToElevationBearingRangeRate(
        mapping=[0, 1, 2],
        ndim_state=4,
        noise_covar=np.eye(4),
        translation_offset=StateVector([[0.], [0.], [0.]]),
        rotation_offset=StateVector([[0.], [0.], [0.]]),
    )
    bias_model.rotation_offset = bias_model.rotation_offset - orientation_bias
    bias_model.translation_offset = bias_model.translation_offset - translation_bias
    # Use only [x, y, z, v] for expected state
    expected_state = StateVector(state_vec[:6])
    expected = bias_model.function(State(expected_state))
    assert np.allclose(result, expected)

    # Additional coverage: mapping, ndim_meas, covar, jacobian
    expected_mapping = [0, 1, 2, 6, 7, 8, 9, 10, 11]
    assert wrapper.mapping == expected_mapping
    assert wrapper.ndim_meas == 4
    covar = wrapper.covar()
    assert covar.shape == (4, 4)
    jac = wrapper.jacobian(state)
    assert jac.shape[0] == wrapper.ndim_meas
    assert jac.shape[1] == wrapper.ndim_state


def test_translation_bias_wrapper(translation_bias):
    # Setup measurement model
    model = CartesianToElevationBearingRangeRate(
        mapping=[0, 1, 2],
        ndim_state=4,
        noise_covar=np.eye(4),
        translation_offset=StateVector([[0.], [0.], [0.]]),
        rotation_offset=StateVector([[0.], [0.], [0.]]),
    )
    # Wrap with translation bias
    wrapper = TranslationBiasModelWrapper(
        measurement_model=model,
        state_mapping=[0, 1, 2, 3, 4, 5],
        bias_mapping=(6, 7, 8),
        ndim_state=9
    )
    # State vector: [x, y, z, vx, vy, vz, tx, ty, tz]
    state_vec = np.vstack((
        [1.], [2.], [3.], [0.1], [0.2], [0.3],
        translation_bias.reshape(3, 1)
    ))
    state = State(StateVector(state_vec))
    # The wrapper should subtract the bias from the model's translation_offset
    result = wrapper.function(state)
    # Check that the model's translation_offset was set correctly
    assert np.allclose(
        wrapper.measurement_model.translation_offset, StateVector([[0.], [0.], [0.]]))
    # Check that the bias was applied (the result should be as if translation_offset was -bias)
    bias_model = CartesianToElevationBearingRangeRate(
        mapping=[0, 1, 2],
        ndim_state=4,
        noise_covar=np.eye(4),
        translation_offset=StateVector([[0.], [0.], [0.]]),
        rotation_offset=StateVector([[0.], [0.], [0.]]),
    )
    bias_model.translation_offset = bias_model.translation_offset - translation_bias
    # Use only [x, y, z, v] for expected state
    expected_state = StateVector(state_vec[:6])
    expected = bias_model.function(State(expected_state))
    assert np.allclose(result, expected)

    # Additional coverage: mapping, ndim_meas, covar, jacobian
    expected_mapping = [0, 1, 2, 6, 7, 8]
    assert wrapper.mapping == expected_mapping
    assert wrapper.ndim_meas == 4
    covar = wrapper.covar()
    assert covar.shape == (4, 4)
    jac = wrapper.jacobian(state)
    assert jac.shape[0] == wrapper.ndim_meas
    assert jac.shape[1] == wrapper.ndim_state


def test_orientation_bias_wrapper(orientation_bias):
    # Setup measurement model
    model = CartesianToElevationBearingRangeRate(
        mapping=[0, 1, 2],
        ndim_state=4,
        noise_covar=np.eye(4),
        translation_offset=StateVector([[0.], [0.], [0.]]),
        rotation_offset=StateVector([[0.], [0.], [0.]]),
    )
    # Wrap with orientation bias
    wrapper = OrientationBiasModelWrapper(
        measurement_model=model,
        state_mapping=[0, 1, 2, 3, 4, 5],
        bias_mapping=(6, 7, 8),
        ndim_state=9
    )
    # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw]
    state_vec = np.vstack((
        [1.], [2.], [3.], [0.1], [0.2], [0.3],
        orientation_bias.reshape(3, 1)
    ))
    state = State(StateVector(state_vec))
    # The wrapper should subtract the bias from the model's rotation_offset
    result = wrapper.function(state)
    # Check that the model's rotation_offset was not changed
    assert np.allclose(wrapper.measurement_model.rotation_offset, StateVector([[0.], [0.], [0.]]))
    # Check that the bias was applied (the result should be as if rotation_offset was -bias)
    bias_model = CartesianToElevationBearingRangeRate(
        mapping=[0, 1, 2],
        ndim_state=4,
        noise_covar=np.eye(4),
        translation_offset=StateVector([[0.], [0.], [0.]]),
        rotation_offset=StateVector([[0.], [0.], [0.]]),
    )
    bias_model.rotation_offset = bias_model.rotation_offset - orientation_bias
    expected_state = StateVector(state_vec[:6])
    expected = bias_model.function(State(expected_state))
    assert np.allclose(result, expected)

    # Additional coverage: mapping, ndim_meas, covar, jacobian
    expected_mapping = [0, 1, 2, 6, 7, 8]
    assert wrapper.mapping == expected_mapping
    assert wrapper.ndim_meas == 4
    covar = wrapper.covar()
    assert covar.shape == (4, 4)
    jac = wrapper.jacobian(state)
    assert jac.shape[0] == wrapper.ndim_meas
    assert jac.shape[1] == wrapper.ndim_state


@pytest.mark.parametrize('bias_mapping', [-1, (-1, )])
def test_time_bias_wrapper(bias_mapping):
    # Setup transition model: 3D constant velocity
    cv_x = ConstantVelocity(noise_diff_coeff=0.1)
    cv_y = ConstantVelocity(noise_diff_coeff=0.1)
    cv_z = ConstantVelocity(noise_diff_coeff=0.1)
    transition_model = CombinedGaussianTransitionModel(model_list=[cv_x, cv_y, cv_z])

    # Setup measurement model (dummy, just returns state)
    class DummyMeasurementModel:
        mapping = [0, 2, 4]
        ndim_meas = 3

        def function(self, state, noise=False, **kwargs):
            # Just return the position components
            return state.state_vector[[0, 2, 4], :]

        def covar(self, *args, **kwargs):
            return np.eye(3)

        def jacobian(self, state, **kwargs):
            return np.eye(3, 7)

    # Wrap with time bias
    wrapper = TimeBiasModelWrapper(
        measurement_model=DummyMeasurementModel(),
        transition_model=transition_model,
        state_mapping=[0, 1, 2, 3, 4, 5],
        bias_mapping=bias_mapping,
        ndim_state=7
    )

    # State vector: [x, vx, y, vy, z, vz, bias]
    state_vec = np.array([[1.], [2.], [3.], [4.], [5.], [6.], [1.]])
    state = State(StateVector(state_vec))

    # The wrapper should apply the transition model with time_interval = -bias
    expected_state = transition_model.function(
        State(StateVector(state_vec[:6])),
        time_interval=datetime.timedelta(seconds=-state_vec[6, 0])
    )
    expected = wrapper.measurement_model.function(State(expected_state))
    result = wrapper.function(state)
    assert np.allclose(result, expected)

    # Also test with negative bias (forward in time)
    state_vec2 = np.array([[1.], [2.], [3.], [4.], [5.], [6.], [-2.]])
    state2 = State(StateVector(state_vec2))
    expected_state2 = transition_model.function(
        State(StateVector(state_vec2[:6])),
        time_interval=datetime.timedelta(seconds=-state_vec2[6, 0])
    )
    expected2 = wrapper.measurement_model.function(State(expected_state2))
    result2 = wrapper.function(state2)
    assert np.allclose(result2, expected2)
