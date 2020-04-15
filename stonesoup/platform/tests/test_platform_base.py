# coding: utf-8
import datetime

import numpy as np
import pytest

from stonesoup.types.array import StateVector
from ...types.state import State
from ...models.transition.linear import (
    ConstantVelocity, CombinedLinearGaussianTransitionModel)
from ..base import MovingPlatform, FixedPlatform


def test_base():
    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    # Define a static 2d platform and check it does not move
    platform_state2d = State(np.array([[2],
                                       [2]]),
                             timestamp)
    platform = FixedPlatform(state=platform_state2d, position_mapping=np.array([0, 1]))
    platform.move(new_timestamp)
    new_statevector = np.array([[2],
                                [2]])
    # Ensure 2d platform has not moved
    assert (np.array_equal(platform.state.state_vector, new_statevector))
    # Test to ensure platform time has updated
    assert (platform.state.timestamp == new_timestamp)
    assert np.array_equal(platform.velocity, StateVector([0, 0]))
    assert platform.ndim == 2
    assert not platform.is_moving

    # Define a static 3d platform and check it does not move
    platform_state3d = State(np.array([[2],
                                       [2],
                                       [2]]),
                             timestamp)
    platform = FixedPlatform(state=platform_state3d, position_mapping=[0, 1, 2])
    platform.move(new_timestamp)
    new_statevector = np.array([[2],
                                [2],
                                [2]])
    # Ensure 2d platform has not moved
    assert np.array_equal(platform.state.state_vector, new_statevector)
    assert np.array_equal(platform.velocity, StateVector([0, 0, 0]))
    assert platform.ndim == 3
    assert not platform.is_moving

    # Define zero noise 2d constant velocity transition model
    model_1d = ConstantVelocity(0.0)
    model_2d = CombinedLinearGaussianTransitionModel(
        [model_1d, model_1d])

    # Define a 2d platform with constant velocity motion and test motion
    platform_state2d = State(np.array([[2],
                                       [1],
                                       [2],
                                       [1]]),
                             timestamp)
    platform = MovingPlatform(state=platform_state2d, transition_model=model_2d,
                              position_mapping=[0, 2])
    platform.move(new_timestamp)

    # Define expected platform location after movement
    new_statevector = np.array([[4],
                                [1],
                                [4],
                                [1]])
    assert (np.array_equal(platform.state.state_vector, new_statevector))
    assert np.array_equal(platform.velocity, StateVector([1, 1]))
    assert platform.ndim == 2
    assert platform.is_moving

    # Define zero noise 3d constant velocity transition model
    model_3d = CombinedLinearGaussianTransitionModel(
        [model_1d, model_1d, model_1d])

    # Define a 3d platform with constant velocity motion and test motion
    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [1],
                                     [0],
                                     [1]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=model_3d,
                              position_mapping=[0, 2, 4])
    platform.move(new_timestamp)

    # Define expected platform location in 3d after movement
    new_statevector = np.array([[4],
                                [1],
                                [4],
                                [1],
                                [2],
                                [1]])
    assert (np.array_equal(platform.state.state_vector, new_statevector))
    assert np.array_equal(platform.velocity, StateVector([1, 1, 1]))
    assert platform.ndim == 3
    assert platform.is_moving


def test_velocity_properties():
    model_1d = ConstantVelocity(0.0)
    model_3d = CombinedLinearGaussianTransitionModel(
        [model_1d, model_1d, model_1d])
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    platform_state = State(np.array([[2],
                                     [0],
                                     [2],
                                     [0],
                                     [0],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=model_3d,
                              position_mapping=[0, 2, 4])
    old_position = platform.position
    assert not platform.is_moving
    assert np.array_equal(platform.velocity, StateVector([0, 0, 0]))
    # check it doesn't move with timestamp = None
    platform.move(timestamp=None)
    assert np.array_equal(platform.position, old_position)
    # check it doesn't move (as it has zero velocity)
    platform.move(timestamp)
    assert np.array_equal(platform.position, old_position)
    platform.move(new_timestamp)
    assert np.array_equal(platform.position, old_position)

    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [1],
                                     [0],
                                     [1]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=None,
                              position_mapping=[0, 2, 4])
    assert platform.is_moving
    assert np.array_equal(platform.velocity, StateVector([1, 1, 1]))
    old_position = platform.position
    # check it doesn't move with timestamp = None
    platform.move(None)
    assert np.array_equal(platform.position, old_position)

    with pytest.raises(AttributeError):
        platform.move(timestamp)

    # moving platform without velocity defined
    platform_state = State(np.array([[2],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=None,
                              position_mapping=[0, 2, 4])
    with pytest.raises(AttributeError):
        _ = platform.velocity


orientation_tests_3d = [(StateVector([0, 1, 0, 0, 0, 0]), StateVector([0, 0, 0])),
                        (StateVector([0, 0, 0, 0, 0, 1]), StateVector([0, 0, np.pi/2])),
                        (StateVector([0, 0, 0, 1, 0, 0]), StateVector([0, np.pi/2, 0])),
                        (StateVector([0, 2, 0, 0, 0, 0]), StateVector([0, 0, 0])),
                        (StateVector([0, 0, 0, 0, 0, 2]), StateVector([0, 0, np.pi/2])),
                        (StateVector([0, 0, 0, 2, 0, 0]), StateVector([0, np.pi/2, 0])),
                        (StateVector([0, 1, 0, 0, 0, 1]), StateVector([0, 0, np.pi/4])),
                        (StateVector([0, 0, 0, 1, 0, 1]), StateVector([0, np.pi/2, np.pi/4])),
                        (StateVector([0, 1, 0, 1, 0, 1]),
                            StateVector([0, np.pi/4, np.arctan(1/np.sqrt(2))])),
                        (StateVector([0, -1, 0, 0, 0, 0]), StateVector([0, np.pi, 0])),
                        (StateVector([0, 0, 0, -1, 0, 0]), StateVector([0, -np.pi/2, 0])),
                        (StateVector([0, 0, 0, 0, 0, -1]), StateVector([0, 0, -np.pi/2])),
                        (StateVector([0, -2, 0, 0, 0, 0]), StateVector([0, np.pi, 0])),
                        (StateVector([0, 0, 0, -2, 0, 0]), StateVector([0, -np.pi/2, 0])),
                        (StateVector([0, 0, 0, 0, 0, -2]), StateVector([0, 0, -np.pi/2])),
                        (StateVector([0, -1, 0, 0, 0, -1]), StateVector([0, np.pi, -np.pi/4])),
                        (StateVector([0, 0, 0, -1, 0, -1]), StateVector([0, -np.pi/2, -np.pi/4])),
                        ]


@pytest.mark.parametrize('state, orientation', orientation_tests_3d)
def test_platform_orientation_3d(state, orientation):
    model_1d = ConstantVelocity(0.0)
    model_3d = CombinedLinearGaussianTransitionModel(
        [model_1d, model_1d, model_1d])
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    platform_state = State(state, timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=model_3d,
                              position_mapping=[0, 2, 4])
    assert np.allclose(platform.orientation, orientation)
    # moving with a constant velocity model should not change the orientation
    platform.move(new_timestamp)
    assert np.allclose(platform.orientation, orientation)


orientation_tests_2d = [(StateVector([0, 1, 0, 0]), StateVector([0, 0])),
                        (StateVector([0, 0, 0, 1]), StateVector([0, np.pi/2])),
                        (StateVector([0, 2, 0, 0]), StateVector([0, 0])),
                        (StateVector([0, 0, 0, 2]), StateVector([0, np.pi/2])),
                        (StateVector([0, 1, 0, 1]), StateVector([0, np.pi/4])),
                        (StateVector([0, -1, 0, 0]), StateVector([0, np.pi])),
                        (StateVector([0, 0, 0, -1]), StateVector([0, -np.pi/2])),
                        (StateVector([0, -2, 0, 0]), StateVector([0, np.pi])),
                        (StateVector([0, 0, 0, -2]), StateVector([0, -np.pi/2])),
                        (StateVector([0, -1, 0, -1]), StateVector([0, -3*np.pi/4])),
                        ]


@pytest.mark.parametrize('state, orientation', orientation_tests_2d)
def test_platform_orientation_2d(state, orientation):
    model_1d = ConstantVelocity(0.0)
    model_2d = CombinedLinearGaussianTransitionModel(
        [model_1d, model_1d])
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    platform_state = State(state, timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=model_2d,
                              position_mapping=[0, 2])
    assert np.allclose(platform.orientation, orientation)
    # moving with a constant velocity model should not change the orientation
    platform.move(new_timestamp)
    assert np.allclose(platform.orientation, orientation)


def test_orientation_error():
    # moving platform without velocity defined
    timestamp = datetime.datetime.now()
    platform_state = State(np.array([[2],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=None,
                              position_mapping=[0, 1, 2])
    with pytest.raises(AttributeError):
        _ = platform.orientation
    platform_state = State(np.array([[2],
                                     [0],
                                     [2],
                                     [0],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=None,
                              position_mapping=[0, 2, 4])
    with pytest.raises(AttributeError):
        _ = platform.orientation

    platform_state = State(np.array([[2],
                                     [0],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=None, position_mapping=[0, 2])
    with pytest.raises(AttributeError):
        _ = platform.orientation


# noinspection PyPropertyAccess
def test_setting_position():
    timestamp = datetime.datetime.now()
    platform_state = State(np.array([[2],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=None,
                              position_mapping=[0, 1, 2])
    with pytest.raises(AttributeError):
        platform.position = [0, 0, 0]
    with pytest.raises(AttributeError):
        platform.velocity = [0, 0, 0]

    platform_state = State(np.array([[2],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = FixedPlatform(state=platform_state, position_mapping=[0, 1, 2])
    assert np.array_equal(platform.position, StateVector([2, 2, 0]))
    platform.position = StateVector([0, 0, 0])
    assert np.array_equal(platform.position, StateVector([0, 0, 0]))
    assert np.array_equal(platform.state_vector, StateVector([0, 0, 0]))

    with pytest.raises(AttributeError):
        platform.velocity = [0, 0, 0]

    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [-1],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=None,
                              position_mapping=[0, 1, 2])
    with pytest.raises(AttributeError):
        platform.position = [0, 0]

    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [-1],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = FixedPlatform(state=platform_state, position_mapping=[0, 2, 4])
    assert np.array_equal(platform.position, StateVector([2, 2, 2]))
    platform.position = StateVector([0, 0, 1])
    assert np.array_equal(platform.position, StateVector([0, 0, 1]))
    assert np.array_equal(platform.state_vector, StateVector([0, 1, 0, -1, 1, 0]))

    with pytest.raises(AttributeError):
        platform.velocity = [0, 0, 0]


# noinspection PyPropertyAccess
def test_setting_orientation():
    timestamp = datetime.datetime.now()
    platform_state = State(np.array([[2],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=None,
                              position_mapping=[0, 1, 2])
    with pytest.raises(AttributeError):
        platform.orientation = [0, 0, 0]

    platform_state = State(np.array([[2],
                                     [2],
                                     [0]]),
                           timestamp)
    platform_orientation = StateVector([0, 0, 0])
    platform = FixedPlatform(state=platform_state, position_mapping=[0, 1, 2],
                             orientation=platform_orientation)
    assert np.array_equal(platform.orientation, StateVector([0, 0, 0]))
    platform.orientation = StateVector([0, 1, 0])
    assert np.array_equal(platform.orientation, StateVector([0, 1, 0]))

    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [-1],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=None,
                              position_mapping=[0, 1, 2])
    with pytest.raises(AttributeError):
        platform.orientation = [0, 0]

    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [-1],
                                     [2],
                                     [0]]),
                           timestamp)
    platform_orientation = StateVector([0, 0, 0])
    platform = FixedPlatform(state=platform_state, position_mapping=[0, 2, 4],
                             orientation=platform_orientation)
    assert np.array_equal(platform.orientation, StateVector([0, 0, 0]))
    platform.orientation = StateVector([0, 1, 0])
    assert np.array_equal(platform.orientation, StateVector([0, 1, 0]))


@pytest.mark.parametrize('mapping_type', (tuple, list, np.array))
def test_mapping_types(mapping_type):
    timestamp = datetime.datetime.now()
    platform_state = State(np.array([[2],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = FixedPlatform(state=platform_state, position_mapping=mapping_type([0, 1, 2]))
    assert np.array_equal(platform.position, StateVector([2, 2, 0]))
    platform.position = StateVector([0, 0, 1])
    assert np.array_equal(platform.position, StateVector([0, 0, 1]))

    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [-1],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=None,
                              position_mapping=mapping_type([0, 2, 4]))
    assert np.array_equal(platform.position, StateVector([2, 2, 2]))
    assert np.array_equal(platform.velocity, StateVector([1, -1, 0]))
