import copy
import datetime

import numpy as np
import pytest

from stonesoup.models.measurement.tests.test_models import position_measurement_sets
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity, KnownTurnRate
from stonesoup.movable import FixedMovable, MovingMovable
from stonesoup.platform.base import Obstacle
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.angle import Bearing
from stonesoup.types.array import StateVector
from stonesoup.platform import MovingPlatform, FixedPlatform, MultiTransitionMovingPlatform
from ...types.state import State
from ...types.groundtruth import GroundTruthPath
from stonesoup.platform.shape import Shape


def test_base():
    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    # Define a static 2d platform and check it does not move
    platform_state2d = State(np.array([[2],
                                       [2]]),
                             timestamp)
    platform = FixedPlatform(states=platform_state2d, position_mapping=np.array([0, 1]))
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
    platform = FixedPlatform(states=platform_state3d, position_mapping=[0, 1, 2])
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
    platform = MovingPlatform(states=platform_state2d, transition_model=model_2d,
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
    platform = MovingPlatform(states=platform_state, transition_model=model_3d,
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


class DummySensor(Sensor):
    @property
    def measurement_model(self):
        raise NotImplementedError

    def measure(self, **kwargs):
        pass


@pytest.mark.parametrize('class_', [FixedPlatform, MovingPlatform])
def test_add_sensor(class_):
    platform_state = State(StateVector([2, 1, 2, 1, 0, 1]), timestamp=datetime.datetime.now())

    platform_args = {'transition_model': None} if class_ is MovingPlatform else {}

    sensor = DummySensor()
    platform = class_(states=platform_state, position_mapping=[0, 2, 4],
                      **platform_args)
    platform.add_sensor(sensor)
    assert len(platform.sensors) == 1 and platform.sensors[0] is sensor

    sensor2 = DummySensor()
    platform.add_sensor(sensor2)
    assert (len(platform.sensors) == 2
            and platform.sensors[0] is sensor
            and platform.sensors[1] is sensor2)


@pytest.mark.parametrize('velocity_mapping', [None, [1, 3, 5]])
def test_velocity_properties(velocity_mapping):
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
    platform = MovingPlatform(states=platform_state, transition_model=model_3d,
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
    platform = MovingPlatform(states=platform_state, transition_model=None,
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
    platform = MovingPlatform(states=platform_state, transition_model=None,
                              position_mapping=[0, 2, 4])
    with pytest.raises(AttributeError):
        _ = platform.velocity

    # pass in a velocity mapping
    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [1],
                                     [0],
                                     [1]]),
                           timestamp)
    platform = MovingPlatform(states=platform_state, transition_model=None,
                              position_mapping=[0, 2, 4], velocity_mapping=velocity_mapping)
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

    platform = MovingPlatform(states=platform_state, transition_model=None,
                              position_mapping=[0, 2, 4], velocity_mapping=velocity_mapping)
    with pytest.raises(AttributeError):
        _ = platform.velocity


def test_orientation_dimensionality_error():
    platform_state = State(StateVector([2, 1, 1, 1, 2, 0, 1, 0]),
                           timestamp=datetime.datetime.now())

    platform = MovingPlatform(states=platform_state, position_mapping=[0, 1, 2, 3],
                              transition_model=None)

    with pytest.raises(NotImplementedError):
        _ = platform.orientation

    platform = MovingPlatform(states=platform_state, position_mapping=[0], transition_model=None)

    with pytest.raises(NotImplementedError):
        _ = platform.orientation


def test_moving_with_no_initial_timestamp():
    timestamp = datetime.datetime.now()
    platform_state = State(StateVector([2, 1, 1, 1, 2, 0]),
                           timestamp=None)

    platform = MovingPlatform(states=platform_state, position_mapping=[0, 2, 4],
                              transition_model=None)

    assert platform.timestamp is None
    platform.move(timestamp=timestamp)
    assert platform.timestamp == timestamp


orientation_tests_3d = [(StateVector([0, 1, 0, 0, 0, 0]), StateVector([0, 0, 0])),
                        (StateVector([0, 0, 0, 0, 0, 1]), StateVector([0, np.pi/2, 0])),
                        (StateVector([0, 0, 0, 1, 0, 0]), StateVector([0, 0, np.pi/2])),
                        (StateVector([0, 2, 0, 0, 0, 0]), StateVector([0, 0, 0])),
                        (StateVector([0, 0, 0, 0, 0, 2]), StateVector([0, np.pi/2, 0])),
                        (StateVector([0, 0, 0, 2, 0, 0]), StateVector([0, 0, np.pi/2])),
                        (StateVector([0, 1, 0, 0, 0, 1]), StateVector([0, np.pi/4, 0])),
                        (StateVector([0, 0, 0, 1, 0, 1]), StateVector([0, np.pi/4, np.pi/2])),
                        (StateVector([0, 1, 0, 1, 0, 1]),
                            StateVector([0, np.arctan(1/np.sqrt(2)), np.pi/4])),
                        (StateVector([0, -1, 0, 0, 0, 0]), StateVector([0, 0, np.pi])),
                        (StateVector([0, 0, 0, -1, 0, 0]), StateVector([0, 0, -np.pi/2])),
                        (StateVector([0, 0, 0, 0, 0, -1]), StateVector([0, -np.pi/2, 0])),
                        (StateVector([0, -2, 0, 0, 0, 0]), StateVector([0, 0, np.pi])),
                        (StateVector([0, 0, 0, -2, 0, 0]), StateVector([0, 0, -np.pi/2])),
                        (StateVector([0, 0, 0, 0, 0, -2]), StateVector([0, -np.pi/2, 0])),
                        (StateVector([0, -1, 0, 0, 0, -1]), StateVector([0, -np.pi/4, np.pi])),
                        (StateVector([0, 0, 0, -1, 0, -1]), StateVector([0, -np.pi/4, -np.pi/2])),
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
    platform = MovingPlatform(states=platform_state, transition_model=model_3d,
                              position_mapping=[0, 2, 4])
    assert np.allclose(platform.orientation, orientation)
    # moving with a constant velocity model should not change the orientation
    platform.move(new_timestamp)
    assert np.allclose(platform.orientation, orientation)


orientation_tests_2d = [(StateVector([0, 1, 0, 0]), StateVector([0, 0, 0])),
                        (StateVector([0, 0, 0, 1]), StateVector([0, 0, np.pi/2])),
                        (StateVector([0, 2, 0, 0]), StateVector([0, 0, 0])),
                        (StateVector([0, 0, 0, 2]), StateVector([0, 0, np.pi/2])),
                        (StateVector([0, 1, 0, 1]), StateVector([0, 0, np.pi/4])),
                        (StateVector([0, -1, 0, 0]), StateVector([0, 0, np.pi])),
                        (StateVector([0, 0, 0, -1]), StateVector([0, 0, -np.pi/2])),
                        (StateVector([0, -2, 0, 0]), StateVector([0, 0, np.pi])),
                        (StateVector([0, 0, 0, -2]), StateVector([0, 0, -np.pi/2])),
                        (StateVector([0, -1, 0, -1]), StateVector([0, 0, -3*np.pi/4])),
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
    platform = MovingPlatform(states=platform_state, transition_model=model_2d,
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
    platform = MovingPlatform(states=platform_state, transition_model=None,
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
    platform = MovingPlatform(states=platform_state, transition_model=None,
                              position_mapping=[0, 2, 4])
    with pytest.raises(AttributeError):
        _ = platform.orientation

    platform_state = State(np.array([[2],
                                     [0],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(states=platform_state,
                              transition_model=None,
                              position_mapping=[0, 2])
    with pytest.raises(AttributeError):
        _ = platform.orientation


# noinspection PyPropertyAccess
def test_setting_position():
    timestamp = datetime.datetime.now()
    platform_state = State(np.array([[2],
                                     [2],
                                     [0]]),
                           timestamp)
    model_1d = ConstantVelocity(0.0)
    model_3d = CombinedLinearGaussianTransitionModel(
        [model_1d, model_1d, model_1d])

    platform = MovingPlatform(states=platform_state, transition_model=model_3d,
                              position_mapping=[0, 1, 2])
    with pytest.raises(AttributeError):
        platform.position = [0, 0, 0]
    with pytest.raises(AttributeError):
        platform.velocity = [0, 0, 0]

    platform_state = State(np.array([[2],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(states=platform_state, transition_model=None,
                              position_mapping=[0, 1, 2])
    assert np.array_equal(platform.position, StateVector([2, 2, 0]))
    platform.position = StateVector([0, 0, 0])
    assert np.array_equal(platform.position, StateVector([0, 0, 0]))
    with pytest.raises(AttributeError):
        platform.velocity = [0, 0, 0]

    platform_state = State(np.array([[2],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = FixedPlatform(states=platform_state, position_mapping=[0, 1, 2])
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
    platform = MovingPlatform(states=platform_state, transition_model=model_3d,
                              position_mapping=[0, 2, 4])
    with pytest.raises(AttributeError):
        platform.position = [0, 0, 0]

    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [-1],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = FixedPlatform(states=platform_state, position_mapping=[0, 2, 4])
    assert np.array_equal(platform.position, StateVector([2, 2, 2]))
    platform.position = StateVector([0, 0, 1])
    assert np.array_equal(platform.position, StateVector([0, 0, 1]))
    assert np.array_equal(platform.state_vector, StateVector([0, 1, 0, -1, 1, 0]))

    with pytest.raises(AttributeError):
        platform.velocity = [0, 0, 0]

    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [-1],
                                     [2],
                                     [0]]),
                           timestamp)
    platform = MovingPlatform(states=platform_state, transition_model=None,
                              position_mapping=[0, 2, 4])
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
    platform = MovingPlatform(states=platform_state, transition_model=None,
                              position_mapping=[0, 1, 2])
    with pytest.raises(AttributeError):
        platform.orientation = [0, 0, 0]

    platform_state = State(np.array([[2],
                                     [2],
                                     [0]]),
                           timestamp)
    platform_orientation = StateVector([0, 0, 0])
    platform = FixedPlatform(states=platform_state, position_mapping=[0, 1, 2],
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
    platform = MovingPlatform(states=platform_state, transition_model=None,
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
    platform = FixedPlatform(states=platform_state, position_mapping=[0, 2, 4],
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
    platform = FixedPlatform(states=platform_state, position_mapping=mapping_type([0, 1, 2]))
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
    platform = MovingPlatform(states=platform_state, transition_model=None,
                              position_mapping=mapping_type([0, 2, 4]))
    assert np.array_equal(platform.position, StateVector([2, 2, 2]))
    assert np.array_equal(platform.velocity, StateVector([1, -1, 0]))


def test_multi_transition():
    transition_model1 = CombinedLinearGaussianTransitionModel(
        (ConstantVelocity(0), ConstantVelocity(0)))
    transition_model2 = KnownTurnRate((0, 0), np.radians(4.5))

    transition_models = [transition_model1, transition_model2]
    transition_times = [datetime.timedelta(seconds=10), datetime.timedelta(seconds=20)]

    platform_state = State(state_vector=[[0], [1], [0], [0]], timestamp=datetime.datetime.now())

    platform = MultiTransitionMovingPlatform(transition_models=transition_models,
                                             transition_times=transition_times,
                                             states=platform_state,
                                             position_mapping=[0, 2],
                                             )

    assert len(platform.transition_models) == 2
    assert len(platform.transition_times) == 2

    #  Check that platform states length increases as platform moves
    assert len(platform.movement_controller) == 1
    time = datetime.datetime.now()
    time += datetime.timedelta(seconds=1)
    platform.move(timestamp=time)
    assert len(platform.movement_controller) == 2
    time += datetime.timedelta(seconds=1)
    platform.move(timestamp=time)
    assert len(platform.movement_controller) == 3
    time += datetime.timedelta(seconds=1)
    platform.move(timestamp=time)
    assert len(platform.movement_controller) == 4

    px, py = platform.position[0], platform.position[1]

    # Starting transition model is index 0
    assert platform.transition_index == 0

    time += datetime.timedelta(seconds=7)
    platform.move(timestamp=time)
    assert len(platform.movement_controller) == 5
    x, y = platform.position[0], platform.position[1]
    # Platform initially moves horizontally
    assert x > px
    assert np.allclose(y, py, atol=1e-6)
    px, py = x, y
    # Transition model changes after corresponding interval is done/ Next transition is left-turn
    assert platform.transition_index == 1

    time += datetime.timedelta(seconds=10)
    platform.move(timestamp=time)
    assert len(platform.movement_controller) == 6
    x, y = platform.position[0], platform.position[1]
    # Platform starts turning left to 45 degrees
    assert x > px
    assert y > py
    # Transition interval is not done. Next transition is left-turn
    assert platform.transition_index == 1

    time += datetime.timedelta(seconds=10)
    platform.move(timestamp=time)
    assert len(platform.movement_controller) == 7
    x, y = platform.position[0], platform.position[1]
    px, py = x, y
    # Platform turned left to 90 degrees
    # Transition interval is done. Next transition is straight-on
    assert platform.transition_index == 0

    time += datetime.timedelta(seconds=10)
    platform.move(timestamp=time)
    assert len(platform.movement_controller) == 8
    x, y = platform.position[0], platform.position[1]
    # Platform travelling vertically up
    assert np.allclose(x, px, atol=1e-6)
    assert y > py
    # Next transition is left-turn
    assert platform.transition_index == 1

    # Add new transition model (right-turn) to list
    transition_model3 = KnownTurnRate((0, 0), np.radians(-9))
    platform.transition_models.append(transition_model3)
    platform.transition_times.append(datetime.timedelta(seconds=10))

    # New model and transition interval are added to model list and to interval list
    assert len(platform.transition_models) == 3
    assert len(platform.transition_times) == 3

    time += datetime.timedelta(seconds=20)
    platform.move(timestamp=time)
    assert len(platform.movement_controller) == 9
    # Platform turned left by 90 degrees (now travelling in -x direction)
    px, py = platform.position[0], platform.position[1]
    # Next transition is right-turn
    assert platform.transition_index == 2

    time += datetime.timedelta(seconds=10)
    platform.move(timestamp=time)
    assert len(platform.movement_controller) == 10
    x, y = platform.position[0], platform.position[1]
    px, py = x, y
    # Next transition straight-on, travelling vertically up again
    assert platform.transition_index == 0

    time += datetime.timedelta(seconds=10)
    platform.move(timestamp=time)
    assert len(platform.movement_controller) == 11
    x, y = platform.position[0], platform.position[1]
    # Platform travelled vertically up
    assert np.allclose(x, px, atol=1e-6)
    assert y > py
    # Next transition is left-turn
    assert platform.transition_index == 1


@pytest.mark.parametrize('first_state, second_state, expected_measurement',
                         position_measurement_sets)
def test_range_and_angles_to_other(first_state, second_state, expected_measurement):
    # Note that, due to platform orientation,  range_and_angles_to_other is not symmetric, so we
    # cannot test simple inversion here.
    timestamp = datetime.datetime.now()
    platform1 = MovingPlatform(states=State(first_state, timestamp=timestamp),
                               position_mapping=(0, 2, 4),
                               transition_model=None)
    platform2 = MovingPlatform(states=State(second_state, timestamp=timestamp),
                               position_mapping=(0, 2, 4),
                               transition_model=None)

    range_, azimuth, elevation = platform1.range_and_angles_to_other(platform2)
    # Assert difference close to zero, to handle angle wrapping (-pi == pi)
    delta = np.array([elevation, Bearing(azimuth), range_]) - expected_measurement[0:3]
    assert np.allclose(np.asarray(delta, dtype=np.float64), 0.0)


# @pytest.mark.parametrize('platform_class', [FixedPlatform, MovingPlatform])
def test_setting_movement_controller():
    timestamp = datetime.datetime.now()
    fixed_state = State(np.array([[2],
                                  [2],
                                  [0]]),
                        timestamp)
    fixed = FixedMovable(states=fixed_state, position_mapping=(0, 1, 2))
    platform = MovingPlatform(movement_controller=fixed)
    assert np.array_equal(platform.position, StateVector([2, 2, 0]))
    assert np.array_equal(platform.velocity, StateVector([0, 0, 0]))

    moving_state = State(np.array([[2],
                                   [1],
                                   [2],
                                   [-1],
                                   [2],
                                   [0]]),
                         timestamp)
    moving = MovingMovable(states=moving_state, position_mapping=(0, 2, 4), transition_model=None)
    platform = MovingPlatform(movement_controller=moving)
    assert np.array_equal(platform.position, StateVector([2, 2, 2]))
    assert np.array_equal(platform.velocity, StateVector([1, -1, 0]))


def test_platform_getitem():
    timestamp = datetime.datetime.now()
    state_before = State(np.array([[2],
                                   [1],
                                   [2],
                                   [1],
                                   [0],
                                   [1]]),
                         timestamp)
    cv_model = CombinedLinearGaussianTransitionModel((ConstantVelocity(0),
                                                      ConstantVelocity(0),
                                                      ConstantVelocity(0)))
    platform = MovingPlatform(states=state_before,
                              transition_model=cv_model,
                              position_mapping=[0, 2, 4], velocity_mapping=[1, 3, 5])
    platform.move(timestamp + datetime.timedelta(seconds=1))
    state_after = platform.state
    assert platform[0] is state_before
    assert platform[1] is state_after


def test_ground_truth_path():
    timestamp = datetime.datetime.now()
    state_before = State(np.array([[2], [1], [2], [1], [0], [1]]), timestamp)
    cv_model = CombinedLinearGaussianTransitionModel((ConstantVelocity(0),
                                                      ConstantVelocity(0),
                                                      ConstantVelocity(0)))
    platform = MovingPlatform(states=state_before,
                              transition_model=cv_model,
                              position_mapping=[0, 2, 4], velocity_mapping=[1, 3, 5])

    platform_gtp: GroundTruthPath = platform.ground_truth_path

    # Test the id and states match
    assert platform.id == platform_gtp.id
    assert platform.state is platform_gtp.state

    # Test the platform states are dynamically linked to the ground truth path state
    platform.move(timestamp + datetime.timedelta(seconds=1))
    assert platform.state is platform_gtp.state
    assert platform.states is platform_gtp.states


def test_default_platform_id():
    fixed_state = State(np.array([[2], [2], [0]]), datetime.datetime.now())
    platform1 = FixedPlatform(states=fixed_state, position_mapping=(0, 1, 2))

    # Test `id` is string
    assert isinstance(platform1.id, str)

    # Test string is suitable long to avoid conflict
    assert len(platform1.id) >= 32

    # Test id isn't replicated
    platform2 = FixedPlatform(states=fixed_state, position_mapping=(0, 1, 2))
    assert platform1.id != platform2


test_id_str = "hello"


def test_platform_id_assignment():
    fixed_state = State(np.array([[2], [2], [0]]), datetime.datetime.now())
    fixed_platform = FixedPlatform(states=fixed_state, position_mapping=(0, 1, 2))
    fixed_platform.id = test_id_str
    assert fixed_platform.id == test_id_str


def test_platform_initialised_with_id():
    fixed_state = State(np.array([[2], [2], [0]]), datetime.datetime.now())
    platform = FixedPlatform(id=test_id_str, states=fixed_state, position_mapping=(0, 1, 2))
    assert platform.id == test_id_str


def test_ground_truth_path_story():
    # Set Up
    timestamp = datetime.datetime.now()
    state_before = State(np.array([[2], [1], [2], [1], [0], [1]]), timestamp)
    cv_model = CombinedLinearGaussianTransitionModel((ConstantVelocity(0),
                                                      ConstantVelocity(0),
                                                      ConstantVelocity(0)))
    platform = MovingPlatform(states=state_before,
                              transition_model=cv_model,
                              position_mapping=[0, 2, 4], velocity_mapping=[1, 3, 5])

    def are_equal(gtp1: GroundTruthPath, gtp2: GroundTruthPath):
        return gtp1.id == gtp2.id and gtp1.states == gtp2.states

    # `Platform.ground_truth_path` produces a GroundTruthPath with `id` and `states` matching the
    # platform.
    platform_gtp = platform.ground_truth_path
    assert isinstance(platform_gtp, GroundTruthPath)

    # Generally `Platform.ground_truth_path` produces equal objects
    assert are_equal(platform.ground_truth_path, platform_gtp)

    # This is still true even if the platform moves. This is because they share the same `states`
    # object
    platform.move(timestamp + datetime.timedelta(seconds=1))
    assert are_equal(platform.ground_truth_path, platform_gtp)

    # However they are not the same object
    assert platform.ground_truth_path is not platform_gtp

    # Therefore changing the `id` will result in a different GroundTruthPath
    platform.id = test_id_str
    assert not are_equal(platform.ground_truth_path, platform_gtp)

    # Reset the id. They should now be equal again
    platform.id = platform_gtp.id
    assert are_equal(platform.ground_truth_path, platform_gtp)

    # They will remain equal if the `states` property is appended/altered
    platform_gtp.states.append(State(np.array([[10], [9], [8], [7], [6], [5]]),
                                     timestamp + datetime.timedelta(seconds=2)))
    platform.move(timestamp + datetime.timedelta(seconds=3))
    assert are_equal(platform.ground_truth_path, platform_gtp)

    # However if the `states` property is replaced, they are no longer equal
    platform_gtp.states = []
    assert not are_equal(platform.ground_truth_path, platform_gtp)


@pytest.mark.xfail
def test_setting_movement_controller_sensors():
    timestamp = datetime.datetime.now()
    fixed_state = State(np.array([[2], [2], [0]]),
                        timestamp)
    fixed = FixedMovable(states=fixed_state, position_mapping=(0, 1, 2))
    platform = MovingPlatform(movement_controller=fixed)

    sensor = DummySensor()
    platform.add_sensor(sensor)
    assert platform.movement_controller is sensor.movement_controller

    moving_state = State(np.array([[2], [1], [2], [-1], [2], [0]]), timestamp)
    moving = MovingMovable(states=moving_state, position_mapping=(0, 2, 4), transition_model=None)

    platform.movement_controller = moving

    assert platform.movement_controller is sensor.movement_controller


@pytest.mark.parametrize(
    'shape_data, position1, orientation1, position2, orientation2, simplices',
    [(np.array([[-2, 2, 2, -2], [2, 2, -2, -2]]),  # shape_data
      StateVector([1, 2]),  # position1
      StateVector([[0], [0], [0]]),  # orientation1
      StateVector([3, 4]),  # position2
      StateVector([[0], [0], [np.radians(45)]]),  # orientation2
      (1, 2, 3, 0),  # simplices
      ),
     (np.array([[-2, 2, 2, -2], [2, 2, -2, -2]]),  # shape_data
      StateVector([1, 2]),  # position1
      StateVector([[0], [0], [np.radians(22)]]),  # orientation1
      StateVector([3, 4]),  # position2
      StateVector([[0], [0], [0]]),  # orientation2
      None,  # simplices
      ),
     (np.array([[-2, 0, 2, 2, -2], [2, 3, 2, -2, -2]]),  # shape_data
      StateVector([1, 2]),  # position1
      StateVector([[0], [0], [0]]),  # orientation1
      StateVector([3, 4]),  # position2
      StateVector([[0], [0], [np.radians(45)]]),  # orientation2
      None,  # simplices
      ),
     (None,  # shape_data
      StateVector([1, 2]),  # position1
      StateVector([[0], [0], [0]]),  # orientation1
      StateVector([3, 4]),  # position2
      StateVector([[0], [0], [np.radians(45)]]),  # orientation2
      None,  # simplices
      )
     ],
    ids=['test_defined_simpleces', 'test_undefined_simplices', 'test_5_sides', 'test_no_shape'])
def test_obstacle(shape_data, position1, orientation1, position2, orientation2,
                  simplices):

    if np.any(orientation1 != StateVector([[0], [0], [0]])) and simplices:
        shape = Shape(shape_data=shape_data,
                      simplices=simplices,)
        test_obstacle = Obstacle(states=State(position1),
                                 orientation=orientation1,
                                 position_mapping=(0, 1),
                                 shape=shape)
    elif simplices:
        shape = Shape(shape_data=shape_data,
                      simplices=simplices,)
        test_obstacle = Obstacle(states=State(position1),
                                 position_mapping=(0, 1),
                                 shape=shape)
    elif np.any(orientation1 != StateVector([[0], [0], [0]])):
        shape = Shape(shape_data=shape_data,)
        test_obstacle = Obstacle(states=State(position1),
                                 orientation=orientation1,
                                 position_mapping=(0, 1),
                                 shape=shape)
    elif not np.any(shape_data):
        with pytest.raises(ValueError, match="The 'Obstacle' platform type requires "
                                             "that property 'shape' is defined. Currently 'None'"):
            test_obstacle = Obstacle(states=State(position1),
                                     position_mapping=(0, 1),)
        return

    else:
        shape = Shape(shape_data=shape_data,)
        test_obstacle = Obstacle(states=State(position1),
                                 position_mapping=(0, 1),
                                 shape=shape)

    # Check that shape data is correct
    assert np.all(test_obstacle.shape.shape_data == shape_data)
    # Check that position is correct
    assert np.all(test_obstacle.position == position1)
    # check that orientation is correct
    assert np.all(test_obstacle.orientation == orientation1)

    # check vertices
    rot_m = np.array([[1, 0, 0],
                      [0, np.cos(-orientation1[0, 0]), -np.sin(-orientation1[0, 0])],
                      [0, np.sin(-orientation1[0, 0]), np.cos(-orientation1[0, 0])]]) @ \
        np.array([[np.cos(orientation1[1, 0]), 0, np.sin(orientation1[1, 0])],
                  [0, 1, 0],
                  [-np.sin(orientation1[1, 0]), 0, np.cos(orientation1[1, 0])]]) @ \
        np.array([[np.cos(-orientation1[2, 0]), -np.sin(-orientation1[2, 0]), 0],
                  [np.sin(-orientation1[2, 0]), np.cos(-orientation1[2, 0]), 0],
                  [0, 0, 1]])

    if not simplices:
        simplices = np.roll(np.linspace(0, shape_data.shape[1]-1, shape_data.shape[1]),
                            -1).astype(int)

    true_vertices = rot_m[:2, :2] @ \
        test_obstacle.shape.shape_data[:, simplices] + position1

    edge_index = np.roll(np.linspace(0, len(simplices)-1, len(simplices)), 1).astype(int)
    true_relative_edges = np.array([true_vertices[0, :] - true_vertices[0, edge_index],
                                    true_vertices[1, :] - true_vertices[1, edge_index]])

    assert np.all(test_obstacle.vertices == true_vertices)
    assert np.all(test_obstacle.relative_edges == true_relative_edges)

    # Ensure changing position and/or orientation changes vertices
    test_obstacle.movement_controller.states = State(position2),
    if orientation2 is not None:
        rot_m = np.array([[1, 0, 0],
                          [0, np.cos(-orientation2[0, 0]), -np.sin(-orientation2[0, 0])],
                          [0, np.sin(-orientation2[0, 0]), np.cos(-orientation2[0, 0])]]) @ \
            np.array([[np.cos(orientation2[1, 0]), 0, np.sin(orientation2[1, 0])],
                      [0, 1, 0],
                      [-np.sin(orientation2[1, 0]), 0, np.cos(orientation2[1, 0])]]) @ \
            np.array([[np.cos(-orientation2[2, 0]), -np.sin(-orientation2[2, 0]), 0],
                      [np.sin(-orientation2[2, 0]), np.cos(-orientation2[2, 0]), 0],
                      [0, 0, 1]])
        test_obstacle.orientation = orientation2

    assert np.all(test_obstacle.position == position2)
    if orientation2 is not None:
        assert np.all(test_obstacle.orientation == orientation2)
    else:
        assert np.all(test_obstacle.orientation == orientation1)

    true_vertices2 = rot_m[:2, :2] @ shape_data[:, simplices] + position2
    assert np.all(test_obstacle.vertices == true_vertices2)

    true_relative_edges2 = np.array([true_vertices2[0, :] -
                                     true_vertices2[0, edge_index],
                                    true_vertices2[1, :] -
                                    true_vertices2[1, edge_index]])

    assert np.all(test_obstacle.relative_edges == true_relative_edges2)


@pytest.mark.parametrize(
    'position2, orientation2, mapping2',
    [(StateVector([3, 4]),  # position2
      None,  # orientation2
      None),  # mapping2
     (StateVector([3, 4]),  # position2
      StateVector([[0], [0], [np.radians(22.2)]]),  # orientation2
      None),  # mapping2
     (StateVector([3, 4, 5]),  # position2
      None,  # orientation2
      (1, 2)),  # mapping2
     (StateVector([3, 4, 5]),  # position2
      StateVector([[0], [0], [np.radians(45)]]),  # orientation2
      (1, 2))  # mapping2
     ],
    ids=['from_obs_position_change', 'from_obs_orientation_change', 'from_obs_mapping_change',
         'from_obs_mapping_and_orient_change']
)
def test_from_obstacle(position2, orientation2, mapping2):

    shape_data = np.array([[-2, 2, 2, -2], [2, 2, -2, -2]])
    position = StateVector([1, 2])
    mapping = (0, 1)

    shape = Shape(shape_data=shape_data)
    initial_obstacle = Obstacle(shape=shape,
                                states=State(position),
                                position_mapping=mapping)

    initial_verts = copy.deepcopy(initial_obstacle.vertices)
    initial_relative_edges = copy.deepcopy(initial_obstacle.relative_edges)

    # Generate obstacle from initial_obstacle
    if orientation2 is not None and mapping2 is not None:
        sub_obstacle = Obstacle.from_obstacle(initial_obstacle,
                                              states=State(position2),
                                              orientation=orientation2,
                                              position_mapping=mapping2)
        rot_orient = orientation2
    elif orientation2 is not None:
        sub_obstacle = Obstacle.from_obstacle(initial_obstacle,
                                              states=State(position2),
                                              orientation=orientation2)
        rot_orient = orientation2
    elif mapping2 is not None:
        sub_obstacle = Obstacle.from_obstacle(initial_obstacle,
                                              states=State(position2),
                                              position_mapping=mapping2)
        rot_orient = initial_obstacle.orientation
    else:
        sub_obstacle = Obstacle.from_obstacle(initial_obstacle,
                                              states=State(position2))
        rot_orient = initial_obstacle.orientation

    rot_m = np.array([[1, 0, 0],
                      [0, np.cos(-rot_orient[0, 0]), -np.sin(-rot_orient[0, 0])],
                      [0, np.sin(-rot_orient[0, 0]), np.cos(-rot_orient[0, 0])]]) @ \
        np.array([[np.cos(rot_orient[1, 0]), 0, np.sin(rot_orient[1, 0])],
                  [0, 1, 0],
                  [-np.sin(rot_orient[1, 0]), 0, np.cos(rot_orient[1, 0])]]) @ \
        np.array([[np.cos(-rot_orient[2, 0]), -np.sin(-rot_orient[2, 0]), 0],
                  [np.sin(-rot_orient[2, 0]), np.cos(-rot_orient[2, 0]), 0],
                  [0, 0, 1]])

    true_vertices = rot_m[:2, :2] @ \
        shape_data[:, initial_obstacle.shape.simplices] + \
        position2[mapping2 if mapping2 is not None else mapping, :]
    edge_index = np.roll(np.linspace(0,
                                     len(initial_obstacle.shape.simplices)-1,
                                     len(initial_obstacle.shape.simplices)), 1). astype(int)
    true_relative_edges = np.array([true_vertices[0, :] -
                                    true_vertices[0, edge_index],
                                    true_vertices[1, :] -
                                    true_vertices[1, edge_index]])

    # Check that shape data has not changed
    assert np.all(sub_obstacle.shape.shape_data == initial_obstacle.shape.shape_data)
    # check that changed properties are correct
    if mapping2 is not None:
        assert np.all(sub_obstacle.position == position2[mapping2, :])
        assert np.all(sub_obstacle.position_mapping == mapping2)
    else:
        assert np.all(sub_obstacle.position == position2)
        assert np.all(sub_obstacle.position_mapping == initial_obstacle.position_mapping)

    if orientation2 is not None:
        assert np.all(sub_obstacle.orientation == orientation2)
    else:
        assert np.all(sub_obstacle.orientation == initial_obstacle.orientation)

    # check that vertices are correctly calculated
    assert np.all(sub_obstacle.vertices == true_vertices)
    # check that relative edges are calculated correctly
    assert np.all(sub_obstacle.relative_edges == true_relative_edges)

    # check that initial_obstacle properties have not changed
    assert np.all(initial_obstacle.orientation == StateVector([[0], [0], [0]]))
    assert np.all(initial_obstacle.position == position)
    assert np.all(initial_obstacle.position_mapping == mapping)
    assert np.all(initial_obstacle.vertices == initial_verts)
    assert np.all(initial_obstacle.relative_edges == initial_relative_edges)
