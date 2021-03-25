import copy
import datetime

import pytest

from stonesoup.sensor.radar.radar import RadarElevationBearingRangeRate
from stonesoup.platform import FixedPlatform
from ..base import PlatformMountable
from ..sensor import Sensor
from ...types.array import StateVector, CovarianceMatrix
from ...types.state import State

import numpy as np


class DummySensor(Sensor):
    def measure(self, **kwargs):
        pass


class DummyBaseSensor(PlatformMountable):
    def measure(self, **kwargs):
        pass


def test_sensor_position_orientation_setting():
    sensor = DummySensor(position=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 0]))
    sensor.position = StateVector([0, 1, 0])
    assert np.array_equal(sensor.position, StateVector([0, 1, 0]))
    sensor.orientation = StateVector([0, 1, 0])
    assert np.array_equal(sensor.orientation, StateVector([0, 1, 0]))

    position = StateVector([0, 0, 1])
    sensor = DummySensor()
    platform_state = State(state_vector=position + 1, timestamp=datetime.datetime.now())
    platform = FixedPlatform(states=platform_state, position_mapping=[0, 1, 2])
    platform.add_sensor(sensor)
    with pytest.raises(AttributeError):
        sensor.position = StateVector([0, 1, 0])
    with pytest.raises(AttributeError):
        sensor.orientation = StateVector([0, 1, 0])


def test_default_platform():
    sensor = DummySensor(position=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 0]))

    sensor = DummySensor(orientation=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 0]))


def test_internal_platform_flag():
    sensor = DummySensor(position=StateVector([0, 0, 1]))
    assert sensor._has_internal_controller

    sensor = DummySensor()
    assert not sensor._has_internal_controller

    sensor = DummyBaseSensor()
    assert not sensor._has_internal_controller


def test_changing_platform_from_default():
    position = StateVector([0, 0, 1])
    sensor = DummySensor(position=StateVector([0, 0, 1]))

    platform_state = State(state_vector=position+1, timestamp=datetime.datetime.now())
    platform = FixedPlatform(states=platform_state, position_mapping=[0, 1, 2])
    with pytest.raises(AttributeError):
        platform.add_sensor(sensor)


@pytest.mark.parametrize('sensor', [DummyBaseSensor, DummySensor])
def test_sensor_measure(sensor):
    # needed for test coverage... Does no harm
    assert sensor().measure() is None


@pytest.fixture
def radar_platform_target():
    pos_mapping = np.array([0, 2, 4])
    vel_mapping = np.array([1, 3, 5])
    noise_covar = CovarianceMatrix(np.eye(4))

    radar = RadarElevationBearingRangeRate(ndim_state=6,
                                           position_mapping=pos_mapping,
                                           velocity_mapping=vel_mapping,
                                           noise_covar=noise_covar)

    # create a platform with the simple radar mounted
    timestamp_init = datetime.datetime.now()
    platform_1_prior_state_vector = StateVector([[5], [0], [0], [0.25], [0], [0]])
    platform_1_state = State(platform_1_prior_state_vector, timestamp_init)

    radar.mounting_offset = StateVector([[0], [0], [0]])
    radar.rotation_offset = StateVector([[0], [0], [0]])

    platform = FixedPlatform(states=platform_1_state,
                             position_mapping=pos_mapping,
                             velocity_mapping=vel_mapping,
                             sensors=[radar])
    target = State(StateVector([[0], [0], [0], [0], [0], [0]]), timestamp_init)

    return radar, platform, target


def test_platform_deepcopy(radar_platform_target):
    # Set up Radar Model
    radar_1, platform_1, target_state = radar_platform_target

    # Make a measurement with a platform
    _ = platform_1.sensors[0].measure({target_state})

    assert platform_1.sensors[0] is radar_1

    # 1st Error - Try and create another platform and make a measurement
    platform_2 = copy.deepcopy(platform_1)

    assert platform_2 is not platform_1
    assert platform_2.sensors[0] is not platform_1.sensors[0]

    assert platform_2.sensors[0].movement_controller is platform_2.movement_controller
    # check no error in measurement
    _ = platform_2.sensors[0].measure({target_state})


def test_sensor_assignment(radar_platform_target):
    radar_1, platform, target_state = radar_platform_target

    # platform_2 = copy.deepcopy(platform)
    radar_2 = copy.deepcopy(radar_1)
    radar_3 = copy.deepcopy(radar_1)

    platform.add_sensor(radar_2)
    platform.add_sensor(radar_3)

    assert len(platform.sensors) == 3
    assert platform.sensors == (radar_1, radar_2, radar_3)

    with pytest.raises(AttributeError):
        platform.sensors = []

    with pytest.raises(TypeError):
        platform.sensors[0] = radar_3

    platform.pop_sensor(1)
    assert len(platform.sensors) == 2
    assert platform.sensors == (radar_1, radar_3)

    platform.add_sensor(radar_2)
    assert len(platform.sensors) == 3
    assert platform.sensors == (radar_1, radar_3, radar_2)

    platform.remove_sensor(radar_3)
    assert len(platform.sensors) == 2
    assert platform.sensors == (radar_1, radar_2)
