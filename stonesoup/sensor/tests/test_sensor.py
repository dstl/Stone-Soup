import datetime

import pytest

from stonesoup.platform.simple import FixedSensorPlatform
from stonesoup.sensor.base import BaseSensor
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.array import StateVector
import numpy as np

from stonesoup.types.state import State


class TSensor(Sensor):
    def measure(self, **kwargs):
        pass


class TBaseSensor(BaseSensor):
    def measure(self, **kwargs):
        pass


def test_sensor_position_orientation_setting():
    sensor = TSensor(position=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 0]))
    sensor.position = StateVector([0, 1, 0])
    assert np.array_equal(sensor.position, StateVector([0, 1, 0]))
    sensor.orientation = StateVector([0, 1, 0])
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 0]))

    position = StateVector([0, 0, 1])
    sensor = TSensor()
    platform_state = State(state_vector=position + 1, timestamp=datetime.datetime.now())
    platform = FixedSensorPlatform(state=platform_state, mapping=[0, 1, 2])
    platform.add_sensor(sensor)
    with pytest.raises(AttributeError):
        sensor.position = StateVector([0, 1, 0])
    with pytest.raises(AttributeError):
        sensor.orientation = StateVector([0, 1, 0])


def test_default_platform():
    sensor = TSensor(position=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 0]))

    sensor = TSensor(orientation=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 0]))


def test_internal_platform_flag():
    sensor = TSensor(position=StateVector([0, 0, 1]))
    assert sensor._has_internal_platform

    sensor = TSensor()
    assert not sensor._has_internal_platform

    sensor = TBaseSensor()
    assert not sensor._has_internal_platform


def test_changing_platform_from_default():
    position = StateVector([0, 0, 1])
    sensor = TSensor(position=StateVector([0, 0, 1]))

    platform_state = State(state_vector=position+1, timestamp=datetime.datetime.now())
    platform = FixedSensorPlatform(state=platform_state, mapping=[0, 1, 2])
    with pytest.raises(AttributeError):
        platform.add_sensor(sensor)


@pytest.mark.parametrize('sensor', [TBaseSensor, TSensor])
def test_sensor_measure(sensor):
    # needed for test coverage... Does no harm
    assert sensor().measure() is None
