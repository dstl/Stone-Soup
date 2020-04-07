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


def test_sensor_position_setting():
    # TODO
    pass


def test_default_platform():
    sensor = TSensor(position=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 0]))

    sensor = TSensor(orientation=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 0]))


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
