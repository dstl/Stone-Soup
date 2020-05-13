import datetime

import pytest

from ...platform.base import FixedPlatform
from ..base import BaseSensor
from ..sensor import Sensor
from ...types.array import StateVector
from ...types.state import State

import numpy as np


class DummySensor(Sensor):
    def measure(self, **kwargs):
        pass


class DummyBaseSensor(BaseSensor):
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


def test_warning_on_moving_sensor():
    sensor = DummySensor()
    platform_state = State(StateVector([0, 1, 0]), timestamp=datetime.datetime.now())
    platform1 = FixedPlatform(states=platform_state, position_mapping=[0, 1, 2])
    platform2 = FixedPlatform(states=platform_state, position_mapping=[0, 1, 2])
    platform1.add_sensor(sensor)
    with pytest.warns(UserWarning):
        platform2.add_sensor(sensor)


def test_default_platform():
    sensor = DummySensor(position=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 0]))

    sensor = DummySensor(orientation=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 0]))


def test_internal_platform_flag():
    sensor = DummySensor(position=StateVector([0, 0, 1]))
    assert sensor._has_internal_platform

    sensor = DummySensor()
    assert not sensor._has_internal_platform

    sensor = DummyBaseSensor()
    assert not sensor._has_internal_platform


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
