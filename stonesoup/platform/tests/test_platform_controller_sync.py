import datetime

from stonesoup.movable import FixedMovable, MovingMovable
from stonesoup.platform import MovingPlatform
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.array import StateVector
from stonesoup.types.state import State


class DummySensor(Sensor):
    @property
    def measurement_model(self):
        raise NotImplementedError

    def measure(self, **kwargs):
        pass


def test_replacing_platform_movement_controller_updates_mounted_sensors():
    timestamp = datetime.datetime.now()
    fixed = FixedMovable(
        states=State(StateVector([2, 2, 0]), timestamp),
        position_mapping=(0, 1, 2),
    )
    moving = MovingMovable(
        states=State(StateVector([2, 1, 2, -1, 2, 0]), timestamp),
        position_mapping=(0, 2, 4),
        transition_model=None,
    )
    platform = MovingPlatform(movement_controller=fixed)
    sensors = [DummySensor(), DummySensor()]

    for sensor in sensors:
        platform.add_sensor(sensor)
        assert sensor.movement_controller is fixed

    platform.movement_controller = moving

    assert platform.movement_controller is moving
    assert all(sensor.movement_controller is moving for sensor in sensors)
