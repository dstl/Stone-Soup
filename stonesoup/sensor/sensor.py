from abc import ABC

from stonesoup.platform.simple import FixedSensorPlatform
from stonesoup.sensor.base import BaseSensor
from stonesoup.types.array import StateVector
from stonesoup.types.state import State


class Sensor(BaseSensor, ABC):
    # this functionality requires knowledge of FixedSensorPlatform so cannot go in the BaseSensor
    # class
    def __init__(self, *args, **kwargs):
        position = kwargs.pop('position', None)
        orientation = kwargs.pop('orientation', None)
        self._internal_platform = None
        super().__init__(*args, **kwargs)
        if position is not None or orientation is not None:
            if self.platform_system is not None:
                raise ValueError('Platform system and position/orientation cannot both be '
                                 'specified.')
            if position is None:
                # assuming 3d for a default platform
                position = StateVector([0, 0, 0])
            if orientation is None:
                orientation = StateVector([0, 0, 0])
            self._internal_platform = FixedSensorPlatform(
                state=State(state_vector=position),
                position_mapping=list(range(len(position))),
                orientation=orientation,
                sensors=[self])

    @property
    def _has_internal_platform(self):
        return self._internal_platform is not None

    def _set_platform_system(self, value):
        if self._has_internal_platform:
            raise AttributeError('Platform system cannot be set on sensors that were created with '
                                 'a default platform')
        self._property_platform_system = value
