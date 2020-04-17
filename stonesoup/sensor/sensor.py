from abc import ABC

from ..platform.base import FixedPlatform
from .base import BaseSensor
from ..types.array import StateVector
from ..types.state import State


class Sensor(BaseSensor, ABC):
    """Sensor Base class for general use.

    Most properties and methods are inherited from :class:`~.BaseSensor`, but this class includes
    crucial functionality and so should be used in preference.

    All sensors must be mounted on a platform to calculate their position and orientation. To
    make this easier, if the sensor has a position and/or orientation specified in the constructor,
    and no :attr:`platform_system`, then the default is to create an internally held "private"
    platform for the Sensor. This restricts the later setting of the :attr:`platform_system` but
    does allow the Sensor to control (and set) its own position and orientation.
    """
    # this functionality requires knowledge of FixedPlatform so cannot go in the BaseSensor
    # class
    def __init__(self, *args, **kwargs):
        position = kwargs.pop('position', None)
        orientation = kwargs.pop('orientation', None)
        self._internal_platform = None
        super().__init__(*args, **kwargs)
        if position is not None or orientation is not None:
            if position is None:
                # assuming 3d for a default platform
                position = StateVector([0, 0, 0])
            if orientation is None:
                orientation = StateVector([0, 0, 0])
            self._internal_platform = FixedPlatform(
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
        super()._set_platform_system(value)
