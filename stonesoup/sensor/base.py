# -*- coding: utf-8 -*-
import weakref
from abc import abstractmethod, ABC
from typing import Optional, Set, Union
from warnings import warn

import numpy as np

from ..base import Base
from ..platform import Platform
from ..types.array import StateVector
from ..types.detection import TrueDetection
from ..types.groundtruth import GroundTruthState


class BaseSensor(Base, ABC):
    """Sensor base class

    .. warning::
        This class is private and should not be used or subclassed directly. Instead use the
        :class:`~.Sensor` class which is needed to achieve the functionality described in this
        class's documentation.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._platform_system = None

    @property
    def platform(self) -> Optional[Platform]:
        """Return the platform system to which the sensor is attached. Resolves the ``weakref``
        stored in the :attr:`platform_system` Property."""
        if self.platform_system is None:
            return None
        else:
            return self.platform_system()

    @property
    def platform_system(self) -> Optional[weakref.ref]:
        """Return a ``weakref`` to the platform on which the sensor is mounted"""
        return self._platform_system

    @platform_system.setter
    def platform_system(self, value: weakref.ref):
        # this slightly odd construction is to allow overriding by the Sensor subclass
        self._set_platform_system(value)

    def _set_platform_system(self, value: weakref.ref):
        if self._platform_system is not None:
            warn('Sensor has been moved from one platform to another. This is unexpected '
                 'behaviour')
        self._platform_system = value

    @abstractmethod
    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:
        """Generate a measurement for a given state

        Parameters
        ----------
        ground_truths : Set[:class:`~.GroundTruthState`]
            A set of :class:`~.GroundTruthState`
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is `True`, in which
            case :meth:`~.Model.rvs` is used; if `False`, no noise will be added)

        Returns
        -------
        Set[:class:`~.TrueDetection`]
            A set of measurements generated from the given states. The timestamps of the
            measurements are set equal to that of the corresponding states that they were
            calculated from. Each measurement stores the ground truth path that it was produced
            from.
        """
        raise NotImplementedError

    @property
    def position(self) -> Optional[StateVector]:
        """The sensor position on a 3D Cartesian plane, expressed as a 3x1 :class:`StateVector`
        of Cartesian coordinates in the order :math:`x,y,z`.

        .. note::
            This property delegates the actual calculation of position to the platform on which the
            sensor is mounted.

            It is settable if, and only if, the sensor holds its own internal platform."""
        if self.platform is None:
            return None
        return self.platform.get_sensor_position(self)

    @position.setter
    def position(self, value: StateVector):
        if self._has_internal_platform:
            self.platform.position = value
        else:
            raise AttributeError('Cannot set sensor position unless the sensor has its own '
                                 'default platform')

    @property
    def orientation(self) -> Optional[StateVector]:
        """A 3x1 StateVector of angles (rad), specifying the sensor orientation in terms of the
        counter-clockwise rotation around each Cartesian axis in the order :math:`x,y,z`.
        The rotation angles are positive if the rotation is in the counter-clockwise
        direction when viewed by an observer looking along the respective rotation axis,
        towards the origin.

        .. note::
            This property delegates the actual calculation of orientation to the platform on which
            the sensor is mounted.

            It is settable if, and only if, the sensor holds its own internal platform."""
        if self.platform is None:
            return None
        return self.platform.get_sensor_orientation(self)

    @orientation.setter
    def orientation(self, value: StateVector):
        if self._has_internal_platform:
            self.platform.orientation = value
        else:
            raise AttributeError('Cannot set sensor position unless the sensor has its own '
                                 'default platform')

    @property
    def _has_internal_platform(self) -> bool:
        return False

    @property
    def velocity(self) -> Optional[StateVector]:
        """The sensor velocity on a 3D Cartesian plane, expressed as a 3x1 :class:`StateVector`
        of Cartesian coordinates in the order :math:`x,y,z`.

        .. note::
            This property delegates the actual calculation of velocity to the platform on which the
            sensor is mounted.

            It is settable if, and only if, the sensor holds its own internal platform which is
            a MovingPlatfom."""
        return self.platform.velocity
