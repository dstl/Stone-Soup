# -*- coding: utf-8 -*-
import numpy as np

from ..base import Property
from ..types import StateVector
from ..sensor import Sensor
from .base import Platform


class SensorPlatform(Platform):
    """A simple Platform that can carry a number of different sensors

    Notes
    -----
    The current implementation of this class assumes an 2D Cartesian plane.

    """

    # TODO: Offset sensors relative to platform orientation

    # TODO: Determine where a platform coordinate frame should be maintained

    sensors = Property([Sensor], doc="A list of mounted sensors")
    mounting_offsets = Property(
        [np.array], doc="A list of sensor offsets (For now expressed\
                            as a nxN array of nD Cartesian coordinates)")
    # TODO: determine if mappings are per-sensor or per-platform
    mounting_mappings = Property(
        [np.array], doc="Mappings between the platform state vector and each\
                            individuals sensors mouting offset")
    coordinate_reference_system = Property(str, default="NEU",
                                           doc="Coordinate reference system,\
                                           default is North, East Up")

    def __init__(self, *args, **kwargs):
        """
        Ensure that the platform location and the sensor locations are
        consistent at initialisation.
        """
        super().__init__(*args, **kwargs)
        self._move_sensors()

    # TODO: create add_sensor method

    def move(self, timestamp=None, **kwargs):
        """Propagate the platform position using the :attr:`transition_model`,\
        and use _move_sensors method to update sensor positions

        Parameters
        ----------
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the maneuver completes \
            (the default is `None`)

        """

        # Call superclass method to update platform state
        super().move(timestamp=timestamp, **kwargs)
        # Move the platforms sensors
        self._move_sensors()

    def _move_sensors(self):
        """  Propogste the Sensor positions based upon the mounting
        offsets and the platform position and heading post manoeuvre.
        TODO -  handle heading information
        x' = x.cos[theta] + y.sin[theta]
        y' = y.cos[theta] + x.sin[theta]
        z' =
        """
        # Update the positions of all sensors relative to the platform
        for i in range(len(self.sensors)):
            new_sensor_pos = np.zeros([self.mounting_offsets.shape[1], 1])
            for j in range(self.mounting_offsets.shape[1]):
                new_sensor_pos[j, 0] = (self.state.state_vector
                                        [self.mounting_mappings[i, j]]
                                        + self.mounting_offsets[i, j])
            self.sensors[i].set_position(StateVector(new_sensor_pos))
