# -*- coding: utf-8 -*-
import numpy as np

from ..base import Property
from ..types import StateVector
from ..sensor import Sensor
from .base import Platform


class SensorPlatform(Platform):
    """A simple Platform that can carry a number of different sensors

    """

    sensors = Property([Sensor], doc="A list of mounted sensors")
    mounting_offsets = Property(
        [np.array], doc="A list of sensor offsets (For now expressed\
                            as a 2xN array of 2D Cartesian coordinates)")
    mounting_mappings = Property(
        [np.array], doc="Mappings between the platform state vector and the\
                         respective sensor positions")

    def move(self, timestamp=None, **kwargs):
        """Propagate the platform position using the :attr:`transition_model`,\
        while updating the positions of all mounted sensors.

        Parameters
        ----------
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the end of the maneuver \
            (the default is `None`)

        """

        # Call superclass method to update platform state
        super().move(timestamp=timestamp, **kwargs)

        # Update the positions of all sensors
        for i in range(0, len(self.sensors)):
            self.sensors[i].set_position(StateVector(
                [[self.state.state_vector[self.mounting_mappings[0][i]][0]
                  + self.mounting_offsets[i][0]],
                 [self.state.state_vector[self.mounting_mappings[1][i]][0]
                  + self.mounting_offsets[i][0]]]))
