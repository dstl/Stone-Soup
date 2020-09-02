# -*- coding: utf-8 -*-
import datetime
from typing import Sequence, Tuple

from ...base import Property, Base


class BeamTransitionModel(Base):
    """Base class for Beam Transition Model"""
    centre: Sequence = Property(doc="Centre of the beam pattern")

    def move_beam(self, timestamp, **kwargs):
        """Gives position of beam at given time"""
        raise NotImplementedError


class StationaryBeam(BeamTransitionModel):
    """Stationary beam that points in the direction of centre"""
    def move_beam(self, *args, **kwargs):
        """ generates a beam position based on the centre

        Parameters
        ----------

        Returns
        -------
        azimuth : `float`
            azimuth from the perspective of the radar at given timestamp
        elevation : `float`
            elevation from the perspective of the radar at given timestamp
        """
        return self.centre


class BeamSweep(BeamTransitionModel):
    """This describes a beam moving in a raster pattern"""
    init_time: datetime.datetime = Property(default=None, doc="The time the frame is started")
    angle_per_s: float = Property(doc="The speed that the beam scans at")
    frame: Tuple[float, float] = Property(doc="Dimensions of search frame as [azimuth,elevation]")
    separation: float = Property(doc="Separation of lines in elevation")
    centre: Tuple[float, float] = Property(
        default=None,
        doc="Centre of the search frame in [azimuth,elevation]. Defaults to [0, 0]")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.centre is None:
            self.centre = (0, 0)

    @property
    def length_frame(self):
        return self.frame[0] * (self.frame[1] / self.separation + 1)

    @property
    def frame_time(self):
        return self.length_frame / self.angle_per_s

    def move_beam(self, timestamp, **kwargs):
        """Returns the position of the beam at given timestamp

        Parameters
        ----------
        timestamp : `datetime.datetime`
            current timestep

        Returns
        -------
        azimuth : `float`
            azimuth from the perspective of the radar at given timestamp
        elevation : `float`
            elevation from the perspective of the radar at given timestamp
        """
        if self.init_time is None:
            self.init_time = timestamp
        time_diff = timestamp - self.init_time
        # distance into a frame
        total_angle = (time_diff.total_seconds() * self.angle_per_s) % \
            self.length_frame
        # the row the beam should be in
        row = int(total_angle / (self.frame[0]))
        # how far the beam is into the the current row
        col = total_angle - row*self.frame[0]
        # start from left or right?
        if row % 2 == 0:
            az = col
        else:
            az = self.frame[0] - col
        # azimuth position
        az = az - self.frame[0]/2 + self.centre[0]
        # elevation position
        el = self.frame[1]/2 + self.centre[1] - row*self.separation
        return az, el
