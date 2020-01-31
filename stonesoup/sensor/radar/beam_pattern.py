# -*- coding: utf-8 -*-
import datetime

from ...base import Property, Base


class BeamTransitionModel(Base):
    """Base class for Beam Transition Model"""
    centre = Property(list, doc="Centre of the beam pattern")

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
        : `list` of 2 :class:`floats`, [azimuth, elevation] from the
            perspective of the radar
        """
        return self.centre


class BeamSweep(BeamTransitionModel):
    """This describes a beam moving in a raster pattern"""
    init_time = Property(datetime.datetime,default=None, doc="The time the frame is"
                                                " started")

    angle_per_s = Property(float, doc="The speed that the beam scans at")
    frame = Property(list, doc="Dimensions of search frame as "
                               "[azimuth,elevation]")
    separation = Property(float, doc="Separation of lines in elevation")
    centre = Property(list, default=[0, 0], doc="Centre of the search frame in"
                                                " [azimuth,elevation]")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        :class: `list`
        returns the [azimuth, elevation] of beam at given timestamp
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
        return [az, el]
