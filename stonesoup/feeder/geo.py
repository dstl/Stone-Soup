# -*- coding: utf-8 -*-
from pymap3d import geodetic2enu

from .base import Feeder
from ..base import Property
from ..types.detection import Detection


class LLAtoENUConverter(Feeder):
    """Converts Long., Lat., Alt. to East, North, Up coordinate space.

    This replaces Longitude (°), Latitude (°) and Altitude (m) of a
    :class:`~.Detection` with East (m), North (m) and Up (m) coordinate space
    from a defined :attr:`reference_point`.
    """
    reference_point = Property(
        (float, float, float), doc="(Long, Lat, Altitude)")
    mapping = Property(
        (float, float, float), default=(0, 1, 2),
        doc="Indexes of long, lat, altitude. Default (0, 1, 2)")

    @property
    def detections(self):
        return self._detections.copy()

    def detections_gen(self):
        for time, detections in self.detector.detections_gen():

            self._detections = set()
            for detection in detections:
                enu = geodetic2enu(
                    detection.state_vector[self.mapping[1], 0],  # Lat
                    detection.state_vector[self.mapping[0], 0],  # Long
                    detection.state_vector[self.mapping[2], 0],  # Altitude
                    self.reference_point[1],  # Lat
                    self.reference_point[0],  # Long
                    self.reference_point[2],  # Altitude
                )
                state_vector = detection.state_vector.copy()
                state_vector[self.mapping, 0] = enu
                self._detections.add(
                    Detection(state_vector, timestamp=detection.timestamp))

            yield time, self.detections
