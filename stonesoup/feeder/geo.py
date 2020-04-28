# -*- coding: utf-8 -*-
import warnings
from abc import abstractmethod

import utm
from pymap3d import geodetic2enu, geodetic2ned

from .base import Feeder
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.detection import Detection


class _LLARefConverter(Feeder):
    reference_point = Property(
        (float, float, float), doc="(Long, Lat, Altitude)")
    mapping = Property(
        (float, float, float), default=(0, 1, 2),
        doc="Indexes of long, lat, altitude. Default (0, 1, 2)")

    @property
    @abstractmethod
    def _converter(self):
        raise NotImplementedError

    @BufferedGenerator.generator_method
    def detections_gen(self):
        for time, detections in self.detector:

            new_detections = set()
            for detection in detections:
                new_coord = self._converter(
                    detection.state_vector[self.mapping[1], 0],  # Lat
                    detection.state_vector[self.mapping[0], 0],  # Long
                    detection.state_vector[self.mapping[2], 0],  # Altitude
                    self.reference_point[1],  # Lat
                    self.reference_point[0],  # Long
                    self.reference_point[2],  # Altitude
                )
                state_vector = detection.state_vector.copy()
                state_vector[self.mapping, 0] = new_coord
                new_detections.add(
                    Detection(
                        state_vector,
                        timestamp=detection.timestamp,
                        measurement_model=detection.measurement_model,
                        metadata=detection.metadata))

            yield time, new_detections


class LLAtoENUConverter(_LLARefConverter):
    """Converts Long., Lat., Alt. to East, North, Up coordinate space.

    This replaces Longitude (°), Latitude (°) and Altitude (m) of a
    :class:`~.Detection` with East (m), North (m) and Up (m) coordinate space
    from a defined :attr:`reference_point`.
    """

    @property
    def _converter(self):
        return geodetic2enu


class LLAtoNEDConverter(_LLARefConverter):
    """Converts Long., Lat., Alt. to North, East, Down coordinate space.

    This replaces Longitude (°), Latitude (°) and Altitude (m) of a
    :class:`~.Detection` with North (m), East (m) and Down (m) coordinate space
    from a defined :attr:`reference_point`.
    """

    @property
    def _converter(self):
        return geodetic2ned


class LongLatToUTMConverter(Feeder):
    """Converts long. and lat. to UTM easting and northing.

    This replaces Longitude (°), Latitude (°) of a :class:`~.Detection` with
    East (m), North (m) coordinate space in a Universal Transverse Mercator
    zone.

    Note
    ----
    Whilst this allows for detections to be converted even if they are
    outside of the set zone, the errors will increase the further the
    position is from the zone being used.

    """

    mapping = Property(
        [float, float], default=[0, 1],
        doc="Indexes of long, lat. Default [0, 1]")
    zone_number = Property(
        int, default=None,
        doc="UTM zone number to carry out conversion. Default `None`, where it"
            "will select the zone based on the first detection.")
    northern = Property(
        bool, default=None,
        doc="UTM northern for northern or southern grid. Default `None`, where"
            "it will be base on the first detection.")

    @BufferedGenerator.generator_method
    def detections_gen(self):
        for time, detections in self.detector:

            utm_detections = set()
            for detection in detections:
                easting, northing, zone_num, northern = utm.from_latlon(
                    *detection.state_vector[self.mapping[::-1]],
                    self.zone_number)
                if self.zone_number is None:
                    self.zone_number = zone_num
                if self.northern is None:
                    self.northern = northern >= 'N'
                elif (self.northern and northern < 'N') or (
                            not self.northern and northern >= 'N'):
                    warnings.warn("Detection cannot be converted to UTM zone")
                    continue

                state_vector = detection.state_vector.copy()
                state_vector[self.mapping, 0] = easting, northing
                utm_detections.add(
                    Detection(
                        state_vector,
                        timestamp=detection.timestamp,
                        measurement_model=detection.measurement_model,
                        metadata=detection.metadata))

            yield time, utm_detections
