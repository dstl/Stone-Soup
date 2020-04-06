# -*- coding: utf-8 -*-
import copy
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
        (float, float), default=(0, 1),
        doc="Indexes of long, lat. Default (0, 1)")
    zone_number = Property(
        int, default=None,
        doc="UTM zone number to carry out conversion. Default `None`, where it will select the "
            "zone based on the first detection if :attr:`dynamic` isn't set.")
    zone_letter = Property(
        str, default=None,
        doc="UTM zone letter to carry out conversion. Default `None`, where it will select the "
            "zone based on the first detection if :attr:`dynamic` isn't set.")
    dynamic = Property(
        bool, default=False,
        doc="In dynamic mode, each detection will be put into the UTM zone that it belongs, rather"
            "than being assigned to the same zone. The `utm_zone` will be added to the detections"
            "metadata. Default `False`.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dynamic and (self.zone_number or self.zone_letter):
            raise ValueError("Cannot have both dynamic and fixed zone")

    @BufferedGenerator.generator_method
    def detections_gen(self):
        for time, detections in self.detector:

            utm_detections = set()
            for detection in detections:
                easting, northing, zone_number, zone_letter = utm.from_latlon(
                    *detection.state_vector[self.mapping[::-1], :],
                    self.zone_number, self.zone_letter)
                if self.zone_number is None and not self.dynamic:
                    self.zone_number = zone_number
                if self.zone_letter is None and not self.dynamic:
                    self.zone_letter = zone_letter

                utm_detection = copy.copy(detection)
                # Update state vector
                utm_detection.state_vector = detection.state_vector.copy()
                utm_detection.state_vector[self.mapping, 0] = easting, northing
                # Add meta data
                utm_detection.metadata = detection.metadata.copy()
                utm_detection.metadata['utm_zone'] = (zone_number, zone_letter)

                utm_detections.add(utm_detection)

            yield time, utm_detections
