# -*- coding: utf-8 -*-
from operator import attrgetter

from ..base import Property
from .base import Feeder


class MetadataReducer(Feeder):
    """Reduce detections so unique metadata value present at each time step.

    This allows to reduce detections so a single detection is returned, based
    on a particular metadata value, for example a unique identity. The most
    recent detection will be yielded for each unique metadata value at each
    time step.
    """

    metadata_field = Property(
        str,
        doc="Field used to reduce unique set of detections")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = set()

    @property
    def detections(self):
        return self._detections

    def detections_gen(self):
        for time, detections in self.detector.detections_gen():
            unique_detections = set()
            sorted_detections = sorted(
                detections, key=attrgetter('timestamp'), reverse=True)
            meta_values = set()
            for detection in sorted_detections:
                meta_value = detection.metadata.get(self.metadata_field)
                if meta_value not in meta_values:
                    unique_detections.add(detection)
                    # Ignore those without meta data value
                    if meta_value is not None:
                        meta_values.add(meta_value)
                self._detections = unique_detections
            yield time, unique_detections
