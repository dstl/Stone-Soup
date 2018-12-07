# -*- coding: utf-8 -*-
from ..base import Property
from .state import State, GaussianState


class Detection(State):
    """Detection type"""

    metadata = Property(dict, default=None,
                        doc='Dictionary of metadata items for Detections.')

    def __init__(self, state_vector, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)
        if self.metadata is None:
            self.metadata = {}


class GaussianDetection(Detection, GaussianState):
    """GaussianDetection type"""


class Clutter(Detection):
    """Clutter type for detections classed as clutter

    This is same as :class:`~.Detection`, but can be used to identify clutter
    for metrics and analysis purposes.
    """


class MissedDetection(Detection):
    """Detection type for a missed detection

    This is same as :class:`~.Detection`, but it is used in
    MultipleMeasurementHypothesis to indicate the null hypothesis (no
    detections are associated with the specified track).
    """

    def __init__(self, state_vector=[[0], [0]], timestamp=0, metadata=None):
        super().__init__(state_vector=state_vector, timestamp=timestamp,
                         metadata={})
