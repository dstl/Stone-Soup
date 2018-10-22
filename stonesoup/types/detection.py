# -*- coding: utf-8 -*-
from ..base import Property
from .state import State, GaussianState


class Detection(State):
    """Detection type"""

    metadata = Property(dict, default=None,
                        doc='Non-standard metadata items for Detections. '
                            'Default to empty dictionary.')

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
