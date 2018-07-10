# -*- coding: utf-8 -*-
from ..base import Property
from .state import State, GaussianState
from .groundtruth import GroundTruthPath

class Detection(State):
    """Detection type"""


class GaussianDetection(Detection, GaussianState):
    """GaussianDetection type"""


class Clutter(Detection):
    """Clutter type for detections classed as clutter

    This is same as :class:`~.Detection`, but can be used to identify clutter
    for metrics and analysis purposes.
    """

class TrueDetection(Detection):
    """TrueDetection tupe for detections that come from ground truth tracks

    This is same as :class:`~.Detection`, but can be used to identify true detections
    for metrics and analysis purposes.
     """

    groundtruth_path = Property(GroundTruthPath,
                                doc = 'Ground truth path that this detection came from')
