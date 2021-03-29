# -*- coding: utf-8 -*-
import datetime
from typing import MutableMapping, Sequence

import numpy as np

from stonesoup.sensor.sensor import Sensor
from .groundtruth import GroundTruthPath
from .state import State, GaussianState, StateVector, CompositeState
from ..base import Property
from ..models.measurement import MeasurementModel


class Detection(State):
    """Detection type"""
    measurement_model: MeasurementModel = Property(
        default=None,
        doc="The measurement model used to generate the detection (the default is ``None``)")

    metadata: MutableMapping = Property(
        default=None, doc='Dictionary of metadata items for Detections.')

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


class TrueDetection(Detection):
    """TrueDetection type for detections that come from ground truth

    This is same as :class:`~.Detection`, but can be used to identify true
    detections for metrics and analysis purposes.
    """

    groundtruth_path: GroundTruthPath = Property(
        doc="Ground truth path that this detection came from")


class MissedDetection(Detection):
    """Detection type for a missed detection

    This is same as :class:`~.Detection`, but it is used in
    MultipleHypothesis to indicate the null hypothesis (no
    detections are associated with the specified track).
    """

    state_vector: StateVector = Property(default=None, doc="State vector. Default `None`.")

    def __init__(self, state_vector=None, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)

    def __bool__(self):
        return False


class CompositeDetection(CompositeState):
    inner_states: Sequence[Detection] = Property(default=None,
                                                 doc="Sequence of detections comprising the "
                                                     "composite detection.")
    groundtruth_path: GroundTruthPath = Property(default=None,
        doc="Ground truth path that this detection came from")
    sensor: Sensor = Property(default=None, doc="Sensor that generated the detection.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def mapping(self):

        if self.sensor is None:
            return np.arange(len(self.inner_states))
        return self.sensor.mapping

    @property
    def metadata(self):
        metadata = dict()
        for sub_detection in self.inner_states:
            metadata.update(sub_detection.metadata)
        return metadata


class CompositeMissedDetection(CompositeDetection):

    def __bool__(self):
        return False
