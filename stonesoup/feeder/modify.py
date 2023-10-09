import copy
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Tuple, List

import numpy as np

from .base import DetectionFeeder, Feeder, DetectionOutput
from .simple import IterFeeder
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..models.base import ReversibleModel
from ..models.measurement.linear import LinearGaussian
from ..models.measurement.nonlinear import CartesianToElevationBearing, \
    CartesianToElevationBearingRange
from ..types.array import StateVector
from ..types.detection import Detection
from ..types.state import State


class BaseModifiedFeederIter(IterFeeder):
    def __iter__(self):
        self.reader_iter = iter(self.reader)
        return self

    def __next__(self):
        reader_output = next(self.reader_iter)
        return self.alter_output(reader_output)

    @abstractmethod
    def alter_output(self, reader_output: Tuple[datetime, set]) -> Tuple[datetime, set]:
        ...


class BaseModifiedFeeder(Feeder):

    @BufferedGenerator.generator_method
    def data_gen(self):
        for reader_output in self.reader:
            yield self.alter_output(reader_output)
        return

    @abstractmethod
    def alter_output(self, reader_output: Tuple[datetime, set]) -> Tuple[datetime, set]:
        ...


class CopyFeeder(BaseModifiedFeeder):
    """
    Takes a copy of each object in the set and yields them. This is useful if you want to edit the
    object in the feeder later on but don't want to edit the original object
    """
    def alter_output(self, reader_output: Tuple[datetime, set]) -> Tuple[datetime, set]:
        time, set_of_items = reader_output
        copied_items = {copy.copy(item) for item in set_of_items}
        return time, copied_items


class DelayedFeeder(BaseModifiedFeeder):

    delay: timedelta = Property()

    def alter_output(self, reader_output: Tuple[datetime, set]) -> Tuple[datetime, set]:
        time, states = reader_output
        return time + self.delay, states


class AlteredDetectionFeeder(BaseModifiedFeeder, ABC):

    reader: DetectionFeeder = Property(doc="Source of detections")

    @abstractmethod
    def alter_detection(self, a_detection: Detection) -> Detection:
        raise NotImplementedError

    def alter_output(self, reader_output: DetectionOutput) -> DetectionOutput:
        time, detections = reader_output
        new_detections = {self.alter_detection(a_detection) for a_detection in detections}

        return time, new_detections


class OriginalStateDetectionSpaceFeeder(AlteredDetectionFeeder):

    measurement_model_key = "original measurement model"

    ndim_state: int = Property(
        default=None, doc="Number of state dimensions position and velocity for detection state "
                          "space. Using None, takes this from the measurement model")
    mapping: List[int] = Property(default=None, doc="Index of position components in the state")

    def alter_detection(self, a_detection: Detection) -> Detection:
        new_measurement_model = LinearGaussian(
            # Number of state dimensions (position and velocity in 3D)
            ndim_state=self.ndim_state or a_detection.measurement_model.ndim_state,
            # Mapping measurement vector index to state index
            mapping=self.mapping or a_detection.measurement_model.mapping,
            noise_covar=a_detection.measurement_model.noise_covar
        )  # Covariance matrix for Gaussian PDF
        new_metadata = a_detection.metadata.copy()
        new_metadata[self.measurement_model_key] = a_detection.measurement_model

        new_det = a_detection.from_state(a_detection,
                                         measurement_model=new_measurement_model,
                                         metadata=new_metadata)
        # noinspection PyTypeChecker
        return new_det


class OriginalStateDetectionSpaceFeeder2D(OriginalStateDetectionSpaceFeeder):

    ndim_state: int = Property(
        default=4, doc="Number of state dimensions position and velocity for detection state "
                       "space. Using None, takes this from the measurement model")
    mapping: List[int] = Property(default=(0, 2), doc="Index of position components in the state")


class DegradeDetectionConfidenceFeeder(AlteredDetectionFeeder):

    degrade_amount: float = Property()

    def alter_detection(self, a_detection: Detection) -> Detection:

        new_measurement_model = copy.deepcopy(a_detection.measurement_model)
        new_measurement_model.noise_covar *= self.degrade_amount
        # noinspection PyTypeChecker
        new_det: Detection = a_detection.from_state(a_detection,
                                                    measurement_model=new_measurement_model)
        return new_det


class StaticRotationalFrameAngleDetectionFeeder(AlteredDetectionFeeder):
    static_rotation_offset: StateVector = Property(default=StateVector([[0], [0], [np.pi/2]]))

    def alter_detection(self, a_detection: Detection) -> Detection:
        return self.rotate_detection(a_detection, self.static_rotation_offset)

    @staticmethod
    def rotate_detection(detection: Detection, rotation_offset: StateVector) -> Detection:
        measurement_model = detection.measurement_model
        new_measurement_model = copy.deepcopy(measurement_model)
        new_measurement_model.rotation_offset = rotation_offset

        if isinstance(measurement_model, ReversibleModel):
            xyz = measurement_model.inverse_function(detection)
        elif isinstance(measurement_model, CartesianToElevationBearing):
            converter_measurement_model = CartesianToElevationBearingRange(
                    ndim_state=measurement_model.ndim_state,
                    mapping=measurement_model.mapping,
                    noise_covar=measurement_model.noise_covar,
                    translation_offset=measurement_model.translation_offset,
                    rotation_offset=measurement_model.rotation_offset
            )
            temp_det = copy.copy(detection)
            temp_det.state_vector = StateVector([*detection.state_vector, 1.0])
            xyz = converter_measurement_model.inverse_function(temp_det)
        else:
            raise NotImplementedError
        measurable_state = State(xyz)

        # noinspection PyTypeChecker
        return detection.from_state(
            detection,
            state_vector=new_measurement_model.function(measurable_state, noise=False),
            measurement_model=new_measurement_model,
            metadata=detection.metadata.copy()
        )


class AngleTrackingDetectionFeeder(OriginalStateDetectionSpaceFeeder2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.reader, AngleTrackingDetectionFeeder):
            self.reader = StaticRotationalFrameAngleDetectionFeeder(self.reader)
