import copy
from abc import abstractmethod
from typing import Iterable, List

from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..feeder import Feeder
from stonesoup.feeder.base import DetectionFeeder
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian


class SimpleFeeder(Feeder):
    """Simple data feeder

    Creates a generator from an iterable.
    """
    reader: Iterable = Property(doc="Source of states")

    @BufferedGenerator.generator_method
    def data_gen(self):
        for item in self.reader:
            yield item


class IterDetectionFeeder(DetectionFeeder):

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        ...

    @BufferedGenerator.generator_method
    def data_gen(self):
        detection_iter = iter(self)
        for time, detections in detection_iter:
            yield (time, detections)


class OriginalStateDetectionSpaceFeeder(IterDetectionFeeder):

    reader: DetectionFeeder = Property(doc="Source of detections")

    measurement_model_key = "original measurement model"

    ndim_state: int = Property(default=None,
                               doc="Number of state dimensions position and velocity for detection"
                                   "state space. Using None, takes this from the measurement model"
                               )
    mapping: List[int] = Property(default=None, doc="Index of position components in the state")

    def convert_detection(self, a_detection: Detection) -> Detection:
        new_detection = copy.deepcopy(a_detection)

        new_detection.measurement_model = LinearGaussian(
                ndim_state=self.ndim_state or a_detection.measurement_model.ndim_state,
                mapping=self.mapping or a_detection.measurement_model.mapping,
                noise_covar=a_detection.measurement_model.noise_covar
        )
        new_detection.metadata[self.measurement_model_key] = a_detection.measurement_model

        return new_detection

    def __iter__(self):
        self.reader_iter = iter(self.reader)
        return self

    def __next__(self):
        time, detections = next(self.reader_iter)
        new_detections = set([self.convert_detection(a_detection) for a_detection in detections])

        return time, new_detections


class OriginalStateDetectionSpaceFeeder2D(OriginalStateDetectionSpaceFeeder):

    ndim_state: int = Property(default=4,
                               doc="Number of state dimensions position and velocity for detection"
                                   "state space. Using None, takes this from the measurement model"
                               )
    mapping: List[int] = Property(default=(0, 2), doc="Index of position components in the state")
