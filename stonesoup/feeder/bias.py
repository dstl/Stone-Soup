import datetime
from abc import abstractmethod

from .base import DetectionFeeder
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.state import GaussianState


class _BiasFeeder(DetectionFeeder):
    bias_state: GaussianState = Property(doc="Bias State")

    @property
    def bias(self):
        return self.bias_state.mean

    @abstractmethod
    @BufferedGenerator.generator_method
    def data_gen(self):
        raise NotImplementedError()


class TimeBiasFeeder(_BiasFeeder):
    """Time Bias Feeder

    Apply bias to detection timestamp and overall timestamp yielded.
    """
    bias_state: GaussianState = Property(
        doc="Bias state with state vector shape (1, 1) in units of seconds")

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, detections in self.reader:
            bias = self.bias_state.state_vector[0, 0]
            bias_delta = datetime.timedelta(seconds=float(bias))
            time -= bias_delta
            models = set()
            for detection in detections:
                detection.timestamp -= bias_delta
                models.add(detection.measurement_model)
            for model in models:
                model.applied_bias = bias
            yield time, detections


class OrientationBiasFeeder(_BiasFeeder):
    """Orientation Bias Feeder

    Apply bias to detection measurement model rotation offset
    """
    bias_state: GaussianState = Property(
        doc="Bias state with state vector shape (3, 1) is expected")

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, detections in self.reader:
            bias = self.bias_state.state_vector.copy()
            models = set()
            for detection in detections:
                models.add(detection.measurement_model)
            for model in models:
                model.rotation_offset = model.rotation_offset - bias
                model.applied_bias = bias
            yield time, detections


class TranslationBiasFeeder(_BiasFeeder):
    """Translation Bias Feeder

    Apply bias to detection measurement model translation offset
    """
    bias_state: GaussianState = Property(
        doc="Bias state with state vector shape (n, 1), where n is dimensions of the model")

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, detections in self.reader:
            bias = self.bias_state.state_vector.copy()
            models = set()
            for detection in detections:
                models.add(detection.measurement_model)
            for model in models:
                model.translation_offset = model.translation_offset - bias
                model.applied_bias = bias
            yield time, detections


class OrientationTranslationBiasFeeder(_BiasFeeder):
    """Orientation Translation Bias Feeder

    Apply bias to detection measurement model rotation and translation offset
    """
    bias_state: GaussianState = Property(
        doc="Bias state with state vector shape (3+n, 1), 3 for rotation and where n is "
            "dimensions of the model")

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, detections in self.reader:
            bias = self.bias_state.state_vector.copy()
            models = set()
            for detection in detections:
                models.add(detection.measurement_model)
            for model in models:
                model.rotation_offset = model.rotation_offset - bias[:3]
                model.translation_offset = model.translation_offset - bias[3:]
                model.applied_bias = bias
            yield time, detections
