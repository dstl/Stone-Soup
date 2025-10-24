import datetime
from abc import abstractmethod

from .base import DetectionFeeder
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.state import GaussianState


class _GaussianBiasFeeder(DetectionFeeder):
    bias_state: GaussianState = Property(doc="Prior bias")

    @property
    def bias(self):
        return self.bias_state.state_vector

    @abstractmethod
    @BufferedGenerator.generator_method
    def data_gen(self):
        raise NotImplementedError()


class TimeGaussianBiasFeeder(_GaussianBiasFeeder):

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, detections in self.reader:
            bias = self.bias_state.state_vector.copy()
            bias_delta = datetime.timedelta(seconds=float(bias))
            time -= bias_delta
            models = set()
            for detection in detections:
                detection.timestamp -= bias_delta
                models.add(detection.measurement_model)
            for model in models:
                model.applied_bias = bias
            yield time, detections


class OrientationGaussianBiasFeeder(_GaussianBiasFeeder):

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


class TranslationGaussianBiasFeeder(_GaussianBiasFeeder):

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


class OrientationTranslationGaussianBiasFeeder(_GaussianBiasFeeder):

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
