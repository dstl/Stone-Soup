import copy
import datetime
from abc import abstractmethod
from functools import partial

import numpy as np
from scipy.linalg import block_diag

from .base import DetectionFeeder
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..models.measurement.bias import \
    OrientationBiasWrapper, OrientationTranslationBiasWrapper, \
    TimeBiasWrapper, TranslationBiasWrapper
from ..models.measurement.nonlinear import CombinedReversibleGaussianMeasurementModel
from ..models.transition import TransitionModel
from ..predictor.kalman import KalmanPredictor
from ..updater.kalman import KalmanUpdater, UnscentedKalmanUpdater
from ..types.array import CovarianceMatrix, StateVector
from ..types.detection import Detection
from ..types.hypothesis import SingleHypothesis
from ..types.state import GaussianState
from ..types.update import Update


class _GaussianBiasFeeder(DetectionFeeder):
    bias_prior: GaussianState = Property(doc="Prior bias")
    bias_predictor: KalmanPredictor = Property(doc="Predictor for bias")
    updater: KalmanUpdater = Property(
        default=None,
        doc="Updater for bias and joint states. Must support non-linear models. "
        "Default `None` will create UKF instance.")
    max_bias: list[float] = Property(default=None, doc="Max bias ± from 0 allowed")

    @property
    @abstractmethod
    def _model_wrapper(self):
        raise NotImplementedError()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bias_state = copy.deepcopy(self.bias_prior)
        if self.updater is None:
            self.updater = UnscentedKalmanUpdater(None)

    @property
    def ndim_bias(self):
        return self._bias_state.state_vector.shape[0]

    @property
    def bias_state(self):
        return self._bias_state

    @property
    def bias(self):
        return self._bias_state.state_vector

    @abstractmethod
    @BufferedGenerator.generator_method
    def data_gen(self):
        raise NotImplementedError()

    def update_bias(self, hypotheses):
        if any(not hyp for hyp in hypotheses):
            raise ValueError("Must provide only non-missed detection hypotheses")

        ndim_bias = self.ndim_bias

        # Predict bias
        pred_time = max(hypothesis.prediction.timestamp for hypothesis in hypotheses)
        if self._bias_state.timestamp is None:
            self._bias_state.timestamp = pred_time
        else:
            self._bias_state = self.bias_predictor.predict(self._bias_state, timestamp=pred_time)

        # Create joint state
        states = [hypothesis.prediction for hypothesis in hypotheses]
        applied_bias = next((h.measurement.applied_bias for h in hypotheses),
                            np.zeros((ndim_bias, 1)))
        delta_bias = self.bias - applied_bias
        states.append(GaussianState(delta_bias, self.bias_state.covar))
        combined_pred = GaussianState(
            np.vstack([state.state_vector for state in states]).view(StateVector),
            block_diag(*[state.covar for state in states]).view(CovarianceMatrix),
            timestamp=pred_time,
        )

        # Create joint measurement
        offset = 0
        models = []
        for prediction, measurement in (
                (hypothesis.prediction, hypothesis.measurement) for hypothesis in hypotheses):
            models.append(self._model_wrapper(
                ndim_state=combined_pred.state_vector.shape[0],
                measurement_model=measurement.measurement_model,
                state_mapping=[offset + n for n in range(prediction.ndim)],
                bias_mapping=list(range(-ndim_bias, 0))
            ))
            offset += prediction.ndim
        combined_meas = Detection(
            np.vstack([hypothesis.measurement.state_vector for hypothesis in hypotheses]),
            timestamp=pred_time,
            measurement_model=CombinedReversibleGaussianMeasurementModel(models))

        # Update bias
        update = self.updater.update(SingleHypothesis(combined_pred, combined_meas))
        rel_delta_bias = update.state_vector[-ndim_bias:, :] - delta_bias
        self.bias_state.state_vector += rel_delta_bias
        if self.max_bias is not None:
            self.bias = np.min([abs(self.bias), self.max_bias], axis=0) * np.sign(self.bias)
        self.bias_state.covar = update.covar[-ndim_bias:, -ndim_bias:]

        # Create update states
        offset = 0
        updates = []
        for hypothesis in hypotheses:
            update_slice = slice(offset, offset + hypothesis.prediction.ndim)
            updates.append(Update.from_state(
                hypothesis.prediction,
                state_vector=update.state_vector[update_slice, :],
                covar=update.covar[update_slice, update_slice],
                timestamp=hypothesis.prediction.timestamp,
                hypothesis=hypothesis))
            offset += hypothesis.prediction.ndim

        return updates


class TimeGaussianBiasFeeder(_GaussianBiasFeeder):
    transition_model: TransitionModel = Property()

    @property
    def _model_wrapper(self):
        return partial(TimeBiasWrapper, transition_model=self.transition_model)

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, detections in self.reader:
            bias = self.bias
            bias_delta = datetime.timedelta(seconds=float(bias))
            time -= bias_delta
            for detection in detections:
                detection.timestamp -= bias_delta
                detection.applied_bias = bias
            yield time, detections


class OrientationGaussianBiasFeeder(_GaussianBiasFeeder):
    _model_wrapper = OrientationBiasWrapper

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, detections in self.reader:
            bias = self.bias
            models = set()
            for detection in detections:
                models.add(detection.measurement_model)
                detection.applied_bias = bias
            for model in models:
                model.rotation_offset = model.rotation_offset - bias
            yield time, detections


class TranslationGaussianBiasFeeder(_GaussianBiasFeeder):
    _model_wrapper = TranslationBiasWrapper

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, detections in self.reader:
            bias = self.bias.copy()
            models = set()
            for detection in detections:
                models.add(detection.measurement_model)
                detection.applied_bias = bias
            for model in models:
                model.translation_offset = model.translation_offset - bias
            yield time, detections


class OrientationTranslationGaussianBiasFeeder(_GaussianBiasFeeder):
    _model_wrapper = OrientationTranslationBiasWrapper

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, detections in self.reader:
            bias = self.bias.copy()
            models = set()
            for detection in detections:
                models.add(detection.measurement_model)
                detection.applied_bias = bias
            for model in models:
                model.rotation_offset = model.rotation_offset - bias[:3]
                model.translation_offset = model.translation_offset - bias[3:]
            yield time, detections
