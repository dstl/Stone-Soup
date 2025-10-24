import copy

import numpy as np
from scipy.linalg import block_diag

from ..base import Property
from ..models.measurement.bias import BiasModelWrapper
from ..models.measurement.nonlinear import CombinedReversibleGaussianMeasurementModel
from ..predictor.kalman import KalmanPredictor
from ..types.array import StateVector, CovarianceMatrix
from ..types.detection import Detection
from ..types.hypothesis import SingleHypothesis
from ..types.state import GaussianState
from ..types.update import Update
from ..updater import Updater
from ..updater.kalman import UnscentedKalmanUpdater


class GaussianBiasUpdater(Updater):
    measurement_model = None
    bias_state: GaussianState = Property(doc="Prior bias")
    bias_predictor: KalmanPredictor = Property(doc="Predictor for bias")
    bias_model_wrapper: BiasModelWrapper = Property()
    updater: Updater = Property(
        default=None,
        doc="Updater for bias and joint states. Must support non-linear models. "
        "Default `None` will create UKF instance.")
    max_bias: list[float] = Property(default=None, doc="Max bias ± from 0 allowed")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.updater is None:
            self.updater = UnscentedKalmanUpdater(None)

    def predict_measurement(
            self, predicted_state, measurement_model=None, measurement_noise=True, **kwargs):
        ndim_bias = self.bias_state.ndim

        # Predict bias
        if self.bias_state.timestamp is None:
            pred_bias_state = copy.copy(self.bias_state)
            pred_bias_state.timestamp = predicted_state.timestamp
        else:
            pred_bias_state = self.bias_predictor.predict(
                self.bias_state, timestamp=predicted_state.timestamp)

        applied_bias = getattr(measurement_model, 'applied_bias', np.zeros((ndim_bias, 1)))
        delta_bias = pred_bias_state.state_vector - applied_bias
        combined_pred = GaussianState(
            np.vstack([predicted_state.state_vector, delta_bias]).view(StateVector),
            block_diag(*[predicted_state.covar, pred_bias_state.covar]).view(CovarianceMatrix),
            timestamp=predicted_state.timestamp,
        )

        bias_measurement_model = self.bias_model_wrapper(
            ndim_state=combined_pred.state_vector.shape[0],
            measurement_model=measurement_model,
            state_mapping=list(range(predicted_state.ndim)),
            bias_mapping=list(range(-ndim_bias, 0))
        )

        return self.updater.predict_measurement(
            combined_pred, bias_measurement_model, measurement_noise, **kwargs)

    def update(self, hypotheses, **kwargs):
        if any(not hyp for hyp in hypotheses):
            raise ValueError("Must provide only non-missed detection hypotheses")

        ndim_bias = self.bias_state.ndim

        # Predict bias
        pred_time = max(hypothesis.prediction.timestamp for hypothesis in hypotheses)
        if self.bias_state.timestamp is None:
            self.bias_state.timestamp = pred_time
        else:
            new_bias_state = self.bias_predictor.predict(self.bias_state, timestamp=pred_time)
            self.bias_state.state_vector = new_bias_state.state_vector
            self.bias_state.covar = new_bias_state.covar
            self.bias_state.timestamp = new_bias_state.timestamp

        # Create joint state
        states = [hypothesis.prediction for hypothesis in hypotheses]
        applied_bias = next(
            (h.measurement.measurement_model.applied_bias
             for h in hypotheses
             if hasattr(h.measurement.measurement_model, 'applied_bias')),
            np.zeros((ndim_bias, 1)))
        delta_bias = self.bias_state.state_vector - applied_bias
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
            models.append(self.bias_model_wrapper(
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
        update = self.updater.update(SingleHypothesis(combined_pred, combined_meas), **kwargs)
        rel_delta_bias = update.state_vector[-ndim_bias:, :] - delta_bias
        self.bias_state.state_vector = self.bias_state.state_vector + rel_delta_bias
        if self.max_bias is not None:
            self.bias_state.state_vector = \
                np.min([abs(self.bias_state.state_vector), self.max_bias], axis=0) \
                * np.sign(self.bias_state.state_vector)
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
