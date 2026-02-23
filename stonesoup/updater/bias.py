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
from ..types.track import Track
from ..types.update import Update
from ..updater import Updater
from ..updater.kalman import UnscentedKalmanUpdater


class GaussianBiasUpdater(Updater):
    """Updater that jointly estimates a bias alongside target states.

    Maintains a separate Gaussian bias state and integrates it with target predictions
    to perform joint prediction and update steps. Uses a provided non-linear `updater`
    (defaults to an Unscented Kalman updater) and a `bias_model_wrapper` to build
    joint measurement models for bias estimation.

    Note that this assumes that all measurements/hypotheses are updating a common
    bias i.e. all measurements from the same sensor.
    """
    measurement_model = None
    bias_track: Track[GaussianState] = Property(doc="Prior bias")
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
        bias_state = self.bias_track.state
        ndim_bias = bias_state.ndim
        # Predict bias
        if bias_state.timestamp is None:
            pred_bias_state = copy.copy(bias_state)
            pred_bias_state.timestamp = predicted_state.timestamp
        else:
            pred_bias_state = self.bias_predictor.predict(
                bias_state, timestamp=predicted_state.timestamp)

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
        bias_state = self.bias_track.state
        ndim_bias = bias_state.ndim

        # Predict bias
        pred_time = max(hypothesis.prediction.timestamp for hypothesis in hypotheses)
        if bias_state.timestamp is None:
            bias_state.timestamp = pred_time
        else:
            bias_state = self.bias_predictor.predict(bias_state, timestamp=pred_time)

        # Create joint state
        states = [hypothesis.prediction for hypothesis in hypotheses]
        applied_bias = next(
            (h.measurement.measurement_model.applied_bias
             for h in hypotheses
             if hasattr(h.measurement.measurement_model, 'applied_bias')),
            np.zeros((ndim_bias, 1)))
        delta_bias = bias_state.state_vector - applied_bias
        states.append(GaussianState(delta_bias, bias_state.covar))
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
        relative_delta_bias = update.state_vector[-ndim_bias:, :] - delta_bias
        bias_state.state_vector = bias_state.state_vector + relative_delta_bias
        if self.max_bias is not None:
            bias_state.state_vector = \
                np.min([abs(bias_state.state_vector), self.max_bias], axis=0) \
                * np.sign(bias_state.state_vector)
        bias_state.covar = update.covar[-ndim_bias:, -ndim_bias:]
        self.bias_track.append(bias_state)

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
