from functools import lru_cache

import numpy as np
from scipy.stats import multivariate_normal

from . import Updater
from ..kernel import QuadraticKernel, Kernel
from ..types.array import StateVectors
from ..types.prediction import MeasurementPrediction
from ..types.update import Update
from ..base import Property


class AdaptiveKernelKalmanUpdater(Updater):
    """The adaptive kernel Kalman updater uses the predictions from the predictor to generate the
     measurement particles and update the posterior kernel weight vector and covariance matrix.
     Additionally, the updater generates new proposal particles at every step to refine the state
     estimate.
    """
    kernel: Kernel = Property(
        default_factory=QuadraticKernel,
        doc="Default is None. If None, the default :class:`QuadraticKernel` is used.")
    lambda_updater: float = Property(
        default=1e-3,
        doc="Used to incorporate prior knowledge of the distribution. If the "
            "true distribution is Gaussian, the value of 2 is optimal. "
            "Default is 1e-3")

    @lru_cache()
    def predict_measurement(self, state_prediction, measurement_model=None,
                            **kwargs):

        if measurement_model is None:
            measurement_model = self.measurement_model

        new_state_vector = measurement_model.function(state_prediction, **kwargs)
        return MeasurementPrediction.from_state(
            state_prediction,
            state_vector=new_state_vector)

    def update(self, hypothesis, **kwargs):
        r"""The adaptive kernel Kalman update method. Given a hypothesised association between
        a predicted state or predicted measurement and an actual measurement,
        calculate the posterior state.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.
        **kwargs : various
            These are passed to :meth:`predict_measurement`

        Returns
        -------
        : :class:`~.KernelParticleStateUpdate`
            The posterior state Gaussian with mean :math:`\mathbf{x}_{k|k}` and
            covariance :math:`P_{x|x}`
        """

        # Get the predicted state out of the hypothesis
        predicted_state = hypothesis.prediction

        # If there is no measurement prediction in the hypothesis then do the
        # measurement prediction (and attach it back to the hypothesis).
        if hypothesis.measurement_prediction is None:
            # Get the measurement model out of the measurement if it's there.
            # If not, use the one native to the updater (which might still be
            # none)
            measurement_model = hypothesis.measurement.measurement_model
            measurement_model = self._check_measurement_model(
                measurement_model)

            # Attach the measurement prediction to the hypothesis
            hypothesis.measurement_prediction = self.predict_measurement(
                predicted_state, measurement_model=measurement_model,
                measurement_noise=False, **kwargs)
        G_yy = self.kernel(hypothesis.measurement_prediction)
        g_y = self.kernel(hypothesis.measurement_prediction, hypothesis.measurement)

        Q_AKKF = \
            predicted_state.kernel_covar \
            @ np.linalg.pinv(G_yy @ predicted_state.kernel_covar
                             + self.lambda_updater * np.identity(len(predicted_state)))
        weights = predicted_state.weight[:, np.newaxis]
        updated_weights = (weights + Q_AKKF@(g_y - G_yy@weights)).ravel()
        updated_covariance = \
            predicted_state.kernel_covar - Q_AKKF @ G_yy @ predicted_state.kernel_covar

        # Proposal Calculation
        pred_mean = predicted_state.state_vector @ updated_weights
        pred_covar = np.diag(np.diag(
            predicted_state.state_vector @ updated_covariance @ predicted_state.state_vector.T))

        new_state_vector = multivariate_normal.rvs(
            pred_mean, pred_covar, size=len(predicted_state)
        )

        return Update.from_state(
            hypothesis.prediction,
            state_vector=predicted_state.state_vector,
            proposal=StateVectors(new_state_vector.T),
            weight=updated_weights,  # W^+
            kernel_covar=updated_covariance,  # S^+
            timestamp=hypothesis.measurement.timestamp, hypothesis=hypothesis)
