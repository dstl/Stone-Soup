from ...types.update import Update
from stonesoup.updater.kalman import ExtendedKalmanUpdater


class TwoStateKalmanUpdater(ExtendedKalmanUpdater):

    def update(self, hypothesis, **kwargs):
        r"""The Kalman update method. Given a hypothesised association between
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
        : :class:`~.GaussianStateUpdate`
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
                predicted_state, measurement_model=measurement_model, **kwargs)

        # Kalman gain and posterior covariance
        posterior_covariance, kalman_gain = self._posterior_covariance(hypothesis)

        # Posterior mean
        posterior_mean = predicted_state.state_vector + \
            kalman_gain@(hypothesis.measurement.state_vector -
                         hypothesis.measurement_prediction.state_vector)

        if self.force_symmetric_covariance:
            posterior_covariance = \
                (posterior_covariance + posterior_covariance.T)/2

        return Update.from_state(
            hypothesis.prediction,
            posterior_mean, posterior_covariance,
            start_time=hypothesis.prediction.start_time,
            end_time=hypothesis.measurement.timestamp,
            hypothesis=hypothesis)
