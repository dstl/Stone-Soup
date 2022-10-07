from collections import OrderedDict
from functools import lru_cache

import numpy as np

from .kalman import KalmanUpdater
from ..types.prediction import ASDGaussianMeasurementPrediction
from ..types.state import State
from ..types.update import ASDGaussianStateUpdate


class ASDKalmanUpdater(KalmanUpdater):
    """Accumulated State Densities Kalman Updater

    A linear updater for accumulated state densities, for processing out of
    sequence measurements. This requires the state is represented in
    :class:`ASDGaussianState` multi-state.

    References
    ----------
    1.  W. Koch and F. Govaers, On Accumulated State Densities with Applications to
        Out-of-Sequence Measurement Processing in IEEE Transactions on Aerospace and
        Electronic Systems,
        vol. 47, no. 4, pp. 2766-2778, OCTOBER 2011, doi: 10.1109/TAES.2011.6034663.
    """
    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None,
                            **kwargs):
        r"""Predict the measurement implied by the predicted state mean

        Parameters
        ----------
        predicted_state : :class:`~.ASDState`
            The predicted state :math:`\mathbf{x}_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater
            object is used
        **kwargs : various
            These are passed to :meth:`~.MeasurementModel.function` and
            :meth:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`ASDGaussianMeasurementPrediction`
            The measurement prediction, :math:`\mathbf{z}_{k|k-1}`

        """
        measurement_model = self._check_measurement_model(measurement_model)

        t_index = predicted_state.timestamps.index(predicted_state.act_timestamp)
        t2t_plus = slice(t_index * predicted_state.ndim, (t_index+1) * predicted_state.ndim)

        pred_meas = measurement_model.function(
            State(predicted_state.multi_state_vector[t2t_plus]), **kwargs)

        hh = self._measurement_matrix(predicted_state=predicted_state,
                                      measurement_model=measurement_model,
                                      **kwargs)

        innov_cov = (
            hh
            @ predicted_state.multi_covar[t2t_plus, t2t_plus]
            @ hh.T + measurement_model.covar())

        meas_cross_cov = predicted_state.multi_covar[:, t2t_plus] @ hh.T

        return ASDGaussianMeasurementPrediction(
            multi_state_vector=pred_meas, multi_covar=innov_cov,
            timestamps=[predicted_state.timestamps[0]],
            cross_covar=meas_cross_cov)

    def update(self, hypothesis, force_symmetric_covariance=False, **kwargs):
        r"""The Kalman update method. Given a hypothesised association between
        a predicted state and an actual measurement,
        calculate the posterior state. The Measurement Prediction should be
        calculated by this method. It is overwritten in this method

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.
        force_symmetric_covariance : :obj:`bool`, optional
            A flag to force the output covariance matrix to be symmetric by way
            of a simple geometric combination of the matrix and transpose.
            Default is `False`
        **kwargs : various
            These are passed to :meth:`predict_measurement`

        Returns
        -------
        : :class:`~.ASDGaussianStateUpdate`
            The posterior state Gaussian with mean :math:`\mathbf{x}_{k|k}` and
            covariance :math:`P_{x|x}`
        """

        # Get the predicted state out of the hypothesis
        predicted_state = hypothesis.prediction

        if hypothesis.measurement_prediction is None:
            # Get the measurement model out of the measurement if it's there.
            # If not, use the one native to the updater (which might still be
            # none)
            measurement_model = hypothesis.measurement.measurement_model
            measurement_model = self._check_measurement_model(measurement_model)

            # Attach the measurement prediction to the hypothesis
            hypothesis.measurement_prediction = self.predict_measurement(
                predicted_state, measurement_model=measurement_model, **kwargs)

        # Get the predicted measurement mean, innovation covariance and
        # measurement cross-covariance
        pred_meas = hypothesis.measurement_prediction.state_vector
        innov_cov = hypothesis.measurement_prediction.covar
        m_cross_cov = hypothesis.measurement_prediction.cross_covar

        # Complete the calculation of the posterior
        # This isn't optimised
        kalman_gain = m_cross_cov @ np.linalg.inv(innov_cov)
        posterior_mean = \
            predicted_state.multi_state_vector \
            + kalman_gain @ (hypothesis.measurement.state_vector - pred_meas)
        posterior_covariance = predicted_state.multi_covar - kalman_gain@innov_cov@kalman_gain.T

        if force_symmetric_covariance:
            posterior_covariance = \
                (posterior_covariance + posterior_covariance.T) / 2

        # save the new posterior, if it is no out of sequence measurement
        pred_corr_matrices = predicted_state.correlation_matrices.setdefault(
            predicted_state.act_timestamp, dict())
        t_index = predicted_state.timestamps.index(predicted_state.act_timestamp)
        t2t_plus = slice(t_index * predicted_state.ndim, (t_index+1) * predicted_state.ndim)

        # update covariance after calculating
        pred_corr_matrices['P'] = posterior_covariance[t2t_plus, t2t_plus]
        try:
            pred_corr_matrices['PFP'] = (
                pred_corr_matrices['P']
                @ pred_corr_matrices['F'].T
                @ np.linalg.inv(pred_corr_matrices['P_pred']))
        except KeyError:
            pass
        correlation_matrices = OrderedDict(sorted(
            predicted_state.correlation_matrices.items(), reverse=True))

        return ASDGaussianStateUpdate(multi_state_vector=posterior_mean,
                                      multi_covar=posterior_covariance,
                                      hypothesis=hypothesis,
                                      timestamps=predicted_state.timestamps,
                                      correlation_matrices=correlation_matrices,
                                      max_nstep=predicted_state.max_nstep)
