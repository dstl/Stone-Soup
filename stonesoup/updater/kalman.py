# -*- coding: utf-8 -*-

from functools import lru_cache

import numpy as np

from .base import Updater
from ..base import Property
from ..functions import gauss2sigma, unscented_transform
from ..models.base import LinearModel
from ..models.measurement import MeasurementModel
from ..models.measurement.linear import LinearGaussian
from ..types.prediction import GaussianMeasurementPrediction, ASDGaussianMeasurementPrediction
from ..types.update import GaussianStateUpdate, ASDGaussianStateUpdate


class KalmanUpdater(Updater):
    r"""A class which embodies Kalman-type updaters; also a class which
    performs measurement update step as in the standard Kalman Filter.

    The Kalman updaters assume :math:`h(\mathbf{x}) = H \mathbf{x}` with
    additive noise :math:`\sigma = \mathcal{N}(0,R)`. Daughter classes can
    overwrite to specify a more general measurement model
    :math:`h(\mathbf{x})`.

    :meth:`update` first calls :meth:`predict_measurement` function which
    proceeds by calculating the predicted measurement, innovation covariance
    and measurement cross-covariance,

    .. math::

        \mathbf{z}_{k|k-1} = H_k \mathbf{x}_{k|k-1}

        S_k = H_k P_{k|k-1} H_k^T + R_k

        \Upsilon_k = P_{k|k-1} H_k^T

    where :math:`P_{k|k-1}` is the predicted state covariance.
    :meth:`predict_measurement` returns a
    :class:`~.GaussianMeasurementPrediction`. The Kalman gain is then
    calculated as,

    .. math::

        K_k = \Upsilon_k S_k^{-1}

    and the posterior state mean and covariance are,

    .. math::

        \mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + K_k (\mathbf{z}_k - H_k
        \mathbf{x}_{k|k-1})

        P_{k|k} = P_{k|k-1} - K_k S_k K_k^T

    These are returned as a :class:`~.GaussianStateUpdate` object.
    """

    # TODO: at present this will throw an error if a measurement model is not
    # TODO: specified in either individual measurements or the Updater object
    measurement_model = Property(
        LinearGaussian, default=None,
        doc="A linear Gaussian measurement model. This need not be defined if "
            "a measurement model is provided in the measurement. If no model "
            "specified on construction, or in the measurement, then error "
            "will be thrown.")

    def _check_measurement_model(self, measurement_model):
        """Check that the measurement model passed actually exists. If not
        attach the one in the updater. If that one's not specified, return an
        error.

        Parameters
        ----------
        measurement_model : :class`~.MeasurementModel`
            A measurement model to be checked

        Returns
        -------
        : :class`~.MeasurementModel`
            The measurement model to be used

        """
        if measurement_model is None:
            if self.measurement_model is None:
                raise ValueError("No measurement model specified")
            else:
                measurement_model = self.measurement_model

        return measurement_model

    def _measurement_matrix(self, predicted_state=None, measurement_model=None,
                            **kwargs):
        r"""This is straightforward Kalman so just get the Matrix from the
        measurement model.

        Parameters
        ----------
        predicted_state : :class:`~.State`
            The predicted state :math:`\mathbf{x}_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            Passed to :meth:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`numpy.ndarray`
            The measurement matrix, :math:`H_k`

        """
        return self._check_measurement_model(
            measurement_model).matrix(**kwargs)

    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None,
                            **kwargs):
        r"""Predict the measurement implied by the predicted state mean

        Parameters
        ----------
        predicted_state : :class:`~.State`
            The predicted state :math:`\mathbf{x}_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            These are passed to :meth:`~.MeasurementModel.function` and
            :meth:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`GaussianMeasurementPrediction`
            The measurement prediction, :math:`\mathbf{z}_{k|k-1}`

        """
        # If a measurement model is not specified then use the one that's
        # native to the updater
        measurement_model = self._check_measurement_model(measurement_model)

        pred_meas = measurement_model.function(predicted_state.state_vector,
                                               noise=0, **kwargs)

        hh = self._measurement_matrix(predicted_state=predicted_state,
                                      measurement_model=measurement_model,
                                      **kwargs)

        innov_cov = hh@predicted_state.covar@hh.T + measurement_model.covar()
        meas_cross_cov = predicted_state.covar @ hh.T

        return GaussianMeasurementPrediction(pred_meas, innov_cov,
                                             predicted_state.timestamp,
                                             cross_covar=meas_cross_cov)

    def update(self, hypothesis, force_symmetric_covariance=False, **kwargs):
        r"""The Kalman update method. Given a hypothesised association between
        a predicted state or predicted measurement and an actual measurement,
        calculate the posterior state.

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

        # Get the predicted measurement mean, innovation covariance and
        # measurement cross-covariance
        pred_meas = hypothesis.measurement_prediction.state_vector
        innov_cov = hypothesis.measurement_prediction.covar
        m_cross_cov = hypothesis.measurement_prediction.cross_covar

        # Complete the calculation of the posterior
        # This isn't optimised
        kalman_gain = m_cross_cov @ np.linalg.inv(innov_cov)
        posterior_mean = \
            predicted_state.state_vector \
            + kalman_gain@(hypothesis.measurement.state_vector - pred_meas)
        posterior_covariance = \
            predicted_state.covar - kalman_gain@innov_cov@kalman_gain.T

        if force_symmetric_covariance:
            posterior_covariance = \
                (posterior_covariance + posterior_covariance.T)/2

        return GaussianStateUpdate(posterior_mean, posterior_covariance,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)


class ASDKalmanUpdater(KalmanUpdater):
    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None,
                            **kwargs):
        r"""Predict the measurement implied by the predicted state mean

                Parameters
                ----------
                predicted_state : :class:`~.ASDState`
                    The predicted state :math:`\mathbf{x}_{k|k-1}`
                measurement_model : :class:`~.MeasurementModel`
                    The measurement model. If omitted, the model in the updater object
                    is used
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
        pred_meas = measurement_model.function(predicted_state.multi_state_vector[t_index * predicted_state.ndim: (t_index +1) * predicted_state.ndim],noise=0, **kwargs)


        hh = self._measurement_matrix(predicted_state=predicted_state,
                                      measurement_model=measurement_model,
                                      **kwargs)

        innov_cov = hh @ predicted_state.multi_covar[t_index * predicted_state.ndim: (t_index +1) * predicted_state.ndim,t_index * predicted_state.ndim: (t_index +1) * predicted_state.ndim] @ hh.T + measurement_model.covar()

        meas_cross_cov = predicted_state.multi_covar[:, t_index * predicted_state.ndim: (t_index +1) * predicted_state.ndim] @ hh.T



        return ASDGaussianMeasurementPrediction(multi_state_vector=pred_meas, multi_covar=innov_cov,
                                                timestamps=[predicted_state.timestamps[0]], cross_covar=meas_cross_cov)

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


        # Get the measurement model out of the measurement if it's there.
        # If not, use the one native to the updater (which might still be
        # none)
        measurement_model = hypothesis.measurement.measurement_model
        measurement_model = self._check_measurement_model(
            measurement_model)


        t_index = predicted_state.timestamps.index(predicted_state.act_timestamp)
            # calculate the update for the measrement step
        if t_index!=0:
            P_l = predicted_state.multi_covar[t_index * predicted_state.ndim: (t_index + 1) * predicted_state.ndim, t_index * predicted_state.ndim: (t_index + 1) * predicted_state.ndim]
            #HPH + R
            hh_l = self._measurement_matrix(predicted_state=predicted_state,
                                            measurement_model=measurement_model,
                                            **kwargs)
            innov_cov = hh_l @ predicted_state.multi_covar[t_index * predicted_state.ndim: (t_index +1) * predicted_state.ndim,t_index * predicted_state.ndim: (t_index +1) * predicted_state.ndim] @ hh_l.T + measurement_model.covar()

            kalman_gain_l = predicted_state.multi_covar[t_index * predicted_state.ndim: (t_index +1) * predicted_state.ndim:, t_index * predicted_state.ndim: (t_index +1) * predicted_state.ndim] @ hh_l.T @ np.linalg.inv(innov_cov)
            P_l = P_l - kalman_gain_l @ innov_cov @ kalman_gain_l.T
            predicted_state.correlation_matrices[predicted_state.act_timestamp]['P'] = P_l
            predicted_state.correlation_matrices[predicted_state.act_timestamp]['P_pred'] = predicted_state.correlation_matrices[predicted_state.act_timestamp]['F'] @ P_l @ predicted_state.correlation_matrices[predicted_state.act_timestamp]['F'].T + predicted_state.correlation_matrices[predicted_state.act_timestamp]['P_error']
            predicted_state.correlation_matrices[predicted_state.act_timestamp]['PFP'] = \
                predicted_state.correlation_matrices[predicted_state.act_timestamp]['P'] \
                @ predicted_state.correlation_matrices[predicted_state.act_timestamp]['F'].T \
                @ np.linalg.inv(predicted_state.correlation_matrices[predicted_state.act_timestamp]['P_pred'])
            # get all timestamps which has to be recalculated beginning with the newest one
            timestamps_to_recalculate = [ts for ts in predicted_state.timestamps if ts > predicted_state.act_timestamp]
            covars = [predicted_state.multi_covar[i * predicted_state.ndim:(i + 1) * predicted_state.ndim, i * predicted_state.ndim:(i + 1) * predicted_state.ndim] for i in
                      range(t_index)]

            for i, ts in enumerate(timestamps_to_recalculate):
                prior_ndim = predicted_state.ndim
                C_list = []
                C_list.append(np.eye(prior_ndim))
                corrs = {k: v for k, v in predicted_state.correlation_matrices.items() if k < ts}
                for item in list(corrs.values())[-1::-1]:
                    C_list.append(C_list[-1] @ item['PFP'])
                C_list = C_list[1:]
                W_column = np.array([ c @ covars[i] for c in C_list])
                W_column = np.reshape(W_column, (predicted_state.ndim * len(C_list), predicted_state.ndim))
                W_row = W_column.T

                # set covar
                predicted_state.multi_covar[i * prior_ndim:(i + 1) * prior_ndim, i * prior_ndim:(i + 1) * prior_ndim] = covars[i]

                # set column

                predicted_state.multi_covar[(i + 1) * prior_ndim:, i * prior_ndim: (i + 1) * prior_ndim] = W_column

                # set row
                predicted_state.multi_covar[i * prior_ndim: (i + 1) * prior_ndim, (i + 1) * prior_ndim:] = W_row


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
        posterior_covariance = \
            predicted_state.multi_covar - kalman_gain @ innov_cov @ kalman_gain.T

        if force_symmetric_covariance:
            posterior_covariance = \
                (posterior_covariance + posterior_covariance.T) / 2


        # calculate the rest of the correlation matrix so that the whole matrix is in
        # the correlation_matrices dictionary.
        try:
            predicted_state.correlation_matrices[predicted_state.act_timestamp]
        except KeyError:
            predicted_state.correlation_matrices[predicted_state.act_timestamp] = {}
        t_index = predicted_state.timestamps.index(predicted_state.act_timestamp)
        ndmin = predicted_state.ndim
        predicted_state.correlation_matrices[predicted_state.act_timestamp]['P'] = posterior_covariance[t_index*ndmin: (t_index+1) * ndmin,t_index*ndmin: (t_index+1) * ndmin]


        # update the PFP for the correlations, if it is an out of sequence measurement
        if t_index != 0:
            predicted_state.correlation_matrices[predicted_state.act_timestamp]['PFP'] = \
                predicted_state.correlation_matrices[predicted_state.act_timestamp]['P'] \
                @ predicted_state.correlation_matrices[predicted_state.act_timestamp]['F'].T \
                @ predicted_state.correlation_matrices[predicted_state.act_timestamp]['P_pred']


        return ASDGaussianStateUpdate(multi_state_vector=posterior_mean, multi_covar=posterior_covariance,
                                      hypothesis=hypothesis, timestamps=predicted_state.timestamps,
                                      correlation_matrices=predicted_state.correlation_matrices, max_nstep=predicted_state.max_nstep)


class ExtendedKalmanUpdater(KalmanUpdater):
    r"""The Extended Kalman Filter version of the Kalman Updater. Inherits most
    of the functionality from :class:`~.KalmanUpdater`.

    The difference is that the measurement model may now be non-linear, though
    must be differentiable to return the linearisation of :math:`h(\mathbf{x})`
    via the matrix :math:`H` accessible via :meth:`~.NonLinearModel.jacobian`.

    """
    # TODO: Enforce the fact that this version of MeasurementModel must be
    # TODO: capable of executing :attr:`jacobian()`
    measurement_model = Property(
        MeasurementModel, default=None,
        doc="A measurement model. This need not be defined if a measurement "
            "model is provided in the measurement. If no model specified on "
            "construction, or in the measurement, then error will be thrown. "
            "Must be linear or capable or implement the "
            ":meth:`~.NonLinearModel.jacobian`.")

    def _measurement_matrix(self, predicted_state, measurement_model=None,
                            **kwargs):
        r"""Return the (via :meth:`NonLinearModel.jacobian`) measurement matrix

        Parameters
        ----------
        predicted_state : :class:`~.State`
            The predicted state :math:`\mathbf{x}_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            Passed to :meth:`~.MeasurementModel.matrix` if linear
            or :meth:`~.MeasurementModel.jacobian` if not

        Returns
        -------
        : :class:`numpy.ndarray`
            The measurement matrix, :math:`H_k`

        """

        measurement_model = self._check_measurement_model(measurement_model)

        if isinstance(measurement_model, LinearModel):
            return measurement_model.matrix(**kwargs)
        else:
            return measurement_model.jacobian(predicted_state.state_vector,
                                              **kwargs)


class UnscentedKalmanUpdater(KalmanUpdater):
    """The Unscented Kalman Filter version of the Kalman Updater. Inherits most
    of the functionality from :class:`~.KalmanUpdater`.

    In this case the :meth:`predict_measurement` function uses the
    :func:`unscented_transform` function to estimate a (Gaussian) predicted
    measurement. This is then updated via the standard Kalman update equations.

    """
    # Can be non-linear and non-differentiable
    measurement_model = Property(
        MeasurementModel,
        default=None,
        doc="The measurement model to be used. This need not be defined if a "
            "measurement model is provided in the measurement. If no model "
            "specified on construction, or in the measurement, then error "
            "will be thrown.")
    alpha = Property(
        float,
        default=0.5,
        doc="Primary sigma point spread scaling parameter. Default is 0.5.")
    beta = Property(
        float,
        default=2,
        doc="Used to incorporate prior knowledge of the distribution. If the "
            "true distribution is Gaussian, the value of 2 is optimal. "
            "Default is 2")
    kappa = Property(
        float,
        default=0,
        doc="Secondary spread scaling parameter. Default is calculated as "
            "3-Ns")

    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None):
        """Unscented Kalman Filter measurement prediction step. Uses the
        unscented transform to estimate a Gauss-distributed predicted
        measurement.

        Parameters
        ----------
        predicted_state : :class:`~.GaussianStatePrediction`
            A predicted state
        measurement_model : :class:`~.MeasurementModel`, optional
            The measurement model used to generate the measurement prediction.
            This should be used in cases where the measurement model is
            dependent on the received measurement (the default is `None`, in
            which case the updater will use the measurement model specified on
            initialisation)

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction

        """

        measurement_model = self._check_measurement_model(measurement_model)

        sigma_points, mean_weights, covar_weights = \
            gauss2sigma(predicted_state.state_vector, predicted_state.covar,
                        self.alpha, self.beta, self.kappa)

        def measurement_function_nonoise(state_vector, noise=0, **kwargs):
            return measurement_model.function(state_vector, noise, **kwargs)

        meas_pred_mean, meas_pred_covar, cross_covar, _, _, _ = \
            unscented_transform(sigma_points, mean_weights, covar_weights,
                                measurement_function_nonoise,
                                covar_noise=measurement_model.covar())

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             predicted_state.timestamp,
                                             cross_covar)
