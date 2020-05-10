# -*- coding: utf-8 -*-
import scipy
import numpy as np
from functools import lru_cache

from ..base import Property
from .base import Updater
from ..types.state import SqrtGaussianState
from ..types.prediction import GaussianMeasurementPrediction
from ..types.update import GaussianStateUpdate
from ..models.base import LinearModel
from ..models.measurement.linear import LinearGaussian
from ..models.measurement import MeasurementModel
from ..functions import gauss2sigma, unscented_transform


class KalmanUpdater(Updater):
    r"""A class which embodies Kalman-type updaters; also a class which
    performs measurement update step as in the standard Kalman Filter.

    The Kalman updaters assume :math:`h(\mathbf{x}) = H \mathbf{x}` with
    additive noise :math:`\sigma = \mathcal{N}(0,R)`. Daughter classes can
    overwrite to specify a more general measurement model
    :math:`h(\mathbf{x})`.

    :math:`update` first calls :math:`predict_measurement` function which
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

    force_symmetric_covariance = Property(
        bool, default=False, doc="A flag to force the output covariance matrix"
                                 "to be symmetric by way of a simple geometric"
                                 "combination of the matrix and transpose."
                                 "Default is False.")

    sqrt_form = Property(
        bool, default=False, doc="If True then the update proceeds via a square"
                                 "root form of the covariance matrix and so"
                                 "should be more numerically stable. default is"
                                 " False")

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
        predicted_state : :class:`~.GaussianState`
            The predicted state :math:`\mathbf{x}_{k|k-1}`, :math:`P_{k|k-1}`
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
        predicted_state : :class:`~.GaussianState`
            The predicted state :math:`\mathbf{x}_{k|k-1}`, :math:`P_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            These are passed to :math:`~.MeasurementModel.function` and
            :math:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`GaussianMeasurementPrediction`
            The measurement prediction, :math:`\mathbf{z}_{k|k-1}`

        """
        # If a measurement model is not specified then use the one that's
        # native to the updater
        measurement_model = self._check_measurement_model(measurement_model)

        pred_meas = measurement_model.function(predicted_state, **kwargs)

        hh = self._measurement_matrix(predicted_state=predicted_state,
                                      measurement_model=measurement_model,
                                      **kwargs)

        meas_cross_cov = predicted_state.covar @ hh.T
        innov_cov = hh@meas_cross_cov + measurement_model.covar()

        return GaussianMeasurementPrediction(pred_meas, innov_cov,
                                             predicted_state.timestamp,
                                             cross_covar=meas_cross_cov)

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

        # If the square root form has been requested
        if self.sqrt_form:
            sqrtm_prior_cov = scipy.linalg.sqrtm(predicted_state.covar)
            sqrtm_noise_cov = scipy.linalg.sqrtm(measurement_model.covar())
            pred_meas = hypothesis.measurement_prediction.state_vector

            bigh = self._measurement_matrix(predicted_state=predicted_state,
                                            measurement_model=measurement_model,
                                            **kwargs)

            measdim = measurement_model.ndim_meas
            zeros = np.zeros((measurement_model.ndim_state, measdim))
            biga = np.block([[sqrtm_noise_cov, bigh @ sqrtm_prior_cov], [zeros, sqrtm_prior_cov]])
            [_, upper] = np.linalg.qr(biga.T)
            atheta = upper.T
            sqrtm_innov_cov = atheta[:measdim, :measdim]
            kalman_gain = atheta[measdim:, :measdim]
            sqrtP = atheta[measdim:, measdim:]

            posterior_mean = predicted_state.state_vector \
                + kalman_gain @ (np.linalg.inv(sqrtm_innov_cov)) \
                @ (hypothesis.measurement.state_vector - pred_meas)
            posterior_covariance = sqrtP @ sqrtP.T

        else:  # otherwise do the update the regular way

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

        if self.force_symmetric_covariance:
            posterior_covariance = \
                (posterior_covariance + posterior_covariance.T)/2

        return GaussianStateUpdate(posterior_mean, posterior_covariance,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)


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
            return measurement_model.jacobian(predicted_state,
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
            gauss2sigma(predicted_state,
                        self.alpha, self.beta, self.kappa)

        meas_pred_mean, meas_pred_covar, cross_covar, _, _, _ = \
            unscented_transform(sigma_points, mean_weights, covar_weights,
                                measurement_model.function,
                                covar_noise=measurement_model.covar())

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             predicted_state.timestamp,
                                             cross_covar)


class SqrtKalmanUpdater(KalmanUpdater):
    r"""The Square root version of the Kalman Updater.

    The input :class:`~.State` is a :class:`~.SqrtGaussianState` which means
    that the covariance of the predicted state is stored in square root form.
    This can be achieved by keeping :attr:`covar` attribute as :math:`L` where
    the 'full' covariance matrix :math:`P_{k|k-1} = L_{k|k-1} L^T_{k|k-1}`
    [Eq1].

    In its basic form :math:`L` is the lower triangular matrix returned via
    Cholesky factorisation. There's no reason why other forms that satisfy Eq 1
    above.

    """
    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None,
                            **kwargs):
        r"""Predict the measurement implied by the predicted state mean

        Parameters
        ----------
        predicted_state : :class:`~.SqrtGaussianState`
            The predicted state in square root form
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            These are passed to :math:`~.MeasurementModel.function` and
            :math:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`GaussianMeasurementPrediction`
            The measurement prediction, :math:`\mathbf{z}_{k|k-1}`

        """
        # If a measurement model is not specified then use the one that's
        # native to the updater
        measurement_model = self._check_measurement_model(measurement_model)

        pred_meas = measurement_model.function(predicted_state, **kwargs)

        hh = self._measurement_matrix(predicted_state=predicted_state,
                                      measurement_model=measurement_model,
                                      **kwargs)

        meas_cross_cov = predicted_state.covar.T @ hh.T
        innov_cov = meas_cross_cov.T @ meas_cross_cov + measurement_model.covar()

        return GaussianMeasurementPrediction(pred_meas, innov_cov,
                                             predicted_state.timestamp,
                                             cross_covar=meas_cross_cov)

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
        : :class:`~.SqrtGaussianState`
            The posterior state Gaussian with mean :math:`\mathbf{x}_{k|k}` and
            covariance in square root form, :math:`L_{x|x}` where

        Reference
        ---------
        [1] Andrews A. 1968, A Square Root Formulation of the Kalman Covariance
        Equations, Technical Note, Journal of American Institute of Aeronautics
        and Astronautics

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
        kalman_gain = predicted_state.covar @ m_cross_cov @ np.linalg.inv(innov_cov)

        posterior_mean = \
            predicted_state.state_vector \
            + kalman_gain @ (hypothesis.measurement.state_vector - pred_meas)

        bigu = np.linalg.cholesky(innov_cov)
        bigv = np.linalg.cholesky(measurement_model.covar())

        posterior_covariance = predicted_state.covar @ (
                np.identity(predicted_state.ndim) -
                m_cross_cov @ np.linalg.inv(bigu.T) @ np.linalg.inv(bigu + bigv) @ m_cross_cov.T)

        return SqrtGaussianState(posterior_mean, posterior_covariance)
