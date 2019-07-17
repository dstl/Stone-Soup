# -*- coding: utf-8 -*-

import numpy as np
from functools import lru_cache

from ..base import Property
from .base import Updater
from ..types.prediction import GaussianMeasurementPrediction
from ..types.update import GaussianStateUpdate
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

    The :attr:`update()` function first calls the :attr:`predict_measurement()`
    function which proceeds by calculating the predicted measurement,
    innovation covariance and measurement cross-covariance,

    .. math::

        \mathbf{z}_{k|k-1} = H_k \mathbf{x}_{k|k-1}

        S_k = H_k P_{k|k-1} H_k^T + R_k

        \Upsilon_k = P_{k|k-1} H_k^T

    where :math:`P_{k|k-1}` is the predicted state covariance. The
    :attr:`predict_measurement()` function returns a
    :class:`GaussianMeasurementPrediction`. The Kalman gain is then calculated
    as,

    .. math::

        K_k = \Upsilon_k S_k^{-1}

    and the posterior state mean and covariance are,

    .. math::

        \mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + K_k (\mathbf{z}_k - H_k
        \mathbf{x}_{k|k-1})

        P_{k|k} = P_{k|k-1} - K_k S_k K_k^T

    These are returned as a :class:`GaussianStateUpdate` object.
    """

    # TODO: at present this will throw an error if a measurement model is not
    # TODO: specified in either individual measurements or the Updater object
    measurement_model = Property(LinearGaussian, default=None,
                                 doc="A linear Gaussian measurement model. "
                                     "This need not be defined if a "
                                     "measurement model is provided in the "
                                     "measurement. If no model specified on "
                                     "construction, or in the measurement, "
                                     "then error will be thrown.")

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

    def measurement_matrix(self, predicted_state=None, measurement_model=None,
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
            Passed to :class:`~.MeasurementModel`. :attr:`matrix()`

        Returns
        -------
        : :obj:`ndarray`
            The measurement matrix, :math:`H_k`

        """
        return self.measurement_model.matrix(**kwargs)

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
            These are passed to :class:`~.MeasurementModel`. :attr:`function()`
            and :class:`~.MeasurementModel`. :attr:`matrix()`

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

        hh = self.measurement_matrix(predicted_state=predicted_state,
                                     measurement_model=measurement_model,
                                     **kwargs)

        innov_cov = hh @ predicted_state.covar @ hh.T + \
            measurement_model.covar()
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
        hypothesis : :class:`SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.
        force_symmetric_covariance : :obj:`bool`, default=False
            A flag to force the output covariance matrix to be symmetric by way
            of a simple geometric combination of the matrix and transpose.
        **kwargs : various
            These are passed to :attr:`predict_measurement()` of
            :class:`~.KalmanUpdater` or its daughter classes

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
        kalman_gain = m_cross_cov @ np.linalg.inv(innov_cov)  # This isn't
        # optimised
        posterior_mean = predicted_state.state_vector + \
            kalman_gain @ (hypothesis.measurement.state_vector - pred_meas)
        posterior_covariance = \
            predicted_state.covar - kalman_gain @ innov_cov @ kalman_gain.T

        if force_symmetric_covariance:
            posterior_covariance = (posterior_covariance +
                                    posterior_covariance.T) / 2

        return GaussianStateUpdate(posterior_mean, posterior_covariance,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)


class ExtendedKalmanUpdater(KalmanUpdater):
    r"""The Extended Kalman Filter version of the Kalman Updater. Inherits most
    of the functionality from :class:`~.KalmanUpdater`.

    The difference is that the measurement model may now be non-linear, though
    must be differentiable to return the linearisation of :math:`h(\mathbf{x})`
    via the matrix :math:`H` accessible via the :attr:`jacobian()` function.

    """
    # TODO: Enforce the fact that this version of MeasurementModel must be
    # TODO: capable of executing :attr:`jacobian()`
    measurement_model = Property(MeasurementModel, default=None,
                                 doc="A measurement model. This need not be "
                                     "defined if a measurement model is "
                                     "provided in the measurement. If no "
                                     "model specified on construction, or in "
                                     "the measurement, then error will be "
                                     "thrown. Must be linear or capable of "
                                     "returning the :attr:`jacobian()` "
                                     "function.")

    def measurement_matrix(self, predicted_state, measurement_model=None,
                           **kwargs):
        r"""Return the (approximate via :attr:`jacobian()`) measurement matrix

        Parameters
        ----------
        predicted_state : :class:`~.State`
            The predicted state :math:`\mathbf{x}_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            Passed to :class:`~.MeasurementModel`. :attr:`matrix()` if linear
            or :class:`~.MeasurementModel`. :attr:`jacobian()` if not

        Returns
        -------
        : :obj:`ndarray`
            The measurement matrix, :math:`H_k`

        """

        measurement_model = self._check_measurement_model(measurement_model)

        if hasattr(measurement_model, 'matrix'):
            return measurement_model.matrix(**kwargs)
        else:
            return measurement_model.jacobian(predicted_state.state_vector,
                                              **kwargs)


class UnscentedKalmanUpdater(KalmanUpdater):
    """The Unscented Kalman Filter version of the Kalman Updater. Inherits most
    of the functionality from :class:`~.KalmanUpdater`.

    In this case the :attr:`predict_measurement()` function uses the
    :attr:`unscented_transform()` function to estimate a (Gaussian) predicted
    measurement. This is then updated via the standard Kalman update equations.

    """
    # Can be non-linear and non-differentiable
    measurement_model = Property(MeasurementModel, default=None,
                                 doc="The measurement model to be used. This "
                                     "need not be defined if a measurement "
                                     "model is provided in the measurement. "
                                     "If no model specified on construction, "
                                     "or in the measurement, then error will "
                                     "be thrown.")

    # TODO: Why is default 0.5?
    alpha = Property(float, default=0.5,
                     doc="Primary sigma point spread scaling parameter, "
                         "typically :math:`10^{-3}`")
    beta = Property(float, default=2,
                    doc="Used to incorporate prior knowledge of the "
                        "distribution. If the true distribution is Gaussian, "
                        "the value of 2 is optimal.")
    # TODO: Again, default seems oddly chosen
    kappa = Property(float, default=0,
                     doc="Secondary spread scaling parameter\
                        (default is calculated as :math:`3-Ns`)")

    #
    def measurement_function_nonoise(self, x, w=0, **kwargs):
        """This to ensure that no noise is added to the measurement in the
        unscented transform (below). (Would resolve if the default was to add
        no noise.) This is passed as a function handle to the unscented
        transform to transform the sigma points. Intended for use with the
        sigma points, but will work with any :attr:`statevector` array more
        generally.

        Parameters
        ----------
        x : :obj:`ndarray`
            The array of sigma points
        w : :obj:`ndarray`, optional, default=0
            Array of noise values on the sigma points

        Returns
        -------
        : :obj:`ndarray`
            The 'measurement' generated by passing the sigma points through the
            measurement function without adding noise.

    """

        return self.measurement_model.function(x, w, **kwargs)

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
            dependent on the received measurement (the default is ``None``, in
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

        meas_pred_mean, meas_pred_covar, cross_covar, _, _, _ = \
            unscented_transform(sigma_points, mean_weights, covar_weights,
                                self.measurement_function_nonoise,
                                covar_noise=measurement_model.covar())

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             predicted_state.timestamp,
                                             cross_covar)
