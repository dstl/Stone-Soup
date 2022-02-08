# -*- coding: utf-8 -*-

from functools import lru_cache

import numpy as np

from ..base import Property
from ..types.prediction import GaussianMeasurementPrediction
from ..types.update import Update
from ..models.measurement.linear import LinearGaussian
from ..updater.kalman import KalmanUpdater


class InformationKalmanUpdater(KalmanUpdater):
    r"""A class which implements the update of information form of the Kalman filter. This is
    conceptually very simple. The update proceeds as:

    .. math::

        Y_{k|k} = Y_{k|k-1} + H^{T}_k R^{-1}_k H_k

        \mathbf{y}_{k|k} = \mathbf{y}_{k|k-1} + H^{T}_k R^{-1}_k \mathbf{z}_{k}

    where :math:`\mathbf{y}_{k|k-1}` is the predicted information state and :math:`Y_{k|k-1}` the
    predicted information matrix which form the :class:`~.InformationStatePrediction` object. The
    measurement matrix :math:`H_k` and measurement covariance :math:`R_k` are those in the Kalman
    filter (see tutorial 1). An :class:`~.InformationStateUpdate` object is returned.

    Note
    ----
    Analogously with the :class:`~.InformationKalmanPredictor`, the measurement model is queried
    for the existence of an :meth:`inverse_covar()` property. If absent, the :meth:`covar()` is
    inverted.

    """
    measurement_model: LinearGaussian = Property(
        default=None,
        doc="A linear Gaussian measurement model. This need not be defined if "
            "a measurement model is provided in the measurement. If no model "
            "specified on construction, or in the measurement, then error "
            "will be thrown.")

    def _inverse_measurement_covar(self, measurement_model, **kwargs):
        """Return the inverse of the measurement covariance (or calculate it)

        Parameters
        ----------
        measurement_model
            The measurement model to be queried
        **kwargs : various, optional
            These are passed to :meth:`~.LinearGaussian.covar()`

        Returns
        -------
        : :class:`numpy.ndarray`
            The inverse of the measurement covariance, :math:`R_k^{-1}`

        """
        if hasattr(measurement_model, 'inverse_covar'):
            inv_measurement_covar = measurement_model.inverse_covar(**kwargs)
        else:
            inv_measurement_covar = np.linalg.inv(measurement_model.covar(**kwargs))

        return inv_measurement_covar

    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):
        r"""There's no direct analogue of a predicted measurement in the information form. This
        method is therefore provided to return the predicted measurement as would the standard
        Kalman updater. This is mainly for compatibility as it's not anticipated that it would
        be used in the usual operation of the information filter.

        Parameters
        ----------
        predicted_information_state : :class:`~.State`
            The predicted state in information form :math:`\mathbf{y}_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            These are passed to :meth:`~.MeasurementModel.matrix()`

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction, :math:`H \mathbf{x}_{k|k-1}`

        """
        # If a measurement model is not specified then use the one that's
        # native to the updater
        measurement_model = self._check_measurement_model(measurement_model)

        hh = self._measurement_matrix(predicted_state=predicted_state,
                                      measurement_model=measurement_model,
                                      **kwargs)

        predicted_covariance = np.linalg.inv(predicted_state.precision)
        predicted_state_mean = predicted_covariance @ predicted_state.state_vector

        predicted_measurement = hh @ predicted_state_mean
        innovation_covariance = hh @ predicted_covariance @ hh.T + measurement_model.covar()

        return GaussianMeasurementPrediction(predicted_measurement, innovation_covariance,
                                             predicted_state.timestamp,
                                             cross_covar=predicted_covariance @ hh.T)

    def update(self, hypothesis, **kwargs):
        r"""The Information filter update (corrector) method. Given a hypothesised association
        between a predicted information state and an actual measurement, calculate the posterior
        information state.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            carries a predicted information state.
        **kwargs : various
            These are passed to :meth:`predict_measurement`

        Returns
        -------
        : :class:`~.InformationStateUpdate`
            The posterior information state with information state :math:`\mathbf{y}_{k|k}` and
            precision :math:`Y_{k|k}`

        """

        measurement_model = hypothesis.measurement.measurement_model
        measurement_model = self._check_measurement_model(measurement_model)

        pred_info_mean = hypothesis.prediction.state_vector
        hh = measurement_model.matrix()
        invr = self._inverse_measurement_covar(measurement_model)

        posterior_precision = hypothesis.prediction.precision + hh.T @ invr @ hh
        posterior_information_mean = pred_info_mean + hh.T @ invr @ \
            hypothesis.measurement.state_vector

        if self.force_symmetric_covariance:
            posterior_precision = (posterior_precision + posterior_precision.T)/2

        return Update.from_state(hypothesis.prediction, posterior_information_mean,
                                 posterior_precision,
                                 timestamp=hypothesis.measurement.timestamp, hypothesis=hypothesis)
