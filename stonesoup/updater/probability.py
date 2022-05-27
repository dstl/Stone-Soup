# -*- coding: utf-8 -*-
import warnings
import numpy as np
import scipy.linalg as la
from functools import lru_cache

from .kalman import ExtendedKalmanUpdater


class PDAUpdater(ExtendedKalmanUpdater):
    r"""An updater which undertakes probabilistic data association (PDA), as defined in [1]. It
    differs slightly from the Kalman updater it inherits from in that instead of a single
    hypothesis object, the :meth:`update` method it takes a hypotheses object returned by a
    :class:`~.PDA` (or similar) data associator. Functionally this is a list of single hypothesis
    objects which group tracks together with associated measuments and probabilities.

    The :class:`~.ExtendedKalmanUpdater` is used in order to inherit the ability to cope with
    (slight) non-linearities.

    The update step proceeds as:

    .. math::

        \mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + K_k \mathbf{y}_{k}

        P_{k|k} = \beta_0 P_{k|k-1} + (1 - \beta_0) P_^C_{k|k} + \tilde{P}

    where :math:`K_k` and :math:`P^C_{k|k} are the Kalman gain and posterior covariance
    respectively returned by the single-target Kalman update, :math:`\beta_0` is the probability
    of missed detection. In this instance :math:`\mathbf{y}_k` is the {\em combined} innovation,
    over :math:`m_k` detections:

    .. math::

        \mathbf{y}_k = \Sigma_{i=1}^{m_k} \beta_i \mathbf{y}_{k,i}.

    The posterior covariance is composed of a term to account for the covariance due to missed
    detection, that due to the true detection, and a term (:math:`\tilde{P}`) which quantifies the
    effect of the measurement origin uncertainty.

    .. math::

        \tilde{P} \def K_k \[ \Sigma_{i=1}^{m_k} \beta_i \mathbf{y}_{k,i} \beta_i
        \mathbf{y}_{k,i}^T - \mathbf{y}_k \mathbf{y}_k^T \] K_k^T

    """

    def update(self, hypotheses, **kwargs):
        r"""Of n hypotheses there should be 1 prediction (a missed detection) and n-1 different
        measurement associations. Because we don't know where in the list this occurs, but we do
        know that there's only one, we iterate til we find it, then start the calculation from
        there, then return to the values we missed."""

        position_prediction = False  # flag to record whether we have the prediction
        for n, hypothesis in enumerate(hypotheses):

            # Check for the existence of an associated measurement. Because of the way the
            # hypothesis is constructed, you can do this:
            if not hypothesis:

                # Predict the measurement and attach it to the hypothesis
                hypothesis.measurement_prediction = self.predict_measurement(hypothesis.prediction)
                # Now get P_k|k
                posterior_covariance, kalman_gain = self._posterior_covariance(hypothesis)

                # Add the weighted prediction to the weighted posterior
                posterior_covariance = hypothesis.probability * hypothesis.prediction.covar + \
                                       (1 - hypothesis.probability) * posterior_covariance

                posterior_mean = hypothesis.prediction.state_vector
                total_innovation = 0 * hypothesis.measurement_prediction.state_vector
                weighted_innovation_cov = 0 * total_innovation @ total_innovation.T

                # record the position in the list of the prediction
                position_prediction = n
            else:
                if not position_prediction:
                    continue
                else:
                    innovation = hypothesis.measurement.state_vector - hypothesis.measurement_prediction.state_vector
                    total_innovation += hypothesis.probability * innovation
                    weighted_innovation_cov += hypothesis.probability * innovation @ innovation.T

        # and then return to do those elements on the list that weren't covered prior to the prediction.
        for n, hypothesis in enumerate(hypotheses):
            if n == position_prediction:
                break
            else:
                innovation = hypothesis.measurement.state_vector - hypothesis.measurement_prediction.state_vector
                total_innovation += hypothesis.probability * innovation
                weighted_innovation_cov += hypothesis.probability * innovation @ innovation.T

        posterior_mean += kalman_gain @ total_innovation
        posterior_covariance += \
            kalman_gain @ (weighted_innovation_cov - total_innovation @ total_innovation.T) @ kalman_gain.T

        return Update.from_state(
            hypotheses[0].prediction,
            posterior_mean, posterior_covariance,
            timestamp=hypotheses[0].measurement.timestamp, hypothesis=hypotheses[0])