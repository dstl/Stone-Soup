# -*- coding: utf-8 -*-
import warnings
import numpy as np
import scipy.linalg as la
from functools import lru_cache

from .kalman import ExtendedKalmanUpdater
from ..types.update import Update
from ..types.array import StateVectors
from ..functions import gm_reduce_single


class PDAUpdater(ExtendedKalmanUpdater):
    r"""An updater which undertakes probabilistic data association (PDA), as defined in [1]. It
    differs slightly from the Kalman updater it inherits from in that instead of a single
    hypothesis object, the :meth:`update` method takes a hypotheses object returned by a
    :class:`~.PDA` (or similar) data associator. Functionally this is a list of single hypothesis
    objects which group tracks together with associated measuments and probabilities.

    The :class:`~.ExtendedKalmanUpdater` is used in order to inherit the ability to cope with
    (slight) non-linearities. Other inheritance structures should be trivial to implement.

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

    def update(self, hypotheses, gm_method=False, **kwargs):
        r"""The update step.

        Parameters
        ----------
        hypotheses : :class:`~.MultipleHypothesis`
            the prediction-measurement association hypotheses. This hypotheses object carries
            tracks, associated sets of measurements for each track together with a probability
            measure which enumerates the likelihood of each track-measurement pair. (This is most
            likely output by the :class:`~.PDA` associator).

            In a single case (the missed detection hypothesis), the hypothesis will not have an
            associated measurement or measurement prediction.
            latter case a predicted measurement will be calculated.
        gm_method : bool
            either use the innovation-based update methods (detailed above and in [1]), if False,
            or use the GM-reduction (True).
        **kwargs : various
            These are passed to :meth:`predict_measurement`

        Returns
        -------
        : :class:`GaussianMeasurementPrediction`
            The measurement prediction, :math:`\mathbf{z}_{k|k-1}`

        """
        if gm_method:
            posterior_mean, posterior_covariance = self._update_via_GM_reduction(hypotheses,**kwargs)
        else:
            posterior_mean, posterior_covariance = self._update_via_innovation(hypotheses, **kwargs)

        # Note that this does not check if all hypotheses are of the same type.
        # It also assumes that all measurements have the same timestamp (updates are
        # contemporaneous).
        return Update.from_state(
            hypotheses[0].prediction,
            posterior_mean, posterior_covariance,
            timestamp=hypotheses[0].measurement.timestamp, hypothesis=hypotheses)

    def _update_via_GM_reduction(self, hypotheses, **kwargs):
        """This method delivers the same outcome as what's described above. It's slightly
        different, but potentially more intuitive.

        Here, each the hypotheses, including missed detection, are updated and then a weighted
        Gaussian reduction is used to resolve the hypotheses to a single Gaussian distribution.

        The reason this is equivalent is:
            TBA

        Parameters
        ----------
        hypotheses : :class:`~.MultipleHypothesis`
            the prediction-measurement association hypotheses. This hypotheses object carries
            tracks, associated sets of measurements for each track together with a probability
            measure which enumerates the likelihood of each track-measurement pair. (This is most
            likely output by the :class:`~.PDA` associator).

            In a single case (the missed detection hypothesis), the hypothesis will not have an
            associated measurement or measurement prediction.
            latter case a predicted measurement will be calculated.
         **kwargs : various
            These are passed to :class:`~.ExtendedKalmanUpdater`:meth:`update`

        Returns
        -------
        : :class:`~.StateVector`
            The mean of the reduced/single Gaussian
        : :class:`~.CovarianceMatrix`
            The covariance of the reduced/single Gaussian
        """

        posterior_states = []
        posterior_state_weights = []
        for hypothesis in hypotheses:
            if not hypothesis:
                posterior_states.append(hypothesis.prediction)
            else:
                posterior_state = super().update(hypothesis, **kwargs)  # Use the EKF update
                posterior_states.append(posterior_state)
            posterior_state_weights.append(hypothesis.probability)

        means = StateVectors([state.state_vector for state in posterior_states])
        covars = np.stack([state.covar for state in posterior_states], axis=2)
        weights = np.asarray(posterior_state_weights)

        # Reduce mixture of states to one posterior estimate Gaussian.
        post_mean, post_covar = gm_reduce_single(means, covars, weights)

        return post_mean, post_covar

    def _update_via_innovation(self, hypotheses, **kwargs):
        """Of n hypotheses there should be 1 prediction (a missed detection hypothesis) and n-1
        different measurement associations. The update proceeds as described above.

        Parameters
        ----------
        hypotheses : :class:`~.MultipleHypothesis`
            the prediction-measurement association hypotheses. This hypotheses object carries
            tracks, associated sets of measurements for each track together with a probability
            measure which enumerates the likelihood of each track-measurement pair. (This is most
            likely output by the :class:`~.PDA` associator).

            In a single case (the missed detection hypothesis), the hypothesis will not have an
            associated measurement or measurement prediction.
            latter case a predicted measurement will be calculated.
         **kwargs : various
            These are passed to :meth:`predict_measurement`

        Returns
        -------
        : :class:`~.StateVector`
            The mean of the reduced/single Gaussian
        : :class:`~.CovarianceMatrix`
            The covariance of the reduced/single Gaussian
        """

        for n, hypothesis in enumerate(hypotheses):
            # Check for the existence of an associated measurement. Because of the way the
            # hypothesis is constructed, you can do this:
            if not hypothesis:
                hypothesis.measurement_prediction = self.predict_measurement(hypothesis.prediction,
                                                                             **kwargs)
                innovation = hypothesis.measurement_prediction.state_vector - \
                             hypothesis.measurement_prediction.state_vector  # is zero in this case
                posterior_covariance, kalman_gain = self._posterior_covariance(hypothesis)
                # Add the weighted prediction to the weighted posterior
                posterior_covariance = hypothesis.probability * hypothesis.prediction.covar + \
                                       (1 - hypothesis.probability) * posterior_covariance
                posterior_mean = hypothesis.prediction.state_vector
            else:
                innovation = hypothesis.measurement.state_vector - hypothesis.measurement_prediction.state_vector

            if n == 0:  # probably exists a less clunky way of doing this using exists() or overwritten +=
                sum_of_innovations = hypothesis.probability * innovation
                sum_of_weighted_cov = hypothesis.probability * innovation @ innovation.T
            else:
                sum_of_innovations += hypothesis.probability * innovation
                sum_of_weighted_cov += hypothesis.probability * innovation @ innovation.T

        posterior_mean += kalman_gain @ sum_of_innovations
        posterior_covariance += \
                kalman_gain @ (sum_of_weighted_cov - sum_of_innovations @ sum_of_innovations.T) \
                @ kalman_gain.T

        return posterior_mean, posterior_covariance
