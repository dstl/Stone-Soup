# -*- coding: utf-8 -*-
import numpy as np

from ..base import Property
from .base import Updater
from ..types.prediction import GaussianMeasurementPrediction
from ..types.update import GaussianStateUpdate


class ChernoffUpdater(Updater):
    r"""A class which performs state updates using the Chernoff fusion rule. In this context,
    measurements come in the form of states with a mean and covariance instead of just the
    traditional mean. The measurements are expected to come as :class:`~.GaussianDetection`
    objects.


    The Chernoff fusion rule is written as
    .. math::
           p_{\omega}(x_{k}) = \frac{p_{1}(x_{k})^{\omega}p_{2}(x_{k})^{1-\omega}}
                                    {\int p_{1}(x)^{\omega}p_{2}(x)^{1-\omega} \d x}

    where :math:`omega` is a weighting parameter in the range :math:`(0,1]`, which can be found
    using an optimization algorithm.

    In situations where :math:`p_1(x)` and :math:`p_2(x)` are multivariate Gaussian distributions,
    the above formula is equal to the Covariance Intersection Algorithm from Julier and Uhlmann.
    Let :math:`(a,A)` and :math:`(b,B)` be the means and covariances of the measurement and
    prediction respectively. The Covariance Intersection Algorithm was reformulated for use in
    Bayesian state estimation by Clark et al, yielding the update formulas for the covariance,
    mean, and innovation:

    .. math::

            D = \left ( \omega A^{-1} + (1-\omega)B^{-1} \right )
            d = D \left ( \omega A^{-1}a + (1-\omega)B^{-1}b \right )
            V = \frac{A}{1-\omega} + \frac{B}{\omega}


    In filters where gating is required, the gating region can be written using the innovation
    covariance matrix as:

    .. math::

            \mathcal{V}(\gamma) = \left { (a,A) : (a-b)^T V^{-1} (a-b) \leq \gamma \right }


    Note: If you have tracks that you would like to use as measurements for this updater, the
    :class:`~.Tracks2GaussianDetectionFeeder` class can be used to convert the tracks to the
    appropriate format.

    References
    ----------
    [1] Hurley, M. B., “An information theoretic justification for covariance intersection and its
    generalization,” in [Proceedings of the Fifth International Conference on Information Fusion.
    FUSION 2002.(IEEE Cat. No. 02EX5997) ], 1, 505–511, IEEE (2002).
    https://ieeexplore.ieee.org/document/1021196.
    [2] Julier, S., Uhlmann, J., and Durrant-Whyte, H. F., “A new method for the nonlinear
    transformation of means and covariances in filters and estimators,” IEEE Transactions on
    automatic control 45(3), 477–482 (2000).
    https://ieeexplore.ieee.org/abstract/document/847726/similar#similar.
    [3] Clark, D. E. and Campbell, M. A., “Integrating covariance intersection into Bayesian
    multi-target tracking filters,” preprint on TechRxiv. submitted to IEEE Transactions on
    Aerospace and Electronic Systems .
    """

    omega: float = Property(
        default=0.5,
        doc="A weighting parameter in the range :math:`(0,1]`")

    def predict_measurement(self, predicted_state, measurement_model=None,  **kwargs):
        '''
        This function predicts the measurement in situations where the predicted state consists
        of a covariance and state vector.
        '''

        measurement_model = self._check_measurement_model(measurement_model)

        # The innovation covariance uses the noise covariance from the measurement model
        state_covar_m = measurement_model.noise_covar
        innov_covar = 1/(1-self.omega)*state_covar_m + 1/self.omega*predicted_state.covar

        # The predicted measurement and measurement cross covariance can be taken from
        # the predicted state
        predicted_meas = predicted_state.state_vector
        meas_cross_cov = predicted_state.covar

        # Combine everything into a GaussianMeasurementPrediction object
        return GaussianMeasurementPrediction(predicted_meas, innov_covar,
                                             predicted_state.timestamp,
                                             cross_covar=meas_cross_cov)

    def update(self, hypothesis, force_symmetric_covariance=False, **kwargs):
        '''
        Given a hypothesis, calculate the posterior mean and covariance
        '''

        # Get the predicted state out of the hypothesis. These are 'B' and 'b', the
        # covariance and mean of the predicted Gaussian
        predicted_covar = hypothesis.prediction.covar
        predicted_mean = hypothesis.prediction.state_vector

        # Extract the vector and covariance from the measurement. These are 'A' and 'a', the
        # covariance and mean of the Gaussian measurement.
        measurement_covar = hypothesis.measurement.covar
        measurement_mean = hypothesis.measurement.state_vector

        # Predict the measurement if it is not already done
        if hypothesis.measurement_prediction is None:
            hypothesis.measurement_prediction = self.predict_measurement(
                hypothesis.prediction,
                measurement_model=hypothesis.measurement.measurement_model,
                **kwargs
            )

        # Calculate the updated mean and covariance from covariance intersection
        posterior_covariance = np.linalg.inv(self.omega*np.linalg.inv(measurement_covar) +
                                             (1-self.omega)*np.linalg.inv(predicted_covar))
        posterior_mean = posterior_covariance @ (self.omega*np.linalg.inv(measurement_covar)
                                                 @ measurement_mean +
                                                 (1-self.omega)*np.linalg.inv(predicted_covar)
                                                 @ predicted_mean)

        # Optionally force the posterior covariance to be a symmetric matrix
        if force_symmetric_covariance:
            posterior_covariance = \
                (posterior_covariance + posterior_covariance.T)/2

        # Return the updated state
        return GaussianStateUpdate(posterior_mean, posterior_covariance,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)
