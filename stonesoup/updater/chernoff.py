import numpy as np

from ..base import Property
from .base import Updater
from ..types.prediction import MeasurementPrediction
from ..types.update import Update


class ChernoffUpdater(Updater):
    r"""A class which performs state updates using the Chernoff fusion rule. In this context,
    measurements come in the form of states with a mean and covariance (compared to traditional
    measurements which contain solely a mean). The measurements are expected to come as
    :class:`~.GaussianDetection` objects.


    The Chernoff fusion rule is written as [#]_

    .. math::
           p_{\omega}(x_{k}) = \frac{p_{1}(x_{k})^{\omega}p_{2}(x_{k})^{1-\omega}}
                                    {\int p_{1}(x)^{\omega}p_{2}(x)^{1-\omega} \mathrm{d} x}

    where :math:`\omega` is a weighting parameter in the range :math:`(0,1]`, which can be found
    using an optimization algorithm.

    In situations where :math:`p_1(x)` and :math:`p_2(x)` are multivariate Gaussian distributions,
    the above formula is equal to the Covariance Intersection Algorithm from Julier et al [#]_.
    Let :math:`(a,A)` and :math:`(b,B)` be the means and covariances of the measurement and
    prediction respectively. The Covariance Intersection Algorithm was reformulated for use in
    Bayesian state estimation by Clark and Campbell [#]_, yielding formulas for the updated
    covariance and mean, :math:`D` and :math:`d`, and the innovation covariance matrix, :math:`V`,
    as follows:

    .. math::

            D &= \left ( \omega A^{-1} + (1-\omega)B^{-1} \right )\\
            d &= D \left ( \omega A^{-1}a + (1-\omega)B^{-1}b \right )\\
            V &= \frac{A}{1-\omega} + \frac{B}{\omega}


    In filters where gating is required, the gating region can be written using the innovation
    covariance matrix as:

    .. math::

            \mathcal{V}(\gamma) = \left\{ (a,A) : (a-b)^T V^{-1} (a-b) \leq \gamma \right\}


    The specifics for implementing the Covariance Intersection Algorithm in several popular
    multi-target tracking algorithms was expanded upon by Clark et al [#]_. The work includes a
    discussion of Stone Soup and can be used to apply this class to a tracking algorithm of
    choice.

    Note
    ----
    If you have tracks that you would like to use as measurements for this updater, the
    :class:`~.Tracks2GaussianDetectionFeeder` class can be used to convert the tracks to the
    appropriate format.

    References
    ----------
    .. [#] Hurley, M., “An information theoretic justification for covariance intersection and its
       generalization,” in [Proceedings of the Fifth International Conference on Information
       Fusion. FUSION 2002.(IEEE Cat. No. 02EX5997) ], 1, 505–511, IEEE (2002).
       https://ieeexplore.ieee.org/document/1021196.
    .. [#] Julier, S., Uhlmann, J., and Durrant-Whyte, H., “A new method for the nonlinear
       transformation of means and covariances in filters and estimators,” IEEE Transactions on
       automatic control 45(3), 477–482 (2000).
       https://ieeexplore.ieee.org/abstract/document/847726/similar#similar.
    .. [#] Clark, D. and Campbell, M., “Integrating covariance intersection into Bayesian
       multi-target tracking filters,” preprint on TechRxiv. submitted to IEEE Transactions on
       Aerospace and Electronic Systems.
    .. [#] Clark, D. and Hunter, E. and Balaji, B. and O'Rourke, S., “Centralized multi-sensor
       multi-target data fusion with tracks as measurements,” to be submitted to SPIE Defense and
       Security Symposium 2023.
    """

    omega: float = Property(
        default=0.5,
        doc="A weighting parameter in the range :math:`(0,1]`")

    def predict_measurement(self, predicted_state, measurement_model=None,  **kwargs):
        r"""
        This function predicts the measurement of a state in situations where measurements consist
        of a covariance and state vector.

        Parameters
        ----------
        predicted_state : :class:`~.GaussianState`
            The predicted state :math:`\mathbf{x}_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the updater will use the model that was specified
            on initialization.

        Returns
        -------
        : :class:`~.MeasurementPrediction`
            The measurement prediction
        """

        measurement_model = self._check_measurement_model(measurement_model)

        # The innovation covariance uses the noise covariance from the measurement model
        state_covar_m = measurement_model.noise_covar
        innov_covar = 1/(1-self.omega)*state_covar_m + 1/self.omega*predicted_state.covar

        # The predicted measurement and measurement cross covariance can be taken from
        # the predicted state
        predicted_meas = predicted_state.state_vector
        meas_cross_cov = predicted_state.covar

        # Combine everything into a GaussianMeasurementPrediction object
        return MeasurementPrediction.from_state(predicted_state, predicted_meas, innov_covar,
                                                predicted_state.timestamp,
                                                cross_covar=meas_cross_cov)

    def update(self, hypothesis, force_symmetric_covariance=False, **kwargs):
        r"""
        Given a hypothesis, calculate the posterior mean and covariance.

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with the predicted state and the actual/associated measurement which should
            be used for updating. If the hypothesis does not contain a measurement prediction, one
            will be calculated.

        force_symmetric_covariance: bool
            A flag to force the output covariance matrix to be symmetric by way of a simple
            geometric combination of the matrix and transpose. Default is False.

        Returns
        -------
        : :class:`~.Update`
            The state posterior, saved in a generic :class:`~.Update` object.
        """

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
        return Update.from_state(hypothesis.prediction, posterior_mean, posterior_covariance,
                                 hypothesis, hypothesis.measurement.timestamp)
