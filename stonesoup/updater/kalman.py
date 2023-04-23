import warnings

import numpy
import numpy as np
import scipy.linalg as la
from functools import lru_cache

from ..base import Property
from .base import Updater
from ..types.array import CovarianceMatrix, StateVector
from ..types.prediction import MeasurementPrediction
from ..types.update import Update
from ..models.base import LinearModel
from ..models.measurement.linear import LinearGaussian
from ..models.measurement import MeasurementModel
from ..functions import gauss2sigma, unscented_transform
from ..measures import Measure, Euclidean


class KalmanUpdater(Updater):
    r"""A class which embodies Kalman-type updaters; also a class which
    performs measurement update step as in the standard Kalman filter.

    The Kalman updaters assume :math:`h(\mathbf{x}) = H \mathbf{x}` with
    additive noise :math:`\sigma = \mathcal{N}(0,R)`. Daughter classes can
    overwrite to specify a more general measurement model
    :math:`h(\mathbf{x})`.

    :meth:`update` first calls :meth:`predict_measurement` function which
    proceeds by calculating the predicted measurement, innovation covariance
    and measurement cross-covariance,

    .. math::

        \mathbf{z}_{k|k-1} &= H_k \mathbf{x}_{k|k-1}

        S_k &= H_k P_{k|k-1} H_k^T + R_k

        \Upsilon_k &= P_{k|k-1} H_k^T

    where :math:`P_{k|k-1}` is the predicted state covariance.
    :meth:`predict_measurement` returns a
    :class:`~.GaussianMeasurementPrediction`. The Kalman gain is then
    calculated as,

    .. math::

        K_k = \Upsilon_k S_k^{-1}

    and the posterior state mean and covariance are,

    .. math::

        \mathbf{x}_{k|k} &= \mathbf{x}_{k|k-1} + K_k (\mathbf{z}_k - H_k
        \mathbf{x}_{k|k-1})

        P_{k|k} &= P_{k|k-1} - K_k S_k K_k^T

    These are returned as a :class:`~.GaussianStateUpdate` object.
    """

    # TODO: at present this will throw an error if a measurement model is not
    # TODO: specified in either individual measurements or the Updater object
    measurement_model: LinearGaussian = Property(
        default=None,
        doc="A linear Gaussian measurement model. This need not be defined if "
            "a measurement model is provided in the measurement. If no model "
            "specified on construction, or in the measurement, then error "
            "will be thrown.")
    force_symmetric_covariance: bool = Property(
        default=False,
        doc="A flag to force the output covariance matrix to be symmetric by way of a simple "
            "geometric combination of the matrix and transpose. Default is False.")

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

    def _measurement_cross_covariance(self, predicted_state, measurement_matrix):
        """
        Return the measurement cross covariance matrix, :math:`P_{k~k-1} H_k^T`

        Parameters
        ----------
        predicted_state : :class:`GaussianState`
            The predicted state which contains the covariance matrix :math:`P` as :attr:`.covar`
            attribute
        measurement_matrix : numpy.array
            The measurement matrix, :math:`H`

        Returns
        -------
        :  numpy.ndarray
            The measurement cross-covariance matrix

        """
        return predicted_state.covar @ measurement_matrix.T

    def _innovation_covariance(self, m_cross_cov, meas_mat, meas_mod):
        """Compute the innovation covariance

        Parameters
        ----------
        m_cross_cov : numpy.ndarray
            The measurement cross covariance matrix
        meas_mat : numpy.ndarray
            Measurement matrix
        meas_mod : :class:~.MeasurementModel`
            Measurement model

        Returns
        -------
        : numpy.ndarray
            The innovation covariance

        """
        return meas_mat @ m_cross_cov + meas_mod.covar()

    def _posterior_mean(self, predicted_state, kalman_gain, measurement, measurement_prediction):
        r"""Compute the posterior mean, :math:`\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + K_k
        \mathbf{y}_k`, where the innovation :math:`\mathbf{y}_k = \mathbf{z}_k -
        h(\mathbf{x}_{k|k-1}).

        Parameters
        ----------
        predicted_state : :class:`State`, :class:`Prediction`
            The predicted state
        kalman_gain : numpy.ndarray
            Kalman gain
        measurement : :class:`Detection`
            The measurement
        measurement_prediction : :class:`MeasurementPrediction`
            Predicted measurement

        Returns
        -------
        : :class:`StateVector`
            The posterior mean estimate
        """
        post_mean = predicted_state.state_vector + \
            kalman_gain @ (measurement.state_vector - measurement_prediction.state_vector)
        return post_mean.view(StateVector)

    def _posterior_covariance(self, hypothesis):
        """
        Return the posterior covariance for a given hypothesis

        Parameters
        ----------
        hypothesis: :class:`~.Hypothesis`
            A hypothesised association between state prediction and measurement. It returns the
            measurement prediction which in turn contains the measurement cross covariance,
            :math:`P_{k|k-1} H_k^T and the innovation covariance,
            :math:`S = H_k P_{k|k-1} H_k^T + R`

        Returns
        -------
        : :class:`~.CovarianceMatrix`
            The posterior covariance matrix rendered via the Kalman update process.
        : numpy.ndarray
            The Kalman gain, :math:`K = P_{k|k-1} H_k^T S^{-1}`

        """
        kalman_gain = hypothesis.measurement_prediction.cross_covar @ \
            np.linalg.inv(hypothesis.measurement_prediction.covar)

        post_cov = hypothesis.prediction.covar - kalman_gain @ \
            hypothesis.measurement_prediction.covar @ kalman_gain.T

        return post_cov.view(CovarianceMatrix), kalman_gain

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

        pred_meas = measurement_model.function(predicted_state, **kwargs)

        hh = self._measurement_matrix(predicted_state=predicted_state,
                                      measurement_model=measurement_model,
                                      **kwargs)

        # The measurement cross covariance and innovation covariance
        meas_cross_cov = self._measurement_cross_covariance(predicted_state, hh)
        innov_cov = self._innovation_covariance(meas_cross_cov, hh, measurement_model)

        return MeasurementPrediction.from_state(
            predicted_state, pred_meas, innov_cov, cross_covar=meas_cross_cov)

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
        posterior_mean = self._posterior_mean(predicted_state, kalman_gain,
                                              hypothesis.measurement,
                                              hypothesis.measurement_prediction)

        if self.force_symmetric_covariance:
            posterior_covariance = \
                (posterior_covariance + posterior_covariance.T)/2

        return Update.from_state(
            hypothesis.prediction,
            posterior_mean, posterior_covariance,
            timestamp=hypothesis.measurement.timestamp, hypothesis=hypothesis)


class ExtendedKalmanUpdater(KalmanUpdater):
    r"""The Extended Kalman Filter version of the Kalman Updater. Inherits most
    of the functionality from :class:`~.KalmanUpdater`.

    The difference is that the measurement model may now be non-linear, though
    must be differentiable to return the linearisation of :math:`h(\mathbf{x})`
    via the matrix :math:`H` accessible via :meth:`~.NonLinearModel.jacobian`.

    """
    # TODO: Enforce the fact that this version of MeasurementModel must be
    # TODO: capable of executing :attr:`jacobian()`
    measurement_model: MeasurementModel = Property(
        default=None,
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
    measurement_model: MeasurementModel = Property(
        default=None,
        doc="The measurement model to be used. This need not be defined if a "
            "measurement model is provided in the measurement. If no model "
            "specified on construction, or in the measurement, then error "
            "will be thrown.")
    alpha: float = Property(
        default=0.5,
        doc="Primary sigma point spread scaling parameter. Default is 0.5.")
    beta: float = Property(
        default=2,
        doc="Used to incorporate prior knowledge of the distribution. If the "
            "true distribution is Gaussian, the value of 2 is optimal. "
            "Default is 2")
    kappa: float = Property(
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

        return MeasurementPrediction.from_state(
            predicted_state, meas_pred_mean, meas_pred_covar, cross_covar=cross_covar)


class SqrtKalmanUpdater(KalmanUpdater):
    r"""The Square root version of the Kalman Updater.

    The input :class:`~.State` is a :class:`~.SqrtGaussianState` which means
    that the covariance of the predicted state is stored in square root form.
    This can be achieved by keeping :attr:`covar` attribute as :math:`L` where
    the 'full' covariance matrix :math:`P_{k|k-1} = L_{k|k-1} L^T_{k|k-1}`
    [Eq1].

    In its basic form :math:`L` is the lower triangular matrix returned via
    Cholesky factorisation. There's no reason why other forms that satisfy Eq 1
    above can't be used.

    References
    ----------
    1. Schmidt, S.F. 1970, Computational techniques in Kalman filtering, NATO advisory group for
       aerospace research and development, London 1970
    2. Andrews, A. 1968, A square root formulation of the Kalman covariance equations, AIAA
       Journal, 6:6, 1165-1166

    """
    qr_method: bool = Property(
        default=False,
        doc="A switch to do the update via a QR decomposition, rather than using the (vector form "
            "of) the Potter method.")

    def _measurement_cross_covariance(self, predicted_state, measurement_matrix):
        """
        Return the measurement cross covariance matrix, :math:`P_{k|k-1} H_k^T`. This differs
        slightly from its parent in that it the predicted state covariance (now a square root
        matrix) is transposed.

        Parameters
        ----------
        predicted_state : :class:`SqrtGaussianState`
            The predicted state which contains the square root form of the covariance matrix
            :math:`W` as :attr:`.covar` attribute
        measurement_matrix : numpy.array
            The measurement matrix, :math:`H`

        Returns
        -------
        :  numpy.ndarray
            The measurement cross-covariance matrix

        """
        return predicted_state.sqrt_covar.T @ measurement_matrix.T

    def _innovation_covariance(self, m_cross_cov, meas_mat, meas_mod):
        """Compute the innovation covariance

        Parameters
        ----------
        m_cross_cov : numpy.array
            The measurement cross covariance matrix
        meas_mat : numpy.array
            The measurement matrix. Not required in this instance. Ignored.
        meas_mod : :class:`~.MeasurementModel`
            Measurement model. The class attribute :attr:`sqrt_covar` indicates whether this is
            passed in square root form. If it doesn't exist then :attr:`covar` is assumed to exist
            and is used instead.

        Returns
        -------
        : numpy.ndarray
            The innovation covariance

        """
        # If the measurement covariance matrix is square root then square it
        try:
            meas_cov = meas_mod.sqrt_covar @ meas_mod.sqrt_covar.T
        except AttributeError:
            meas_cov = meas_mod.covar()

        return m_cross_cov.T @ m_cross_cov + meas_cov

    def _posterior_covariance(self, hypothesis):
        """
        Return the posterior covariance for a given hypothesis. Hypothesis contains the predicted
        state covariance in square root form, the measurement prediction (which in turn contains
        the measurement cross covariance, :math:`P_{k|k-1} H_k^T and the innovation covariance,
        :math:`S = H_k P_{k|k-1} H_k^T + R`, not in square root form). The hypothesis or the
        updater contain the measurement noise matrix. The :attr:`sqrt_measurement_noise` flag
        indicates whether we should use the square root form of this matrix (True) or its full
        form (False).

        Parameters
        ----------
        hypothesis: :class:`~.Hypothesis`
            A hypothesised association between state prediction and measurement

        Method
        ------
        If the :attr:`qr_method` flag is set to True then the update proceeds via a QR
        decomposition which requires only one further matrix inversion (see [1]), rather than
        three plus a Cholesky factorisation, for the method set out in [2].

        Returns
        -------
        : numpy.array
            The posterior covariance matrix rendered via the Kalman update process in
            lower-triangular form.
        : numpy.array
            The Kalman gain, :math:`K = P_{k|k-1} H_k^T S^{-1}`

        """
        # Do we already have a measurement model?
        measurement_model = \
            self._check_measurement_model(hypothesis.measurement.measurement_model)
        # Square root of the noise covariance, account for the fact that it may be supplied in one
        # of two ways
        try:
            sqrt_noise_cov = measurement_model.sqrt_covar
        except AttributeError:
            sqrt_noise_cov = la.sqrtm(measurement_model.covar())

        if self.qr_method:
            # The prior and noise covariances and the measurement matrix
            sqrt_prior_cov = hypothesis.prediction.sqrt_covar
            bigh = measurement_model.matrix()

            # Set up and execute the QR decomposition
            measdim = measurement_model.ndim_meas
            zeros = np.zeros((measurement_model.ndim_state, measdim))
            biga = np.block([[sqrt_noise_cov, bigh@sqrt_prior_cov], [zeros, sqrt_prior_cov]])
            _, upper = np.linalg.qr(biga.T)

            # Extract meaningful quantities
            atheta = upper.T
            sqrt_innov_cov = atheta[:measdim, :measdim]
            kalman_gain = atheta[measdim:, :measdim]@(np.linalg.inv(sqrt_innov_cov))
            post_cov = atheta[measdim:, measdim:]
        else:
            # Kalman gain
            kalman_gain = \
                hypothesis.prediction.sqrt_covar @ \
                hypothesis.measurement_prediction.cross_covar @ \
                np.linalg.inv(hypothesis.measurement_prediction.covar)
            # Square root of the innovation covariance
            sqrt_innov_cov = la.sqrtm(hypothesis.measurement_prediction.covar)
            # Posterior covariance
            post_cov = hypothesis.prediction.sqrt_covar @ \
                (np.identity(hypothesis.prediction.ndim) -
                 hypothesis.measurement_prediction.cross_covar @ np.linalg.inv(sqrt_innov_cov.T) @
                 np.linalg.inv(sqrt_innov_cov + sqrt_noise_cov) @
                 hypothesis.measurement_prediction.cross_covar.T)

        return post_cov, kalman_gain


class IteratedKalmanUpdater(ExtendedKalmanUpdater):
    r"""This version of the Kalman updater runs an iteration over the linearisation of the
    sensor function in order to refine the posterior state estimate. Specifically,

    .. math::

        \mathbf{x}_{k,i+1} &= \mathbf{x}_{k|k-1} + K_i [\mathbf{z} - h(\mathbf{x}_{k,i}) -
        H_i (\mathbf{x}_{k|k-1} - \mathbf{x}_{k,i}) ]

        P_{k,i+1} &= (I - K_i H_i) P_{k|k-1}

    where,

    .. math::

        H_i &= h^{\prime}(\mathbf{x}_{k,i}),

        K_i &= P_{k|k-1} H_i^T (H_i P_{k|k-1} H_i^T + R)^{-1}

    and

    .. math::

        \mathbf{x}_{k,0} &= \mathbf{x}_{k|k-1}

        P_{k,0} &= P_{k|k-1}

    It inherits from the ExtendedKalmanUpdater as it uses the same linearisation of the sensor
    function via the :meth:`_measurement_matrix()` function.
    """

    tolerance: float = Property(
        default=1e-6,
        doc="The value of the difference in the measure used as a stopping criterion.")
    measure: Measure = Property(
        default=Euclidean(),
        doc="The measure to use to test the iteration stopping criterion. Defaults to the "
            "Euclidean distance between current and prior posterior state estimate.")
    max_iterations: int = Property(
        default=1000,
        doc="Number of iterations before while loop is exited and a non-convergence warning is "
            "returned")

    def update(self, hypothesis, **kwargs):
        r"""The iterated Kalman update method. Given a hypothesised association between a predicted
        state or predicted measurement and an actual measurement,
        calculate the posterior state.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.
        **kwargs : various
            These are passed to the measurement model function

        Returns
        -------
        : :class:`~.GaussianStateUpdate`
            The posterior state Gaussian with mean :math:`\mathbf{x}_{k|k}` and
            covariance :math:`P_{k|k}`

        """

        # Record the starting point
        prev_state = hypothesis.prediction

        # Get the measurement model
        measurement_model = self._check_measurement_model(hypothesis.measurement.measurement_model)

        # The first iteration is just the application of the EKF
        post_state = super().update(hypothesis, **kwargs)

        # Now update the measurement prediction mean and loop
        iterations = 0
        while self.measure(prev_state, post_state) > self.tolerance:

            if iterations > self.max_iterations:
                warnings.warn("Iterated Kalman update did not converge")
                break

            # These lines effectively bypass the predict_measurement function in update()
            # by attaching new linearised quantities to the measurement_prediction. Those
            # would otherwise be calculated (from the original prediction) by the update() method.
            hh = self._measurement_matrix(post_state, measurement_model=measurement_model)

            post_state.hypothesis.measurement_prediction.state_vector = \
                measurement_model.function(post_state, noise=None) + \
                hh@(hypothesis.prediction.state_vector - post_state.state_vector)

            cross_cov = self._measurement_cross_covariance(hypothesis.prediction, hh)
            post_state.hypothesis.measurement_prediction.cross_covar = cross_cov
            post_state.hypothesis.measurement_prediction.covar = \
                self._innovation_covariance(cross_cov, hh, measurement_model)

            prev_state = post_state
            post_state = super().update(post_state.hypothesis, **kwargs)

            # increment counter
            iterations += 1

        return post_state


class SchmidtKalmanUpdater(ExtendedKalmanUpdater):
    r"""A class which extends the standard Kalman filter to employ the Schmidt-Kalman version of
    the update. The key thing here is that the state vector is split into parameters to be
    estimated, and those which are merely 'considered'. The consider parameters are not updated,
    though their relative covariances are maintained through the process. The state vector,
    covariance and measurement matrix are defined as,

    .. math ::

        \mathbf{x}^T &\triangleq [\mathbf{s}^T \ \mathbf{p}^T]

        H &= [H_s \ H_p]

    .. math ::

        P &= \begin{bmatrix}
        P_{ss} & P_{sp} \\
        P_{ps} & P_{pp}
        \end{bmatrix}


    where the consider parameters are denoted :math:`p` and those to be estimated :math:`s`. Note
    that though they are separated in the definition above, they may be interleaved in practice.
    The update proceeds as:

    .. math ::

       K_s &= (P_{ss,k|k-1} H_s^T + P_{sp,k|k-1} H_p^T) S^{-1},

       \mathbf{s}_{k|k} &= \mathbf{s}_{k|k-1} + K_s (\mathbf{z} - H_s \mathbf{s}_{k|k-1} - H_p
       \mathbf{p}_{k|k-1}),

       \mathbf{p}_{k|k} &= \mathbf{p}_{k|k-1},

    .. math ::

       P_{k|k} &= \begin{bmatrix}
        P_{ss,k|k-1} - K_s S K_s^T &
        P_{sp,k|k-1} - K_s H \begin{bmatrix} P_{sp,k|k-1} \\ P_{pp,k|k-1} \end{bmatrix} \\
        P_{ps,k|k-1} - \begin{bmatrix} P_{sp,k|k-1} \\ P_{pp,k|k-1} \end{bmatrix}^T H^T K_s^T &
        P_{pp,k|k-1}
        \end{bmatrix}


    References
    ----------
    [1] S. F. Schmidt, “Application of State-Space Methods to Navigation Problems,” Advances in
    Control Systems, Vol. 3, 1966, pp. 293–340

    [2] Zanetti, R. & D’Souza, C. (2013). Recursive Implementations of the Schmidt-Kalman
    ‘Consider’   Filter. The Journal of the Astronautical Sciences. 60. 672-685.
    10.1007/s40295-015-0068-7.

    """
    consider: np.ndarray = Property(default=None,
                                    doc="The boolean vector of 'consider' parameters. True "
                                        "indicates considered, False are state parameters to be "
                                        "estimated. If undefined these default to all False, i.e."
                                        "the standard Kalman filter.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.consider is None:
            self.consider = np.zeros(self.measurement_model.ndim_state, dtype=bool)

    def _posterior_mean(self, predicted_state, kalman_gain, measurement, measurement_prediction):
        """Compute the posterior mean, :math:`s_{k|k} = s_{k|k-1} + K_s (z - H_s s_{k|k-1} -
        H_p p_{k|k-1})`, :math:`p_{k|k} = p_{k|k-1}.

        Parameters
        ----------
        predicted_state : :class:`State`, :class:`Prediction`
            The predicted state
        kalman_gain : numpy.ndarray
            The reduced form of the Kalman gain, :math:`K_s`
         measurement : :class:`Detection`
            The measurement
        measurement_prediction : :class:`MeasurementPrediction`
            Predicted measurement

        Returns
        -------
        : :class:`StateVector`
            The posterior mean estimate
        """
        post_mean = predicted_state.state_vector.copy()
        post_mean[np.ix_(~self.consider)] += \
            kalman_gain @ (measurement.state_vector - measurement_prediction.state_vector)
        return post_mean.view(StateVector)

    def _posterior_covariance(self, hypothesis):
        """
        Return the posterior covariance for a given hypothesis

        Parameters
        ----------
        hypothesis: :class:`~.Hypothesis`
            A hypothesised association between state prediction and measurement. It returns the
            measurement prediction which in turn contains the measurement cross covariance,
            :math:`P_{k|k-1} H_k^T and the innovation covariance,
            :math:`S = H_k P_{k|k-1} H_k^T + R`

        Returns
        -------
        : :class:`~.CovarianceMatrix`
            The posterior covariance matrix rendered via the Schmidt-Kalman update process.
        : numpy.ndarray
            The reduced form of the Kalman gain,
            :math:`K_s = (P_{ss,k|k-1} H_{s,k}^T + P_{sp,k|k-1} H_{p,k}^T) S^{-1}`

        """
        # Intermediate matrices P_p and H.
        pp = hypothesis.prediction.covar[np.ix_(range(0, len(self.consider)), self.consider)]
        hh = self._measurement_matrix(predicted_state=hypothesis.prediction)

        # First get the Kalman gain
        mcc = hypothesis.measurement_prediction.cross_covar
        kalman_gain = mcc[np.ix_(~self.consider)] @ \
            np.linalg.inv(hypothesis.measurement_prediction.covar)

        # Then assemble the quadrants of the posterior covariance (easier to think of them as
        # quadrants even though they're actually submatrices who may appear in somewhat different
        # places.)
        post_cov = hypothesis.prediction.covar.copy()
        post_cov[np.ix_(~self.consider, ~self.consider)] -= \
            kalman_gain @ hypothesis.measurement_prediction.covar @ kalman_gain.T
        post_cov[np.ix_(~self.consider, self.consider)] -= kalman_gain @ hh @ pp
        post_cov[np.ix_(self.consider, ~self.consider)] -= pp.T @ hh.T @ kalman_gain.T

        return post_cov.view(CovarianceMatrix), kalman_gain
