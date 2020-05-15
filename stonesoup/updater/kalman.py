# -*- coding: utf-8 -*-
import warnings
from collections import OrderedDict

import numpy as np
import scipy.linalg as la
from functools import lru_cache

from ..base import Property
from .base import Updater
from ..types.array import CovarianceMatrix
from ..types.prediction import GaussianMeasurementPrediction, ASDGaussianMeasurementPrediction
from ..types.state import State
from ..types.update import GaussianStateUpdate, SqrtGaussianStateUpdate, ASDGaussianStateUpdate
from ..models.base import LinearModel
from ..models.measurement.linear import LinearGaussian
from ..models.measurement import MeasurementModel
from ..functions import gauss2sigma, unscented_transform
from ..measures import Measure, Euclidean


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

    # This attribute tells the :meth:`update()` method what class to return
    _update_class = GaussianStateUpdate

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
        meas_cov : :class:~.CovarianceMatrix`
            Measurement covariance matrix

        Returns
        -------
        : numpy.ndarray
            The innovation covariance

        """
        return meas_mat @ m_cross_cov + meas_mod.covar()

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

        # Kalman gain and posterior covariance
        posterior_covariance, kalman_gain = self._posterior_covariance(hypothesis)

        # Posterior mean
        posterior_mean = predicted_state.state_vector + \
            kalman_gain@(hypothesis.measurement.state_vector -
                         hypothesis.measurement_prediction.state_vector)

        if self.force_symmetric_covariance:
            posterior_covariance = \
                (posterior_covariance + posterior_covariance.T)/2

        return self._update_class(posterior_mean, posterior_covariance, hypothesis,
                                  timestamp=hypothesis.measurement.timestamp)


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
        pred_meas = measurement_model.function(State(predicted_state.multi_state_vector[t_index * predicted_state.ndim: (t_index +1) * predicted_state.ndim]),noise=0, **kwargs)


        hh = self._measurement_matrix(predicted_state=predicted_state,
                                      measurement_model=measurement_model,
                                      **kwargs)

        innov_cov = hh @ predicted_state.multi_covar[t_index * predicted_state.ndim: (t_index +1) * predicted_state.ndim, t_index * predicted_state.ndim: (t_index +1) * predicted_state.ndim] @ hh.T + measurement_model.covar()

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


        # save the new posterior, if it is no out of sequence measurement
        try:
            predicted_state.correlation_matrices[predicted_state.act_timestamp]
        except KeyError:
            predicted_state.correlation_matrices[predicted_state.act_timestamp] = {}
        t_index = predicted_state.timestamps.index(predicted_state.act_timestamp)
        ndmin = predicted_state.ndim

        # update covariance after calculating
        predicted_state.correlation_matrices[predicted_state.act_timestamp]['P'] = posterior_covariance[t_index * ndmin: (t_index + 1) * ndmin, t_index * ndmin: (t_index + 1) * ndmin]
        try:
            predicted_state.correlation_matrices[predicted_state.act_timestamp]['PFP'] = \
                predicted_state.correlation_matrices[predicted_state.act_timestamp]['P'] \
                @ predicted_state.correlation_matrices[predicted_state.act_timestamp]['F'].T \
                @ np.linalg.inv(predicted_state.correlation_matrices[predicted_state.act_timestamp]['P_pred'])
        except:
            pass
        correlation_matrices = OrderedDict(sorted(predicted_state.correlation_matrices.items(), reverse=True))

        return ASDGaussianStateUpdate(multi_state_vector=posterior_mean, multi_covar=posterior_covariance,
                                      hypothesis=hypothesis, timestamps=predicted_state.timestamps,
                                      correlation_matrices=correlation_matrices, max_nstep=predicted_state.max_nstep)


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

    # In this instance the square root form is returned by the :meth:`update()`
    _update_class = SqrtGaussianStateUpdate

    def _measurement_cross_covariance(self, predicted_state, measurement_matrix):
        """
        Return the measurement cross covariance matrix, :math:`P_{k~k-1} H_k^T`. This differs
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

        x_{k,i+1} &= x_{k|k-1} + K_i [z - h(x_{k,i}) - H_i (x_{k|k-1} - x_{k,i}) ]

        P_{k,i+1} &= (I - K_i H_i) P_{k|k-1}

    where,

    .. math::

        H_i &= h^{\prime}(x_{k,i}),

        K_i &= P_{k|k-1} H_i^T (H_i P_{k|k-1} H_i^T + R)^{-1}

    and

    .. math::

        x_{k,0} &= x_{k|k-1}

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
