import copy
import math

import numpy as np
import scipy

from .ensemble import EnsembleUpdater
from .kalman import ExtendedKalmanUpdater
from ..base import Property
from ..types.prediction import Prediction, EnsembleStatePrediction
from ..types.state import State
from ..types.update import Update
from ..types.array import CovarianceMatrix, StateVectors


class BayesianRecursiveUpdater(ExtendedKalmanUpdater):
    """
    Recursive extension of the ExtendedKalmanUpdater.
    """
    number_steps: int = Property(doc="Number of recursive steps",
                                 default=1)
    use_joseph_cov: bool = Property(doc="Bool dictating the method of covariance calculation. If "
                                        "use_joseph_cov is True then the Joseph form of the "
                                        "covariance equation is used.",
                                    default=False)

    @classmethod
    def _get_meas_cov_scale_factor(cls, n=1, step_no=None):
        return n

    def _innovation_covariance(self, m_cross_cov, meas_mat, meas_mod, scale_factor=1):
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
        return meas_mat@m_cross_cov + scale_factor*meas_mod.covar()

    def _posterior_covariance(self, hypothesis, scale_factor=1):
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
        if self.use_joseph_cov:
            # Identity matrix
            id_matrix = np.identity(hypothesis.prediction.ndim)

            # Calculate Kalman gain
            kalman_gain = hypothesis.measurement_prediction.cross_covar @ \
                np.linalg.inv(hypothesis.measurement_prediction.covar)

            # Calculate measurement matrix/jacobian matrix
            meas_matrix = self._measurement_matrix(hypothesis.prediction)

            # Calculate Prior covariance
            prior_covar = hypothesis.prediction.covar

            # Calculate measurement covariance
            meas_covar = hypothesis.measurement.measurement_model.covar()

            # Compute posterior covariance matrix
            I_KH = id_matrix - kalman_gain @ meas_matrix
            post_cov = I_KH @ prior_covar @ I_KH.T \
                + kalman_gain @ (scale_factor * meas_covar) @ kalman_gain.T

            return post_cov.view(CovarianceMatrix), kalman_gain

        else:
            kalman_gain = hypothesis.measurement_prediction.cross_covar @ \
                np.linalg.inv(hypothesis.measurement_prediction.covar)

            post_cov = hypothesis.prediction.covar - kalman_gain @ \
                hypothesis.measurement_prediction.covar @ kalman_gain.T

            return post_cov.view(CovarianceMatrix), kalman_gain

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

        # Get the measurement model out of the measurement if it's there.
        # If not, use the one native to the updater (which might still be
        # none)
        measurement_model = hypothesis.measurement.measurement_model
        measurement_model = self._check_measurement_model(measurement_model)

        # If there is no measurement prediction in the hypothesis then do the
        # measurement prediction (and attach it back to the hypothesis).
        if hypothesis.measurement_prediction is None:
            # Attach the measurement prediction to the hypothesis
            hypothesis.measurement_prediction = self.predict_measurement(
                predicted_state, measurement_model=measurement_model, **kwargs)

        if not self.number_steps > 0:
            raise ValueError("Updater cannot run 0 times (or less). This would not provide an "
                             "updated state")

        nhypothesis = copy.copy(hypothesis)
        for i in range(1, self.number_steps + 1):

            sf = self._get_meas_cov_scale_factor(self.number_steps, i)

            nhypothesis.measurement_prediction = self.predict_measurement(
                nhypothesis.prediction, measurement_model=measurement_model, scale_factor=sf)
            # Kalman gain and posterior covariance
            posterior_covariance, kalman_gain = self._posterior_covariance(nhypothesis,
                                                                           scale_factor=sf)

            # Posterior mean
            posterior_mean = self._posterior_mean(nhypothesis.prediction, kalman_gain,
                                                  nhypothesis.measurement,
                                                  nhypothesis.measurement_prediction)
            nhypothesis.prediction = Prediction.from_state(
                nhypothesis.prediction, state_vector=posterior_mean, covar=posterior_covariance)

        if self.force_symmetric_covariance:
            posterior_covariance = \
                (posterior_covariance + posterior_covariance.T)/2

        return Update.from_state(
            hypothesis.prediction,
            posterior_mean, posterior_covariance,
            timestamp=hypothesis.measurement.timestamp, hypothesis=hypothesis)


class RecursiveEnsembleUpdater(ExtendedKalmanUpdater, EnsembleUpdater):
    """
    Recursive version of EnsembleUpdater. Uses calculated posterior ensemble as prior ensemble to
    recursively update number_steps times.
    """
    number_steps: int = Property(doc="Number of recursive steps")

    def update(self, hypothesis, **kwargs):
        r"""The BayesianRecursiveEnsembleUpdater update method. The Ensemble Kalman filter
        simply uses the Kalman Update scheme
        to evolve a set or Ensemble
        of state vectors as a group. This ensemble of vectors contains all the
        information on the system state.
        Uses calculated posterior ensemble as prior ensemble to
        recursively update number_steps times.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.
        Returns
        -------
        : :class:`~.EnsembleStateUpdate`
            The posterior state which contains an ensemble of state vectors
            and a timestamp.
        """
        # Assigning more readible variable names
        hypothesis = self._check_measurement_prediction(hypothesis)
        num_vectors = hypothesis.prediction.num_vectors

        nhypothesis = copy.copy(hypothesis)

        if not self.number_steps > 0:
            raise ValueError("Updater cannot run 0 times (or less). This would not provide an "
                             "updated state")

        for _ in range(self.number_steps):

            # Clear measurement prediction so that it is automatically recalculated
            nhypothesis.measurement_prediction = None

            # Generate an ensemble of measurements based on measurement
            measurement_ensemble = nhypothesis.prediction.generate_ensemble(
                mean=hypothesis.measurement.state_vector,
                covar=self.measurement_model.covar(),
                num_vectors=num_vectors)

            # Recalculate measurement prediction
            nhypothesis = self._check_measurement_prediction(nhypothesis)

            # Calculate Kalman Gain according to Dr. Jan Mandel's EnKF formalism.
            innovation_ensemble = nhypothesis.prediction.state_vector - nhypothesis.prediction.mean

            meas_innovation = (
                    self.measurement_model.function(nhypothesis.prediction,
                                                    num_samples=num_vectors)
                    - self.measurement_model.function(State(nhypothesis.prediction.mean)))

            # Calculate Kalman Gain
            kalman_gain = 1 / (num_vectors - 1) * innovation_ensemble @ meas_innovation.T @ \
                scipy.linalg.inv(1 / (num_vectors - 1) * meas_innovation @ meas_innovation.T +
                                 self.measurement_model.covar())

            # Calculate Posterior Ensemble
            posterior_ensemble = (
                    nhypothesis.prediction.state_vector
                    + kalman_gain @ (
                                measurement_ensemble -
                                nhypothesis.measurement_prediction.state_vector))

            nhypothesis.prediction = EnsembleStatePrediction(posterior_ensemble,
                                                             timestamp=nhypothesis.measurement.
                                                             timestamp)

        return Update.from_state(hypothesis.prediction,
                                 posterior_ensemble,
                                 timestamp=hypothesis.measurement.timestamp,
                                 hypothesis=hypothesis)


class RecursiveLinearisedEnsembleUpdater(ExtendedKalmanUpdater, EnsembleUpdater):
    """
    Implementation of 'The Bayesian Recursive Update Linearized EnKF' algorithm from "Ensemble
    Kalman Filter with Bayesian Recursive Update" by Kristen Michaelson, Andrey A. Popov and
    Renato Zanetti.
    Recursive version of the LinearisedEnsembleUpdater that recursively iterates over the update
    step for a given number of steps.

    References
    ----------

    1. K. Michaelson, A. A. Popov and R. Zanetti,
    "Ensemble Kalman Filter with Bayesian Recursive Update"
    """
    number_steps: int = Property(doc="Number of recursive steps")
    inflation_factor: float = Property(default=1.,
                                       doc="Parameter to control inflation")

    def update(self, hypothesis, **kwargs):
        r"""The RecursiveLinearisedEnsembleUpdater update method. Uses an alternative form of
        Kalman gain to calculate a value for each member of the ensemble. Iterates over the update
        step recursively to improve upon error caused by linearisation.


        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.


        Returns
        -------
        : :class:`~.EnsembleStateUpdate`
            The posterior state which contains an ensemble of state vectors
            and a timestamp.
        """

        # Number of steps
        N = self.number_steps

        if not self.number_steps > 0:
            raise ValueError("Updater cannot run 0 times (or less). This would not provide an "
                             "updated state")

        # Preserve original hypothesis - use copy instead
        nhypothesis = copy.copy(hypothesis)

        # Measurement covariance
        R = self.measurement_model.covar()

        # Line 1: Ensemble from prior distribution
        X = nhypothesis.prediction.state_vector

        # Line 2: Iterate the update step
        for _ in range(N):

            # Line 3: Generate mean of prediction ensemble
            m = nhypothesis.prediction.mean

            # Line 4: Compute inflation
            X = StateVectors(m + (self.inflation_factor ** (1/N)) * (X - m))

            # Clear measurement prediction so that it is recalculated
            nhypothesis.measurement_prediction = None

            # Update predicted state vector
            nhypothesis.prediction.state_vector = X

            # Recalculate measurement prediction
            nhypothesis = self._check_measurement_prediction(nhypothesis)

            # Number of vectors
            M = hypothesis.prediction.num_vectors

            # Line 5: Compute prior covariance
            P = 1/(M-1) * (X - m) @ (X - m).T

            # Line 7: Y_hat (vectorised)
            Y_hat = nhypothesis.measurement_prediction.state_vector

            # Line 6
            states = list()
            for x, y_hat in zip(X, Y_hat):

                # Line 8: Compute Jacobian
                H = self.measurement_model.jacobian(
                    State(state_vector=x, timestamp=nhypothesis.measurement.timestamp))

                # Line 9: Compute Innovation
                S = H @ P @ H.T + N * R

                # Line 10: Calculate Kalman gain
                K = P @ H.T @ scipy.linalg.inv(S)

                # Line 11: Recalculate X
                x = x + K @ (hypothesis.measurement.state_vector - y_hat)

                states.append(x)

            X = StateVectors(np.hstack(states))

            nhypothesis.prediction = EnsembleStatePrediction(X,
                                                             timestamp=nhypothesis.measurement.
                                                             timestamp)

        return Update.from_state(hypothesis.prediction,
                                 X,
                                 timestamp=hypothesis.measurement.timestamp,
                                 hypothesis=hypothesis)


class VariableStepBayesianRecursiveUpdater(BayesianRecursiveUpdater):
    """
    Extension of the BayesianRecursiveUpdater. The BayesianRecursiveUpdater uses equal
    measurement noise for each recursive step. The VariableStepBayesianUpdater over-inflates
    measurement noise in the earlier steps, requiring the use of a smaller number of steps.

    References
    ----------

    1. K. Michaelson, A. A. Popov and R. Zanetti,
    "Bayesian Recursive Update for Ensemble Kalman Filters"
    """
    number_steps: int = Property(doc="Number of recursive steps",
                                 default=1)
    use_joseph_cov: bool = Property(doc="Bool dictating the method of covariance calculation. If "
                                        "use_joseph_cov is True then the Joseph form of the "
                                        "covariance equation is used.",
                                    default=False)

    @classmethod
    def _get_meas_cov_scale_factor(cls, n=1, step_no=None):
        return 1 / (step_no / ((n * (n + 1)) / 2))


class ErrorControllerBayesianRecursiveUpdater(BayesianRecursiveUpdater):
    atol: float = Property(doc="TODO")
    rtol: float = Property(doc="TODO")
    f: float = Property(doc="TODO")
    fmin: float = Property(doc="TODO")
    fmax: float = Property(doc="TODO")

    def update(self, hypothesis, **kwargs):

        if not self.number_steps > 0:
            raise ValueError("Updater cannot run 0 times (or less). This would not provide an "
                             "updated state")

        # # 1-2) X0 = prior state estimate
        X_0 = hypothesis.prediction.state_vector
        P_0 = hypothesis.prediction.covar

        # Get the measurement model out of the measurement if it's there.
        # If not, use the one native to the updater (which might still be
        # none)
        measurement_model = hypothesis.measurement.measurement_model
        measurement_model = self._check_measurement_model(measurement_model)

        # If there is no measurement prediction in the hypothesis then do the
        # measurement prediction (and attach it back to the hypothesis).
        if hypothesis.measurement_prediction is None:
            # Attach the measurement prediction to the hypothesis
            hypothesis.measurement_prediction = self.predict_measurement(
                hypothesis.prediction, measurement_model=measurement_model, **kwargs)

        # 3) define initial tc value
        tc = 0

        # 4) define initial ds value
        ds = 1/self.number_steps

        # 5) start iteration count
        i = 1

        # Make two deepcopies of hypothesis
        nhypothesis = copy.deepcopy(hypothesis)
        nhypothesis_prime = copy.deepcopy(hypothesis)

        X_iminus1 = X_0
        P_iminus1 = P_0

        # 7) *** BEGIN WHILE LOOP ***
        while tc < 1:
            print(i)
            print("tc = ", tc)

            # 8 *** BEGIN IF STATEMENT ***
            if tc + ds > 1:

                # 9) update ds value
                ds = 1 - tc

                # 10) *** END OF IF STATEMENT ***

            # Update predicted state
            nhypothesis.prediction = Prediction.from_state(nhypothesis_prime.prediction,
                                                           state_vector=X_iminus1,
                                                           covar=P_iminus1)

            # 12-13) Jacobian, innov_cov calculations using prior state. Note different innov cov method
            nhypothesis.measurement_prediction = self.predict_measurement(
                nhypothesis.prediction, measurement_model=measurement_model, scale_factor=1/ds)

            # 14, 17) Posterior cov and Kalman gain calculation
            P_i, K = self._posterior_covariance(nhypothesis, scale_factor=1/ds)

            # 16) posterior state_vector calculation
            X_i = self._posterior_mean(nhypothesis.prediction, K,
                                                  nhypothesis.measurement,
                                                  nhypothesis.measurement_prediction)

            # 15) deltaX calculation
            deltaX_i = X_i - nhypothesis.prediction.state_vector

            # Set nhypothesis_prime.prediction to be the posterior state
            nhypothesis_prime.prediction = Prediction.from_state(nhypothesis.prediction,
                                                                 state_vector=X_i,
                                                                 covar=P_i)

            # 19-20) Jacobian, innov_cov calculations using posterior state. Note different innov cov method
            nhypothesis_prime.measurement_prediction = self.predict_measurement(
                nhypothesis_prime.prediction, measurement_model=measurement_model,
                scale_factor=1/ds)

            # 21) P' and K' calculation
            P_prime, K_prime = self._posterior_covariance(nhypothesis_prime, scale_factor=1 / ds)

            # 22) deltaX_prime calculation
            deltaX_i_prime = self._posterior_mean(nhypothesis_prime.prediction, K_prime,
                                                  nhypothesis_prime.measurement,
                                                  nhypothesis_prime.measurement_prediction)

            # 23) posterior_state_prime calculation
            X_i_prime = X_iminus1 + 0.5*(deltaX_i + deltaX_i_prime)

            # 24) sc calculation
            # sc = self.atol + max(np.abs(X_i), np.abs(X_i_prime))*self.rtol
            sc = self.atol + max(np.max(np.abs(X_i)), np.max(np.abs(X_i_prime))) * self.rtol

            # 25) error calculation
            # error = max(np.sqrt(np.mean(((1/sc)@(X_i - X_i_prime))**2)))
            error = np.sqrt(np.mean(((1 / sc) * (X_i - X_i_prime)) ** 2))

            # 27) if statement
            if error > 1:
                # 28) update ds
                ds = ds * min(0.9, max(self.fmin, self.f * math.sqrt(1/error)))
                print('ds(1) = ', ds)

                # 29) reject update and begin next iteration
                continue
                # 30) *** END OF IF STATEMENT ***

            # 32) update tc (tc = tc + ds)
            tc = tc + ds

            # 33) posterior_state_vector <- posterior_state_vector_tilda
            X_iminus1 = X_i

            # 34) posterior_cov <- posterior_cov_tilda
            P_iminus1 = P_i

            # 35) update iteration count
            i += 1

            # 36) update ds
            ds = ds * min(self.fmax, max(self.fmin, self.f * math.sqrt(1/error)))
            print('ds(2) = ', ds)

        # 37) *** END OF WHILE LOOP ***

        # 39) final_posterior_state_vector <- posterior_state_vector at i-1

        # 40) final_posterior_cov <- posterior_cov

        return Update.from_state(
            hypothesis.prediction,
            X_iminus1, P_i,
            timestamp=hypothesis.measurement.timestamp, hypothesis=hypothesis)



