from abc import ABC
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from datetime import datetime, timedelta
from stonesoup.base import Property
from stonesoup.measures import Euclidean
from stonesoup.models.base import TimeInvariantModel
from stonesoup.models.transition.nonlinear import GaussianTransitionModel
from stonesoup.models.measurement.nonlinear import NonLinearGaussianMeasurement
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.smoother.kalman import IPLSKalmanSmoother
from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import StateVector, StateVectors, GaussianState, State
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.updater.iterated import IPLFKalmanUpdater

class CustomNonlinearMeasurementModel(NonLinearGaussianMeasurement):
    """
    Cubic measurement, adapted from https://livrepository.liverpool.ac.uk/3015339/1/PL_smoothing_accepted1.pdf
    """
    power: int = Property(default=3, doc='Raised to the power of.')
    denominator: float = Property(default=20, doc='Denominator.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 1

    def function(self, state, noise=False, **kwargs) -> StateVector:

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        # Cubic (or other power) measurement
        meas = (state.state_vector[self.mapping, :] ** self.power) / self.denominator

        return meas + noise

class CustomNonlinearTransitionModel(GaussianTransitionModel, TimeInvariantModel, ABC):
    """
    Dynamic model, adapted from https://livrepository.liverpool.ac.uk/3015339/1/PL_smoothing_accepted1.pdf
    """
    covariance_matrix: CovarianceMatrix = Property(
        doc="Transition noise covariance matrix :math:`\\mathbf{Q}`.")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of model state dimensions.
        """

        return 2

    def function(self, state: State, noise: Union[bool, np.ndarray] = False,
                 **kwargs) -> Union[StateVector, StateVectors]:
        sv1 = state.state_vector

        sv2_0 = 0.9 * sv1[0] + 10 * sv1[0] / (1 + sv1[0] ** 2) + 8 * np.cos(1.2 * (sv1[1] + 1))
        sv2_1 = sv1[1] + 1
        sv2 = StateVector([sv2_0, sv2_1])

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0
        return sv2 + noise

    def covar(self, **kwargs):
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """

        return self.covariance_matrix


def do_tracking(prior, measurements, predictor, updater):

    track = Track([prior])
    for i, measurement in enumerate(measurements):
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater.update(hypothesis)
        track.append(post)
        prior = track[-1]

    return track


def main():

    start_time = datetime.now().replace(microsecond=0)
    np.random.seed(1991)

    # Create ground truth
    transition_model = CustomNonlinearTransitionModel(covariance_matrix=np.diag([1, 0]))
    # Prior is set as in paper and the first true state is sampled from it
    prior = GaussianState([5, 0], np.diag([4, np.finfo(float).eps]), timestamp=start_time)
    sample = multivariate_normal.rvs(prior.state_vector.ravel(), prior.covar, size=1)
    truth = GroundTruthPath([GroundTruthState(sample, timestamp=prior.timestamp)])

    num_steps = 50
    timesteps = [start_time]
    for k in range(1, num_steps + 1):
        timesteps.append(start_time + timedelta(seconds=k))  # add next timestep to list of timesteps
        truth.append(GroundTruthState(
            transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=timesteps[k]))

    # Generate measurements
    measurement_model = CustomNonlinearMeasurementModel(
        power=3,
        denominator=20,
        ndim_state=2,  # Number of state dimensions (position and velocity in 2D)
        mapping=(0, ),  # Mapping measurement vector index to state index
        noise_covar=np.array([[1]])
    )

    measurements = []
    for state in truth[1:]:
        measurement = measurement_model.function(state, noise=True)
        measurements.append(Detection(measurement,
                                      timestamp=state.timestamp,
                                      measurement_model=measurement_model))

    # Define the algorithms
    beta, kappa = 2, 30
    predictor = UnscentedKalmanPredictor(transition_model=transition_model, beta=beta, kappa=kappa)
    updater_ukf = UnscentedKalmanUpdater(beta=beta, kappa=kappa)
    updater_iplf = IPLFKalmanUpdater(beta=beta, kappa=kappa, max_iterations=5)
    smoother = IPLSKalmanSmoother(transition_model=transition_model, n_iterations=2, beta=2, kappa=30)


    # Do UKF/IPLF/IPLS
    track_ukf = do_tracking(prior, measurements, predictor, updater_ukf)
    track_iplf = do_tracking(prior, measurements, predictor, updater_iplf)
    track_ipls = smoother.smooth(track_iplf)


    # Plotting the results
    fig0, ax0 = plt.subplots()
    timestamps = [state.timestamp for state in truth]
    timestamps_meas = [state.timestamp for state in measurements]
    ax0.plot(timestamps, [state.state_vector.ravel()[0] for state in truth], label='truth', color='k')
    ax0.scatter(timestamps_meas, [state.state_vector.ravel()[0] for state in measurements], label='meas', marker='x')
    ax0.plot(timestamps, [state.state_vector.ravel()[0] for state in track_ukf], label='ukf')
    ax0.plot(timestamps, [state.state_vector.ravel()[0] for state in track_iplf], label='iplf')
    ax0.plot(timestamps, [state.state_vector.ravel()[0] for state in track_ipls], label='ipls')
    ax0.set_xlabel('Time stamp')
    ax0.set_ylabel('Value')
    plt.legend()

    # Plotting the performance
    fig1, ax1 = plt.subplots()
    error_ukf = []
    error_iplf = []
    error_ipls = []
    for state_true, state_ukf, state_iplf, state_ipls in zip(truth, track_ukf, track_iplf, track_ipls):
        error_ukf.append(Euclidean()(state_true, state_ukf))
        error_iplf.append(Euclidean()(state_true, state_iplf))
        error_ipls.append(Euclidean()(state_true, state_ipls))

    ax1.plot(timesteps, error_ukf, label='ukf')
    ax1.plot(timesteps, error_iplf, label='iplf')
    ax1.plot(timesteps, error_ipls, label='ipls')
    ax1.set_xlabel('Time stamp')
    ax1.set_ylabel('Absolute error (position)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
