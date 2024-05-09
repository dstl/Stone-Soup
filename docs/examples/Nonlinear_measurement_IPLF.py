from abc import ABC
from datetime import datetime
from typing import Union
import matplotlib.pyplot as plt

import numpy as np
from datetime import datetime, timedelta
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.models.transition.nonlinear import GaussianTransitionModel
from stonesoup.models.measurement.nonlinear import NonLinearGaussianMeasurement
from stonesoup.types.state import StateVector, StateVectors, GaussianState, State
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.updater.iterated import IPLFKalmanUpdater
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.models.base import TimeInvariantModel
from stonesoup.types.array import CovarianceMatrix
from stonesoup.base import Property


class MyMeasurementModel(NonLinearGaussianMeasurement):
    """
    Cubic measurement, adapted from https://livrepository.liverpool.ac.uk/3015339/1/PL_smoothing_accepted1.pdf
    """

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        self.n = 3

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
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        # Cubic measurement
        x_meas = [state.state_vector[self.mapping[0]] ** self.n / 20]
        # y_meas = [state.state_vector[self.mapping[1]] ** self.n / 20]

        return StateVectors([x_meas]) + noise


class MyTransitionModel(GaussianTransitionModel, TimeInvariantModel, ABC):
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

        return 1

    def function(self, state: State, noise: Union[bool, np.ndarray] = False,
                 **kwargs) -> Union[StateVector, StateVectors]:
        time_interval_sec = kwargs['time_interval'].total_seconds()
        sv1 = state.state_vector
        sv2 = StateVector(
            [0.9 * sv1[0] + 10 * sv1[0] /(1 + sv1[0] ** 2)])
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

def main():
    start_time = datetime.now().replace(microsecond=0)
    np.random.seed(1991)

    # Create ground truth

    q_x = 0.05
    q_y = 0.05
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                              ConstantVelocity(q_y)])

    transition_model = MyTransitionModel(covariance_matrix=CovarianceMatrix([[1]]))

    timesteps = [start_time]
    truth = GroundTruthPath([GroundTruthState([0], timestamp=timesteps[0])])

    num_steps = 20
    num_steps = 10
    num_steps = 5
    num_steps = 3
    num_steps = 2

    for k in range(1, num_steps + 1):
        timesteps.append(start_time + timedelta(seconds=k))  # add next timestep to list of timesteps
        truth.append(GroundTruthState(
            transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=timesteps[k]))

    print()
    # timesteps = [start_time]
    # truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])
    #
    # num_steps = 20
    # for k in range(1, num_steps + 1):
    #     timesteps.append(start_time + timedelta(seconds=k))  # add next timestep to list of timesteps
    #     truth.append(GroundTruthState(
    #         transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
    #         timestamp=timesteps[k]))

    # Generate measurements

    from stonesoup.types.detection import Detection
    measurement_model = MyMeasurementModel(
        ndim_state=1,  # Number of state dimensions (position and velocity in 2D)
        mapping=(0, ),  # Mapping measurement vector index to state index
        noise_covar=np.array([[1]])
    )

    measurements = []
    for state in truth:
        measurement = measurement_model.function(state, noise=True)
        measurements.append(Detection(measurement,
                                      timestamp=state.timestamp,
                                      measurement_model=measurement_model))
    fig, ax = plt.subplots()
    tr = [state.state_vector.ravel()[0] for state in truth]
    ms = [state.state_vector.ravel()[0] for state in measurements]
    ax.plot(tr, label='truth')
    ax.plot(ms, label='meas')
    # plt.legend()


    predictor = UnscentedKalmanPredictor(transition_model=transition_model, beta=2, kappa=30)
    predictor_iplf = UnscentedKalmanPredictor(transition_model=transition_model, beta=2, kappa=30)
    updater_ukf = UnscentedKalmanUpdater(beta=2, kappa=30)
    updater_iplf = IPLFKalmanUpdater(beta=2, kappa=30, max_iterations=5)

    prior_0 = GaussianState([[0]], np.diag([5]), timestamp=start_time)



    # UKF filtering
    prior = prior_0
    track_ukf = Track([prior])
    prior = track_ukf[-1]
    for i, measurement in enumerate(measurements[1:]):
        if i == 1:
            print()
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater_ukf.update(hypothesis)
        track_ukf.append(post)
        prior = track_ukf[-1]

    tr_ukf = [state.state_vector.ravel()[0] for state in track_ukf]
    ax.plot(tr_ukf, label='ukf')

    # IPLF filtering
    prior = prior_0
    track_iplf = Track([prior])
    prior = track_iplf[-1]
    for i, measurement in enumerate(measurements[1:]):
        print()
        prediction = predictor_iplf.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater_iplf.update(hypothesis)
        track_iplf.append(post)
        prior = track_iplf[-1]

    tr_iplf = [state.state_vector.ravel()[0] for state in track_iplf]
    ax.plot(tr_iplf, label='iplf')
    from matplotlib.ticker import MaxNLocator, FuncFormatter, MultipleLocator
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))

    plt.legend()

    for state in track_iplf:
        if np.any(np.linalg.eigvals(state.covar) < 0):
            print()
    covs_iplf = [state.covar for state in track_iplf]
    covs_ukf = [state.covar for state in track_ukf]

    # IPLS (s)moothing
    from stonesoup.smoother.kalman import IPLSKalmanSmoother
    smoother = IPLSKalmanSmoother(transition_model=transition_model, n_iterations=2)
    track_ipls = smoother.smooth(track_iplf)


    error_ukf = []
    error_iplf = []
    error_ipls = []
    from stonesoup.measures import Euclidean
    mapping = measurement_model.mapping
    for state_true, state_ukf, state_iplf, state_ipls in zip(truth, track_ukf, track_iplf, track_ipls):
        dist_ukf = Euclidean(mapping)(state_true, state_ukf)
        error_ukf.append(dist_ukf)
        dist_iplf = Euclidean(mapping)(state_true, state_iplf)
        error_iplf.append(dist_iplf)
        dist_ipls = Euclidean(mapping)(state_true, state_ipls)
        error_ipls.append(dist_ipls)

    fig, ax = plt.subplots()
    ax.plot(timesteps, error_ukf, label='ukf')
    ax.plot(timesteps, error_iplf, label='iplf')
    ax.plot(timesteps, error_ipls, label='ipls')
    ax.set_xlabel('Time stamp')
    ax.set_ylabel('Absolute error (position)')
    plt.legend()
    plt.show()



    print()

if __name__ == "__main__":
    main()
