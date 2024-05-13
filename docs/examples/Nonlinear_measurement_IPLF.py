from abc import ABC
from datetime import datetime
from typing import Union
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


import numpy as np
from datetime import datetime, timedelta
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.models.transition.nonlinear import GaussianTransitionModel
from stonesoup.models.transition.base import CombinedGaussianTransitionModel
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

        # Cubic (or other power) measurement
        meas = (state.state_vector[self.mapping, :] ** self.power) / self.denominator

        return meas + noise

class CustomNonlinearTransitionModel(GaussianTransitionModel, TimeInvariantModel, ABC):
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

# class CustomNonlinearTransitionModel1D(GaussianTransitionModel, TimeInvariantModel, ABC):
#     covariance_matrix: CovarianceMatrix = Property(
#         doc="Transition noise covariance matrix :math:`\\mathbf{Q}`.")
#
#     @property
#     def ndim_state(self):
#         """ndim_state getter method
#
#         Returns
#         -------
#         : :class:`int`
#             The number of model state dimensions.
#         """
#
#         return 1
#
#     def function(self, state: State, noise: Union[bool, np.ndarray] = False,
#                  **kwargs) -> Union[StateVector, StateVectors]:
#         time_interval_sec = kwargs['time_interval'].total_seconds()
#         sv1 = state.state_vector
#         sv2 = StateVector(
#             [0.9 * sv1[0] + 10 * sv1[0] /(1 + sv1[0] ** 2)])
#         if isinstance(noise, bool) or noise is None:
#             if noise:
#                 noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
#             else:
#                 noise = 0
#         return sv2 + noise
#
#     def rvs(self, num_samples: int = 1, random_state=None, **kwargs) ->\
#             Union[StateVector, StateVectors]:
#         r"""Model noise/sample generation function
#
#         Generates noise samples from the model.
#
#         In mathematical terms, this can be written as:
#
#         .. math::
#
#             v_t \sim \mathcal{N}(0,Q)
#
#         where :math:`v_t =` ``noise`` and :math:`Q` = :attr:`covar`.
#
#         Parameters
#         ----------
#         num_samples: scalar, optional
#             The number of samples to be generated (the default is 1)
#
#         Returns
#         -------
#         noise : 2-D array of shape (:attr:`ndim`, ``num_samples``)
#             A set of Np samples, generated from the model's noise
#             distribution.
#         """
#
#         covar = self.covar(**kwargs)
#
#         # If model has None-type covariance or contains None, it does not represent a Gaussian
#         if covar is None or None in covar:
#             raise ValueError("Cannot generate rvs from None-type covariance")
#
#         random_state = random_state if random_state is not None else self.random_state
#
#         noise = multivariate_normal.rvs(
#             np.zeros(self.ndim), covar, num_samples, random_state=random_state)
#
#         noise = np.atleast_2d(noise)
#
#         if self.ndim > 1:
#             noise = noise.T  # numpy.rvs method squeezes 1-dimensional matrices to integers
#
#         if num_samples == 1:
#             return noise.view(StateVector)
#         else:
#             return noise.view(StateVectors)
#
#     def covar(self, **kwargs):
#         """Returns the transition model noise covariance matrix.
#
#         Returns
#         -------
#         : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
#         (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
#             The process noise covariance.
#         """
#
#         return self.covariance_matrix

class CustomNonlinearTransitionModelVar(GaussianTransitionModel, TimeInvariantModel, ABC):
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
        sv1 = state.state_vector

        sv2_0 = 0.9 * sv1[0] + 10 * sv1[0] / (1 + sv1[0] ** 2)  #  + 8 * np.cos(1.2 * (sv1[1] + 1))
        sv2 = StateVector([sv2_0])

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

class CustomNonlinearTransitionModelStat(GaussianTransitionModel, TimeInvariantModel, ABC):
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
        sv1 = state.state_vector

        sv2 = sv1 + 1

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
    transition_model = CustomNonlinearTransitionModel(covariance_matrix=np.diag([1, 0]))
    # # transition_model = CustomNonlinearTransitionModel1D(covariance_matrix=np.array([[1]]), seed=19915)
    # # x = transition_model.function(State(state_vector=StateVector([1])), time_interval=timedelta(seconds=1))
    # transition_model = CombinedGaussianTransitionModel(
    #     CustomNonlinearTransitionModelVar(covariance_matrix=np.array(np.diag([1]))),
    #     CustomNonlinearTransitionModelStat(covariance_matrix=np.array(np.diag([0])))
    # )
    timesteps = [start_time]
    prior_0 = GaussianState([5, 0], np.diag([4, np.finfo(float).eps]), timestamp=start_time)
    sample = np.random.multivariate_normal(prior_0.state_vector.ravel(), prior_0.covar, 1)
    sample[:, 0] = prior_0.state_vector[0]
    sample[:, 1] = prior_0.state_vector[1]
    truth = GroundTruthPath([GroundTruthState(sample, timestamp=timesteps[0])])

    num_steps = 50

    for k in range(1, num_steps + 1):
        timesteps.append(start_time + timedelta(seconds=k))  # add next timestep to list of timesteps
        truth.append(GroundTruthState(
            transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=timesteps[k]))

    # Generate measurements

    from stonesoup.types.detection import Detection
    measurement_model = CustomNonlinearMeasurementModel(
        power=3,
        denominator=20,
        ndim_state=2,  # Number of state dimensions (position and velocity in 2D)
        mapping=(0, ),  # Mapping measurement vector index to state index
        noise_covar=np.array([[1]])
    )

    measurements = []
    for state in truth:
        measurement = measurement_model.function(state, noise=True)
        measurements.append(Detection(measurement,
                                      timestamp=state.timestamp,
                                      measurement_model=measurement_model))
    # measurements = measurements[1:]
    fig, ax = plt.subplots()
    tr = [state.state_vector.ravel()[0] for state in truth]
    tr_timestamps = [state.timestamp for state in truth]
    ms = [state.state_vector.ravel()[0] for state in measurements]
    ms_timestamps = [state.timestamp for state in measurements]




    predictor = UnscentedKalmanPredictor(transition_model=transition_model, beta=2, kappa=30)
    predictor_iplf = UnscentedKalmanPredictor(transition_model=transition_model, beta=2, kappa=30)
    updater_ukf = UnscentedKalmanUpdater(beta=2, kappa=30)
    updater_iplf = IPLFKalmanUpdater(beta=2, kappa=30, max_iterations=5)




    # UKF filtering
    prior = prior_0
    prior.state_vector[1] = -1
    track_ukf = Track([prior])
    prior = track_ukf[-1]
    for i, measurement in enumerate(measurements):
        if i == 1:
            print()
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater_ukf.update(hypothesis)
        track_ukf.append(post)
        prior = track_ukf[-1]

    tr_ukf = [state.state_vector.ravel()[0] for state in track_ukf]
    ukf_timestamps = [state.timestamp for state in track_ukf]



    # IPLF filtering
    prior = prior_0
    prior.state_vector[1] = -1
    track_iplf = Track([prior])
    prior = track_iplf[-1]
    for i, measurement in enumerate(measurements):
        print()
        prediction = predictor_iplf.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater_iplf.update(hypothesis)
        track_iplf.append(post)
        prior = track_iplf[-1]

    tr_iplf = [state.state_vector.ravel()[0] for state in track_iplf]
    iplf_timestamps = [state.timestamp for state in track_iplf]


    for state in track_iplf:
        if np.any(np.linalg.eigvals(state.covar) < 0):
            print()
    covs_iplf = [state.covar for state in track_iplf]
    covs_ukf = [state.covar for state in track_ukf]

    # IPLS (s)moothing
    from stonesoup.smoother.kalman import IPLSKalmanSmoother, UnscentedKalmanSmoother
    smoother = IPLSKalmanSmoother(transition_model=transition_model, n_iterations=0, beta=2, kappa=30)
    # smoother_ukf = UnscentedKalmanSmoother()
    track_ipls = smoother.smooth(track_iplf)
    tr_ipls = [state.state_vector.ravel()[0] for state in track_ipls]
    ipls_timestamps = [state.timestamp for state in track_ipls]

    ax.plot(ukf_timestamps, tr_ukf, label='ukf')
    ax.plot(iplf_timestamps, tr_iplf, label='iplf')
    ax.plot(ipls_timestamps, tr_ipls, label='ipls')
    ax.plot(tr_timestamps, tr, label='truth', color='k')
    ax.scatter(ms_timestamps, ms, label='meas', marker='x')



    plt.legend()

    error_ukf = []
    error_iplf = []
    error_ipls = []
    from stonesoup.measures import Euclidean
    mapping = measurement_model.mapping
    # mapping = (0, 1)
    for state_true, state_ukf, state_iplf, state_ipls in zip(truth, track_ukf[1:], track_iplf[1:], track_ipls[1:]):
        dist_ukf = Euclidean(mapping)(state_true, state_ukf)
        error_ukf.append(dist_ukf)
        dist_iplf = Euclidean(mapping)(state_true, state_iplf)
        error_iplf.append(dist_iplf)
        dist_ipls = Euclidean(mapping)(state_true, state_ipls)
        error_ipls.append(dist_ipls)

    fig, ax = plt.subplots()
    # timesteps = timesteps[1:]
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
