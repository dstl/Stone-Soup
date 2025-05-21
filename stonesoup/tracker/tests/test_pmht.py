import numpy as np
import datetime

from ...types.array import StateVector
from ...types.state import GaussianState
from ...models.measurement.linear import LinearGaussian
from ...predictor.kalman import KalmanPredictor
from ...smoother.kalman import KalmanSmoother
from ...updater.kalman import KalmanUpdater
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from ..pmht import PMHTTracker


def test_pmht(detector):

    start_time = datetime.datetime(2018, 1, 1, 14, 0)
    timestamp = datetime.datetime.now().replace(microsecond=0)

    # Initial truth states for fixed number of targets
    preexisting_states = [[-20, 5, 0, 10], [20, -5, 0, 10]]

    # Initial estimate for tracks
    init_means = preexisting_states
    init_cov = np.diag([1.0, 1.0, 1.0, 1.0])
    init_priors = [GaussianState(StateVector(init_mean), init_cov, timestamp=timestamp)
                   for init_mean in init_means]

    measurement_model = LinearGaussian(
        ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
        mapping=(0, 2),  # Mapping measurement vector index to state index
        noise_covar=np.array([[1, 0],  # Covariance matrix for Gaussian PDF
                              [0, 1]])
    )

    q_x = 1.0
    q_y = 1.0
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                              ConstantVelocity(q_y)])

    updater = KalmanUpdater(measurement_model)

    # probability of detection
    detection_probability = 0.9

    # clutter will be generated uniformly in this are around the target
    meas_range = np.array([[-1, 1], [-1, 1]]) * 1000

    # currently use very low clutter rate since PMHT seems to struggle with clutter
    # rate is in mean number of clutter points per scan
    clutter_rate = 1.0e-3

    predictor = KalmanPredictor(transition_model)
    smoother = KalmanSmoother(transition_model)

    # Number of measurement scans to run over for each batch
    batch_len = 10

    # Number of scans to overlap between batches
    overlap_len = 5

    # Maximum number of iterations to run each batch over (currently there is no convergence test
    # so this is the actual number of iterations)
    max_num_iterations = 10

    # Whether to update the prior data association values during iterations (True or False)
    update_log_pi = True

    pmht = PMHTTracker(
        detector=detector,
        predictor=predictor,
        smoother=smoother,
        updater=updater,
        meas_range=meas_range,
        clutter_rate=clutter_rate,
        detection_probability=detection_probability,
        batch_len=batch_len,
        overlap_len=overlap_len,
        init_priors=init_priors,
        max_num_iterations=max_num_iterations,
        update_log_pi=update_log_pi)

    for time, ctracks in pmht:
        assert time > start_time
        assert len(ctracks) == 2
        start_time = time
