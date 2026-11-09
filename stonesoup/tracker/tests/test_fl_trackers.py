import numpy as np
import datetime

from ...types.array import StateVector
from ...types.detection import GaussianDetection
from ...updater.particle import ParticleUpdater
from ...deleter.time import UpdateTimeStepsDeleter
from ...resampler.particle import ESSResampler
from ...predictor.particle import ParticlePredictor
from ..fixed_lag_tracker import FixedLagTracker
from ...types.state import GaussianState
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from ...models.measurement.linear import LinearGaussian
from ...initiator.simple import GaussianParticleInitiator, SimpleMeasurementInitiator


def test_fixed_lag_tracker():

    start_time = datetime.datetime(2018, 1, 1, 14, 0)
    n_particles = 10

    measurement_model = LinearGaussian(
        ndim_state=2,  # Number of state dimensions (position and velocity in 2D)
        mapping=[0],  # Mapping measurement vector index to state index
        noise_covar=np.diag([1])
    )

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(1)])

    prior_state = SimpleMeasurementInitiator(
        prior_state=GaussianState([20, 1], np.diag([2, 1])**2),
        measurement_model=measurement_model)

    deleter = UpdateTimeStepsDeleter(4)

    predictor = ParticlePredictor(transition_model=transition_model)
    updater = ParticleUpdater(measurement_model=measurement_model,
                              resampler=ESSResampler())

    times, scans = [], []
    for step in range(10):
        detections = set()
        new_time = start_time + datetime.timedelta(minutes=step)
        detections.add(GaussianDetection(
            StateVector([[step + 20]]),
            [[2]],
            timestamp=new_time))

        times.append(new_time)
        scans.append(detections)

    # Particle filter initiator
    initiator = GaussianParticleInitiator(
        initiator=prior_state,
        number_particles=n_particles)

    detector = zip(times, scans)
    tracker = FixedLagTracker(
        initiator, deleter, detector, predictor, updater)

    previous_time = datetime.datetime(2018, 1, 1, 13, 59)
    total_tracks = set()
    for time, tracks in tracker:
        assert time == previous_time + datetime.timedelta(minutes=1)
        assert len(tracks) <= 1  # Shouldn't have more than one track
        for track in tracks:
            assert len(track.states) <= 10  # Deleter should delete these
        total_tracks |= tracks

        previous_time = time
