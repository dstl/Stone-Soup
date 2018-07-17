import datetime

import numpy as np
import pytest

from stonesoup.models import LinearGaussian
from stonesoup.updater import KalmanUpdater
from stonesoup.types.state import GaussianState, ParticleState
from stonesoup.types.detection import Detection
from stonesoup.initiator import SinglePointInitiator, GaussianParticleInitiator


def test_spi():
    """Test SinglePointInitiator"""

    # Prior state information
    prior_state = GaussianState(
        np.array([[0], [0]]),
        np.array([[100, 0], [0, 1]]))

    # Define a measurement model
    measurement_model = LinearGaussian(2, [0], np.array([[1]]))

    # Define the Initiator
    initiator = SinglePointInitiator(
        prior_state,
        measurement_model)

    # Define 2 detections from which tracks are to be initiated
    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[4.5]]), timestamp),
                  Detection(np.array([[-4.5]]), timestamp)]

    # Run the initiator based on the available detections
    tracks = initiator.initiate(detections)

    # Ensure same number of tracks are initiated as number of measurements
    # (i.e. 2)
    assert(len(tracks) == 2)

    # Ensure that tracks are initiated correctly
    evaluated_tracks = [False, False]
    for detection in detections:

        post_state_vec, post_state_covar, _ =\
            KalmanUpdater.update_lowlevel(prior_state.state_vector,
                                          prior_state.covar,
                                          measurement_model.matrix(),
                                          measurement_model.covar(),
                                          detection.state_vector)

        eval_track_state = GaussianState(
            post_state_vec,
            post_state_covar,
            timestamp=detection.timestamp)

        # Compare against both tracks
        for track_idx, track in enumerate(tracks):
            if(np.array_equal(eval_track_state.mean, track.mean)
               and np.array_equal(eval_track_state.covar, track.covar)):

                evaluated_tracks[track_idx] = True

    # Ensure both tracks have been evaluated
    assert(all(evaluated_tracks))


@pytest.mark.parametrize("gaussian_initiator", [
    SinglePointInitiator(
        GaussianState(np.array([[0]]), np.array([[100]])),
        LinearGaussian(1, [0], np.array([[1]])))
])
def test_gaussian_particle(gaussian_initiator):
    particle_initiator = GaussianParticleInitiator(gaussian_initiator)

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5]]), timestamp),
                  Detection(np.array([[-5]]), timestamp)]

    tracks = particle_initiator.initiate(detections)

    for track in tracks:
        assert isinstance(track.state, ParticleState)
        if track.state_vector > 0:
            assert track.state_vector == pytest.approx(5, 0.2)
        else:
            assert track.state_vector == pytest.approx(-5, 0.2)
        assert track.timestamp == timestamp

        assert track.covar == pytest.approx(1, 0.2)
