import datetime

import numpy as np
import pytest

from ...models.measurement.linear import LinearGaussian
from ...updater.kalman import KalmanUpdater
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.state import GaussianState, ParticleState
from ..simple import (
    SinglePointInitiator, LinearMeasurementInitiator, GaussianParticleInitiator
)


def test_spi():
    """Test SinglePointInitiator"""

    # Prior state information
    prior_state = GaussianState(
        np.array([[0], [0]]),
        np.array([[100, 0], [0, 1]]))

    # Define a measurement model
    measurement_model = LinearGaussian(2, [0], np.array([[1]]))

    # Create the Kalman updater
    kup = KalmanUpdater(measurement_model)

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

        hypo = SingleHypothesis(prediction=prior_state, measurement=detection)
        eval_track_state = kup.update(hypo)

        # Compare against both tracks
        for track_idx, track in enumerate(tracks):

            if(np.array_equal(eval_track_state.mean, track.mean)
               and np.array_equal(eval_track_state.covar, track.covar)):

                evaluated_tracks[track_idx] = True

    # Ensure both tracks have been evaluated
    assert(all(evaluated_tracks))

    assert set(detections) == set(track.state.hypothesis.measurement
                                  for track in tracks)


def test_linear_measurement():
    measurement_model = LinearGaussian(2, [0], np.array([[50]]))
    measurement_initiator = LinearMeasurementInitiator(
        GaussianState(np.array([[0], [0]]), np.diag([100, 10])),
        measurement_model
    )

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5]]), timestamp),
                  Detection(np.array([[-5]]), timestamp)]

    tracks = measurement_initiator.initiate(detections)

    for track in tracks:
        if track.state_vector[0, 0] > 0:
            assert np.array_equal(track.state_vector, np.array([[5], [0]]))
            assert np.array_equal(
                measurement_model.matrix()@track.state_vector,
                detections[0].state_vector)
            assert track.state.hypothesis.measurement is detections[0]
        else:
            assert np.array_equal(track.state_vector, np.array([[-5], [0]]))
            assert np.array_equal(
                measurement_model.matrix()@track.state_vector,
                detections[1].state_vector)
            assert track.state.hypothesis.measurement is detections[1]

        assert track.timestamp == timestamp

        assert np.array_equal(track.covar, np.diag([50, 10]))


def test_linear_measurement_non_direct():

    class _LinearMeasurementModel:
        ndim_state = 2
        ndmim_meas = 2

        @staticmethod
        def matrix():
            return np.array([[0, 1], [2, 0]])

        @staticmethod
        def covar():
            return np.diag([10, 50])

    measurement_model = _LinearMeasurementModel()
    measurement_initiator = LinearMeasurementInitiator(
        GaussianState(np.array([[0], [0]]), np.diag([100, 10])),
        measurement_model
    )

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5], [2]]), timestamp),
                  Detection(np.array([[-5], [8]]), timestamp)]

    tracks = measurement_initiator.initiate(detections)

    for track in tracks:
        if track.state_vector[1, 0] > 0:
            assert np.array_equal(track.state_vector, np.array([[1], [5]]))
            assert np.array_equal(
                measurement_model.matrix()@track.state_vector,
                detections[0].state_vector)
            assert track.state.hypothesis.measurement is detections[0]
        else:
            assert np.array_equal(track.state_vector, np.array([[4], [-5]]))
            assert np.array_equal(
                measurement_model.matrix()@track.state_vector,
                detections[1].state_vector)
            assert track.state.hypothesis.measurement is detections[1]

        assert track.timestamp == timestamp

        assert np.array_equal(track.covar, np.diag([25, 10]))


def test_linear_measurement_extra_state_dim():

    class _LinearMeasurementModel:
        ndim_state = 3
        ndmim_meas = 2

        @staticmethod
        def matrix():
            return np.array([[1, 0, 0], [0, 0, 1]])

        @staticmethod
        def covar():
            return np.diag([10, 50])

    measurement_model = _LinearMeasurementModel()
    measurement_initiator = LinearMeasurementInitiator(
        GaussianState(np.array([[0], [0], [0]]), np.diag([100, 10, 500])),
        measurement_model
    )

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5], [2]]), timestamp),
                  Detection(np.array([[-5], [8]]), timestamp)]

    tracks = measurement_initiator.initiate(detections)

    for track in tracks:
        if track.state_vector[0, 0] > 0:
            assert np.array_equal(
                track.state_vector,
                np.array([[5], [0], [2]]))
            assert np.array_equal(
                measurement_model.matrix()@track.state_vector,
                detections[0].state_vector)
            assert track.state.hypothesis.measurement is detections[0]
        else:
            assert np.array_equal(
                track.state_vector,
                np.array([[-5], [0], [8]]))
            assert np.array_equal(
                measurement_model.matrix()@track.state_vector,
                detections[1].state_vector)
            assert track.state.hypothesis.measurement is detections[1]

        assert track.timestamp == timestamp

        assert np.array_equal(track.covar, np.diag([10, 10, 50]))


@pytest.mark.parametrize("gaussian_initiator", [
    SinglePointInitiator(
        GaussianState(np.array([[0]]), np.array([[100]])),
        LinearGaussian(1, [0], np.array([[1]]))
    ),
    LinearMeasurementInitiator(
        GaussianState(np.array([[0]]), np.array([[100]])),
        LinearGaussian(1, [0], np.array([[1]]))
    ),
], ids=['SinglePoint', 'LinearMeasurement'])
def test_gaussian_particle(gaussian_initiator):
    particle_initiator = GaussianParticleInitiator(gaussian_initiator)

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5]]), timestamp),
                  Detection(np.array([[-5]]), timestamp)]

    tracks = particle_initiator.initiate(detections)

    for track in tracks:
        assert isinstance(track.state, ParticleState)
        if track.state_vector > 0:
            assert np.allclose(track.state_vector, np.array([[5]]), atol=0.4)
            assert track.state.hypothesis.measurement is detections[0]
        else:
            assert np.allclose(track.state_vector, np.array([[-5]]), atol=0.4)
            assert track.state.hypothesis.measurement is detections[1]
        assert track.timestamp == timestamp

        assert np.allclose(track.covar, np.array([[1]]), atol=0.4)
