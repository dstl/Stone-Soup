from datetime import datetime
from datetime import timedelta

import numpy as np
import pytest

from ...models.measurement.linear import LinearGaussian
from ...models.transition.linear import ConstantVelocity
from ...updater.kalman import KalmanUpdater
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.state import GaussianState
from ...predictor.kalman import KalmanPredictor
from ..simple import SinglePointInitiator
from ..wrapper import StatesLengthLimiter


@pytest.mark.parametrize("max_len", (1, 5, 9))
def test_states_length_limiter(max_len):
    """Test StatesLengthLimiter"""

    # Measurements
    start_time = datetime.now()
    measurements = []
    for i in range(10):
        measurements.append(Detection(np.array([[i*2.0]]),
                                      timestamp=start_time+timedelta(seconds=i*100)))

    # Tracking components
    # ===================
    # transition & measurement models
    transition_model = ConstantVelocity(0.05)
    transition_model.matrix(time_interval=timedelta(seconds=1))
    measurement_model = LinearGaussian(2, [0], np.array([[1]]))

    # Predictor and Updater
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    # Track initiator
    prior = GaussianState(
        np.array([[0], [0]]),
        np.array([[100, 0], [0, 1]]), timestamp=start_time)
    initiator = StatesLengthLimiter(SinglePointInitiator(
        prior,
        measurement_model), max_len)

    track = None
    # Do tracking ...
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)
        posterior = updater.update(hypothesis)
        if track is None:
            track = initiator.initiate([measurement]).pop()
        else:
            track.append(posterior)
        prior = track[-1]

        assert len(track) <= max_len

    assert len(track) == max_len
