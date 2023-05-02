import datetime
import numpy as np
import pytest

from stonesoup.hypothesiser.simple import SimpleHypothesiser
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track


@pytest.mark.parametrize(
    "check_timestamp, predict_measurement",
    [
     (True, True),
     (False, False)
    ]
)
def test_simple(predictor, updater, check_timestamp, predict_measurement):
    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])

    # Create 3 detections, 2 of which are at the same time
    detection1 = Detection(np.array([[2]]), timestamp)
    detection2 = Detection(np.array([[3]]), timestamp + datetime.timedelta(seconds=1))
    detection3 = Detection(np.array([[10]]), timestamp + datetime.timedelta(seconds=1))
    detections = {detection1, detection2, detection3}

    hypothesiser = SimpleHypothesiser(predictor, updater, check_timestamp, predict_measurement)

    if check_timestamp:
        # Detection 1 has different timestamp to Detections 2 and 3, so this should raise an error
        with pytest.raises(ValueError):
            hypothesiser.hypothesise(track, detections, timestamp)
        return

    hypotheses = hypothesiser.hypothesise(track, detections, timestamp)

    # There are 3 hypotheses - Detection 1, Detection 2, Detection 3, Missed Detection
    assert len(hypotheses) == 4

    # There is a missed detection hypothesis
    assert any(not hypothesis.measurement for hypothesis in hypotheses)

    if predict_measurement:
        # Each true hypothesis has a measurement prediction
        true_hypotheses = [hypothesis for hypothesis in hypotheses if hypothesis]
        assert all(hypothesis.measurement_prediction is not None for hypothesis in true_hypotheses)
    else:
        assert all(hypothesis.measurement_prediction is None for hypothesis in hypotheses)


def test_invalid_simple_arguments(predictor):
    with pytest.raises(ValueError):
        SimpleHypothesiser(predictor, updater=None, predict_measurement=True)
