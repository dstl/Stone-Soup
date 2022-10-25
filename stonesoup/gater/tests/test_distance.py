import datetime
import pytest
import numpy as np

from ..distance import DistanceGater
from ...hypothesiser.probability import PDAHypothesiser
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.track import Track
from ...types.update import GaussianStateUpdate
from ... import measures as measures

measure = measures.Mahalanobis()


@pytest.mark.parametrize(
    "detections, gate_threshold, num_gated",
    [
        (   # Test 1
            {Detection(np.array([[2]])), Detection(np.array([[3]])), Detection(np.array([[6]])),
             Detection(np.array([[0]])), Detection(np.array([[-1]])), Detection(np.array([[-4]]))},
            1,
            3
        ),
        (   # Test 2
            {Detection(np.array([[2]])), Detection(np.array([[3]])), Detection(np.array([[6]])),
             Detection(np.array([[0]])), Detection(np.array([[-1]])), Detection(np.array([[-4]]))},
            2,
            5
        ),
        (   # Test 3
            {Detection(np.array([[2]])), Detection(np.array([[3]])), Detection(np.array([[6]])),
             Detection(np.array([[0]])), Detection(np.array([[-1]])), Detection(np.array([[-4]]))},
            4,
            7
        )
    ],
    ids=["test1", "test2", "test3"]
)
def test_distance(predictor, updater, detections, gate_threshold, num_gated):

    timestamp = datetime.datetime.now()

    hypothesiser = PDAHypothesiser(
        predictor, updater, clutter_spatial_density=0.000001, include_all=True)
    gater = DistanceGater(hypothesiser, measure=measure, gate_threshold=gate_threshold)

    track = Track([GaussianStateUpdate(
                    np.array([[0]]),
                    np.array([[1]]),
                    SingleHypothesis(
                        None,
                        Detection(np.array([[0]]), metadata={"MMSI": 12345})),
                    timestamp=timestamp)])

    hypotheses = gater.hypothesise(track, detections, timestamp)

    # The number of gated hypotheses matches the expected
    assert len(hypotheses) == num_gated

    # The gated hypotheses are either the null hypothesis or their distance is less than the set
    # gate threshold
    assert all(not hypothesis.measurement or
               measure(hypothesis.measurement_prediction, hypothesis.measurement) < gate_threshold
               for hypothesis in hypotheses)

    # There is a SINGLE missed detection hypothesis
    assert len([hypothesis for hypothesis in hypotheses if not hypothesis]) == 1
