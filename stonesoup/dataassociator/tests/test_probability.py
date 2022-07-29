import datetime

import pytest
import numpy as np

from ..probability import PDA, JPDA
from ...types.detection import Detection, MissedDetection
from ...types.state import GaussianState
from ...types.track import Track
try:
    from ...plugins.pyehm import JPDAWithEHM, JPDAWithEHM2
except ImportError:
    JPDAWithEHM = None
    JPDAWithEHM2 = None


@pytest.fixture(
    params=[
        PDA,
        JPDA,
        pytest.param(
            JPDAWithEHM,
            marks=pytest.mark.skipif(JPDAWithEHM is None, reason="pyehm required")),
        pytest.param(
            JPDAWithEHM2,
            marks=pytest.mark.skipif(JPDAWithEHM2 is None, reason="pyehm required")),
    ])
def associator(request, probability_hypothesiser):
    return request.param(probability_hypothesiser)


def test_probability(associator):

    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    d1 = Detection(np.array([[0, 0]]), timestamp)
    d2 = Detection(np.array([[5, 5]]), timestamp)

    tracks = {t1, t2}
    detections = {d1, d2}

    associations = associator.associate(tracks, detections, timestamp)

    # There should be 2 associations
    assert len(associations) == 2

    # verify association probabilities are correct
    prob_t1_d1_association = [hyp.probability for hyp in associations[t1]
                              if hyp.measurement is d1]
    prob_t1_d2_association = [hyp.probability for hyp in associations[t1]
                              if hyp.measurement is d2]
    prob_t2_d1_association = [hyp.probability for hyp in associations[t2]
                              if hyp.measurement is d1]
    prob_t2_d2_association = [hyp.probability for hyp in associations[t2]
                              if hyp.measurement is d2]

    assert prob_t1_d1_association[0] > prob_t1_d2_association[0]
    assert prob_t2_d1_association[0] < prob_t2_d2_association[0]


def test_missed_detection_probability(associator):

    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    d1 = Detection(np.array([[20, 20]]), timestamp)

    tracks = {t1, t2}
    detections = {d1}

    associations = associator.associate(tracks, detections, timestamp)

    # Best hypothesis should be missed detection hypothesis
    max_track1_prob = max([hyp.probability for hyp in associations[t1]])
    max_track2_prob = max([hyp.probability for hyp in associations[t1]])

    track1_missed_detect_prob = max(
        [hyp.probability for hyp in associations[t1]
         if isinstance(hyp.measurement, MissedDetection)])
    track2_missed_detect_prob = max(
        [hyp.probability for hyp in associations[t1]
         if isinstance(hyp.measurement, MissedDetection)])

    assert max_track1_prob == track1_missed_detect_prob
    assert max_track2_prob == track2_missed_detect_prob


def test_no_detections_probability(associator):

    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])

    tracks = {t1, t2}
    detections = {}

    associations = associator.associate(tracks, detections, timestamp)

    # All hypotheses should be missed detection hypothesis
    assert all(isinstance(hypothesis.measurement, MissedDetection)
               for multihyp in associations.values()
               for hypothesis in multihyp)


def test_no_tracks_probability(associator):

    timestamp = datetime.datetime.now()
    d1 = Detection(np.array([[2, 2]]), timestamp)
    d2 = Detection(np.array([[5, 5]]), timestamp)

    tracks = {}
    detections = {d1, d2}

    associations = associator.associate(tracks, detections, timestamp)

    # Since no Tracks went in, there should be no associations
    assert not associations
