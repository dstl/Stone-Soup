import datetime

import numpy as np
import pytest

from ..probability import PDAHypothesiser
from ..mfa import MFAHypothesiser
from ...gater.distance import DistanceGater
from ...measures import Euclidean
from ...types.detection import Detection, MissedDetection
from ...types.mixture import GaussianMixture
from ...types.numeric import Probability
from ...types.state import TaggedWeightedGaussianState
from ...types.track import Track
from ...types.update import GaussianMixtureUpdate


@pytest.fixture()
def hypothesiser(predictor, updater):
    return PDAHypothesiser(predictor, updater, 1e-3, include_all=True)


def test_mfa(hypothesiser, updater):
    hypothesiser = MFAHypothesiser(hypothesiser)

    timestamp = datetime.datetime.now()
    track = Track([GaussianMixture(
        [TaggedWeightedGaussianState(
            state_vector=[[0]],
            covar=[[1]],
            weight=Probability(1),
            tag=[],
            timestamp=timestamp)])])
    detection1 = Detection(np.array([[2]]), timestamp)
    detection2 = Detection(np.array([[8]]), timestamp)
    detections = {detection1, detection2}
    detections_tuple = (detection1, detection2)

    # All detections
    multi_hypothesis = hypothesiser.hypothesise(
        track, detections, timestamp, detections_tuple=detections_tuple)

    assert all(detection in multi_hypothesis for detection in detections)
    # Should be on missed detection
    assert sum(1 for hypothesis in multi_hypothesis
               if isinstance(hypothesis.measurement, MissedDetection)) == 1

    for hypothesis in multi_hypothesis:
        assert len(hypothesis.prediction.tag) == 1
        if hypothesis:
            # Index offset by 1
            assert hypothesis.prediction.tag \
                == [detections_tuple.index(hypothesis.measurement) + 1]
        else:
            # 0 means missed detection
            assert hypothesis.prediction.tag == [0]

    # Only detection2
    detections.remove(detection1)
    multi_hypothesis = hypothesiser.hypothesise(
        track, detections, timestamp, detections_tuple=detections_tuple)

    assert detection1 not in multi_hypothesis
    assert detection2 in multi_hypothesis

    for hypothesis in multi_hypothesis:
        assert hypothesis.prediction.tag != 1
        if hypothesis:
            # Index offset by 1; only 2 can be present
            assert hypothesis.prediction.tag == [2]
        else:
            # 0 means missed detection
            assert hypothesis.prediction.tag == [0]

    # Add gate; detection2 outside, so should get missed only
    hypothesiser = DistanceGater(hypothesiser, Euclidean(), 5)
    multi_hypothesis = hypothesiser.hypothesise(
        track, detections, timestamp, detections_tuple=detections_tuple)

    assert len(multi_hypothesis) == 1
    assert not multi_hypothesis[0]
    assert multi_hypothesis[0].prediction.tag == [0]

    # Now with only detection1, which is within
    detections = {detection1}
    multi_hypothesis = hypothesiser.hypothesise(
        track, detections, timestamp, detections_tuple=detections_tuple)

    assert len(multi_hypothesis) == 2
    assert detection1 in multi_hypothesis
    assert detection2 not in multi_hypothesis

    components = []  # Keep for later test
    for hypothesis in multi_hypothesis:
        assert hypothesis.prediction.tag != [2]
        if hypothesis:
            # Index offset by 1; only 1 can be present
            assert hypothesis.prediction.tag == [1]
            components.append(updater.update(hypothesis))
        else:
            # 0 means missed detection
            assert hypothesis.prediction.tag == [0]
            components.append(hypothesis.prediction)

    # Update track; increment time
    track.append(GaussianMixtureUpdate(components=components, hypothesis=multi_hypothesis))
    timestamp += datetime.timedelta(seconds=1)

    # And now check that tag history is built up for
    # all components
    multi_hypothesis = hypothesiser.hypothesise(
        track, {}, timestamp, detections_tuple=tuple())

    assert len(multi_hypothesis) == 2

    for hypothesis in multi_hypothesis:
        assert hypothesis.prediction.tag in ([0, 0], [1, 0])


def test_mfa_bad_timestamp(hypothesiser):
    hypothesiser = MFAHypothesiser(hypothesiser)

    timestamp = datetime.datetime.now()
    detection1 = Detection(np.array([[2]]), timestamp)
    detection2 = Detection(np.array([[8]]), timestamp - datetime.timedelta(seconds=1))
    detections = {detection1, detection2}

    with pytest.raises(ValueError, match="All detections must have the same timestamp"):
        hypothesiser.hypothesise({}, detections, timestamp, tuple(detections))
