# -*- coding: utf-8 -*-
from operator import attrgetter
import datetime

import numpy as np

from ..distance import DistanceHypothesiser
from ...types.detection import Detection
from ...types.state import GaussianState
from ...types.track import Track
from ... import measures


def test_mahalanobis(predictor, updater):

    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    detection1 = Detection(np.array([[2]]))
    detection2 = Detection(np.array([[3]]))
    detection3 = Detection(np.array([[10]]))
    detections = {detection1, detection2, detection3}

    measure = measures.Mahalanobis()
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=measure, missed_distance=3)

    hypotheses = hypothesiser.hypothesise(track, detections, timestamp)

    # There are 3 hypotheses - Detection 1, Detection 2, Missed Detection
    assert len(hypotheses) == 3

    # And not detection3
    assert detection3 not in {hypothesis.measurement
                              for hypothesis in hypotheses}

    # There is a missed detection hypothesis
    assert any(not hypothesis.measurement for hypothesis in hypotheses)

    # Each hypothesis has a distance attribute
    assert all(hypothesis.distance >= 0 for hypothesis in hypotheses)

    # The hypotheses are sorted correctly
    assert min(hypotheses, key=attrgetter('distance')) is hypotheses[0]
