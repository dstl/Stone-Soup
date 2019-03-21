# -*- coding: utf-8 -*-
from operator import attrgetter
import datetime

import numpy as np

from ..distance import DistanceHypothesiser
from ...types import Track, Detection, GaussianState
from ... import measures as measures


def test_mahalanobis(predictor, updater):

    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    detection1 = Detection(np.array([[2]]))
    detection2 = Detection(np.array([[3]]))
    detections = {detection1, detection2}

    measure = measures.Mahalanobis()
    hypothesiser = DistanceHypothesiser(predictor, updater, measure=measure)

    hypotheses = hypothesiser.hypothesise(track, detections, timestamp)

    # There are 3 hypotheses - Detection 1, Detection 2, Missed Dectection
    assert len(hypotheses) == 3

    # There is a missed detection hypothesis
    assert any(hypothesis.measurement is None for hypothesis in hypotheses)

    # Each hypothesis has a distance attribute
    assert all(hypothesis.distance >= 0 for hypothesis in hypotheses)

    # The hypotheses are sorted correctly
    assert min(hypotheses, key=attrgetter('distance')) is hypotheses[0]
