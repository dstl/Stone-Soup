# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..probability import PDAHypothesiser
from ...types import Track, Detection, GaussianState, Probability, \
    MissedDetection


def test_pda(predictor, updater):

    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    detection1 = Detection(np.array([[2]]))
    detection2 = Detection(np.array([[8]]))
    detections = {detection1, detection2}

    hypothesiser = PDAHypothesiser(predictor, updater,
                                   clutter_spatial_density=1.2e-2,
                                   prob_detect=0.9, prob_gate=0.99)

    mulltimeasurehypothesis = \
        hypothesiser.hypothesise(track, detections, timestamp)

    # There are 3 weighted detections - Detections 1 and 2, MissedDectection
    assert len(mulltimeasurehypothesis.weighted_measurements) == 3

    # Each measurements has a probability/weight attribute
    assert all(detection["weight"] >= 0 and
               isinstance(detection["weight"], Probability)
               for detection in mulltimeasurehypothesis.weighted_measurements)

    #  Detection 1, 2 and MissedDetection are present
    assert any(detection["measurement"] is detection1 for detection in
               mulltimeasurehypothesis.weighted_measurements)
    assert any(detection["measurement"] is detection2 for detection in
               mulltimeasurehypothesis.weighted_measurements)
    assert any(isinstance(detection["measurement"], MissedDetection)
               for detection in mulltimeasurehypothesis.weighted_measurements)
