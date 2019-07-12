# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..probability import PDAHypothesiser
from ...types.detection import Detection, MissedDetection
from ...types.numeric import Probability
from ...types.state import GaussianState
from ...types.track import Track


def test_pda(predictor, updater):

    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    detection1 = Detection(np.array([[2]]))
    detection2 = Detection(np.array([[8]]))
    detections = {detection1, detection2}

    hypothesiser = PDAHypothesiser(predictor, updater,
                                   clutter_spatial_density=1.2e-2,
                                   prob_detect=0.9, prob_gate=0.99)

    mulltihypothesis = \
        hypothesiser.hypothesise(track, detections, timestamp)

    # There are 3 weighted hypotheses - Detections 1 and 2, MissedDectection
    assert len(mulltihypothesis) == 3

    # Each hypothesis has a probability/weight attribute
    assert all(hypothesis.probability >= 0 and
               isinstance(hypothesis.probability, Probability)
               for hypothesis in
               mulltihypothesis)

    #  Detection 1, 2 and MissedDetection are present
    assert any(hypothesis.measurement is detection1 for hypothesis in
               mulltihypothesis)
    assert any(hypothesis.measurement is detection2 for hypothesis in
               mulltihypothesis)
    assert any(isinstance(hypothesis.measurement, MissedDetection)
               for hypothesis in
               mulltihypothesis)
