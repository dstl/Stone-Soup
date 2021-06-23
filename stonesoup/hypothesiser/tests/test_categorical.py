# -*- coding: utf-8 -*-
from datetime import datetime

from ..categorical import CategoricalHypothesiser
from ...types.detection import CategoricalDetection
from ...types.state import CategoricalState
from ...types.track import Track


def test_categorical(class_predictor, class_updater):
    timestamp = datetime.now()

    track = Track([CategoricalState(state_vector=[0.5, 0.3, 0.2], timestamp=timestamp)])

    detection1 = CategoricalDetection(state_vector=[0.6, 0.4], timestamp=timestamp)
    detection2 = CategoricalDetection(state_vector=[0.5, 0.5], timestamp=timestamp)
    detection3 = CategoricalDetection(state_vector=[0.1, 0.9], timestamp=timestamp)
    detections = {detection1, detection2, detection3}

    hypothesiser = CategoricalHypothesiser(class_predictor, class_updater)

    hypotheses = hypothesiser.hypothesise(track, detections, timestamp)

    # 4 hypotheses: detections 1, 2 and 3, and missed detection
    assert len(hypotheses) == 4

    hypothesis_detections = {hypothesis.measurement for hypothesis in hypotheses}

    # All detections have a hypothesis
    assert all(detection in hypothesis_detections for detection in detections)

    # There is a missed detection hypothesis
    assert any(not hypothesis.measurement for hypothesis in hypotheses)

    # Each hypothesis has a probability attribute
    assert all(hypothesis.probability >= 0 for hypothesis in hypotheses)
