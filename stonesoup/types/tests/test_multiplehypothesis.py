# -*- coding: utf-8 -*-
import numpy as np

from ..prediction import StatePrediction, StateMeasurementPrediction
from ..detection import Detection
from ..hypothesis import ProbabilityHypothesis
from ..multiplehypothesis import ProbabilityMultipleHypothesis

prediction0 = StatePrediction(np.array([[1], [0]]))
prediction1 = StatePrediction(np.array([[2], [0]]))
measurement_prediction0 = StateMeasurementPrediction(np.array([[1], [0]]))
measurement_prediction1 = StateMeasurementPrediction(np.array([[2], [0]]))
detection0 = Detection(np.array([[1], [0]]))
detection1 = Detection(np.array([[2], [0]]))
probability0 = 0.65
probability1 = 0.35


def test_probabilitymultiplehypothesis():
    """ProbabilityMultipleHypothesis type test"""

    # without measurement prediction in hypotheses
    hypothesis0 = ProbabilityHypothesis(prediction0, detection0,
                                        probability=probability0)
    hypothesis1 = ProbabilityHypothesis(prediction1, detection1,
                                        probability=probability1)
    multiplehypothesis = ProbabilityMultipleHypothesis(
        None, None, probability=None,
        hypotheses=np.array([hypothesis0, hypothesis1]))
    assert multiplehypothesis.hypotheses[0].prediction is prediction0
    assert multiplehypothesis.hypotheses[0].measurement is detection0
    assert multiplehypothesis.hypotheses[0].measurement_prediction is None
    assert multiplehypothesis.hypotheses[0].probability is probability0
    assert multiplehypothesis.hypotheses[1].prediction is prediction1
    assert multiplehypothesis.hypotheses[1].measurement is detection1
    assert multiplehypothesis.hypotheses[1].measurement_prediction is None
    assert multiplehypothesis.hypotheses[1].probability is probability1

    # with measurement prediction in hypotheses
    hypothesis0 = \
        ProbabilityHypothesis(prediction0, detection0,
                              measurement_prediction=measurement_prediction0,
                              probability=probability0)
    hypothesis1 = \
        ProbabilityHypothesis(prediction1, detection1,
                              measurement_prediction=measurement_prediction1,
                              probability=probability1)
    multiplehypothesis = ProbabilityMultipleHypothesis(
        None, None, None,
        hypotheses=np.array([hypothesis0, hypothesis1]))
    assert multiplehypothesis.hypotheses[0].measurement_prediction is \
        measurement_prediction0
    assert multiplehypothesis.hypotheses[1].measurement_prediction is \
        measurement_prediction1
