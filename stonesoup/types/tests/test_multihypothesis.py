# -*- coding: utf-8 -*-
import pytest
import numpy as np

from ..detection import Detection
from ..multihypothesis import MultipleHypothesis
from ..hypothesis import SingleProbabilityHypothesis, SingleDistanceHypothesis
from ..prediction import GaussianStatePrediction, GaussianMeasurementPrediction
from ..numeric import Probability


def test_multiplehypothesis():
    """MultipleHypothesis type test"""
    timestamp = 1
    prediction = GaussianStatePrediction([[1], [1]],
                                         np.diag([10, 10]),
                                         timestamp=timestamp)
    measurement_prediction = \
        GaussianMeasurementPrediction(np.array([[0.991], [1.062]]),
                                      np.diag([2, 2]),
                                      timestamp=timestamp,
                                      cross_covar=np.diag([10, 10]))
    detection1 = Detection([[2.26], [3.03]], timestamp=timestamp)
    detection2 = Detection([[1.21], [0.95]], timestamp=timestamp)

    probability_hypothesis_1 = SingleProbabilityHypothesis(
        prediction, detection1,
        measurement_prediction=measurement_prediction,
        probability=Probability(0.1))
    probability_hypothesis_2 = SingleProbabilityHypothesis(
        prediction, detection2,
        measurement_prediction=measurement_prediction,
        probability=Probability(0.2))

    multihypothesis = MultipleHypothesis(
        [probability_hypothesis_1, probability_hypothesis_2])

    # check the basic Properties of the MultipleDetectionHypothesis
    assert (hypothesis.prediction is prediction for hypothesis in
            multihypothesis)
    assert (hypothesis.measurement_prediction is measurement_prediction
            for hypothesis in multihypothesis)
    assert len(multihypothesis) == 2
    assert multihypothesis[0] is \
        probability_hypothesis_1
    assert multihypothesis[1] is \
        probability_hypothesis_2
    assert multihypothesis[detection1].measurement is \
        detection1
    assert multihypothesis[prediction].prediction is \
        prediction
    assert probability_hypothesis_1 in multihypothesis
    assert prediction in multihypothesis
    assert detection1 in multihypothesis

    prediction2 = GaussianStatePrediction([[1], [2]], np.diag([10, 10]),
                                          timestamp=timestamp)
    detection3 = Detection([[2.26], [3.82]], timestamp=timestamp)
    assert prediction2 not in multihypothesis
    assert detection3 not in multihypothesis
    assert multihypothesis[prediction2] is None
    assert multihypothesis[detection3] is None

    multihypothesis = MultipleHypothesis(
        [probability_hypothesis_1, probability_hypothesis_2],
        normalise=True, total_weight=1)

    assert sum(hyp.weight for hyp in multihypothesis).log_value < 1e-12


def test_multiplehypothesis_edge_cases():
    """MultipleHypothesis error casses tests"""

    timestamp = 1
    prediction = GaussianStatePrediction([[1], [1]],
                                         np.diag([10, 10]),
                                         timestamp=timestamp)
    measurement_prediction = \
        GaussianMeasurementPrediction(np.array([[0.991], [1.062]]),
                                      np.diag([2, 2]),
                                      timestamp=timestamp,
                                      cross_covar=np.diag([10, 10]))

    probability_hypothesis_1 = SingleProbabilityHypothesis(
        prediction, Detection([[2.26], [3.03]], timestamp=timestamp),
        measurement_prediction=measurement_prediction,
        probability=Probability(0.1))
    probability_hypothesis_2 = SingleProbabilityHypothesis(
        prediction, Detection([[1.21], [0.95]], timestamp=timestamp),
        measurement_prediction=measurement_prediction,
        probability=Probability(0.2))
    distance_hypothesis_1 = SingleDistanceHypothesis(
        prediction, Detection([[1.21], [0.95]]), 0.5,
        measurement_prediction=measurement_prediction)

    # test case where a non-Hypothesis is passed into MultipleHypothesis
    # constructor
    with pytest.raises(ValueError):
        multihypothesis = MultipleHypothesis(
            [probability_hypothesis_1, probability_hypothesis_2, 'a'])

    # test case where a non-ProbabilityHypothesis is in a
    # MultipleHypothesis and "normalise_probabilities() is called
    multihypothesis = MultipleHypothesis(
        [probability_hypothesis_1, probability_hypothesis_2,
         distance_hypothesis_1])
    with pytest.raises(ValueError):
        multihypothesis.normalise_probabilities(1)

    # test case where 'normalise' is called with 'total_weight'
    # not equal to a number
    with pytest.raises(TypeError):
        multihypothesis = MultipleHypothesis(
            [probability_hypothesis_1, probability_hypothesis_2],
            normalise=True, total_weight='a')

    # test the case where "get_missed_detection_probability()" is called on a
    # MultipleHypothesis that has no MissedDetection
    multihypothesis = MultipleHypothesis(
        [probability_hypothesis_1, probability_hypothesis_2])
    assert multihypothesis.get_missed_detection_probability() is None

    # test the case where no SingleHypotheses are passed in
    multihypothesis = MultipleHypothesis()
    assert len(multihypothesis) == 0
