# -*- coding: utf-8 -*-
import pytest
import numpy as np

from ..detection import Detection, MissedDetection
from ..multihypothesis import MultipleHypothesis, MultipleCompositeHypothesis
from ..hypothesis import SingleProbabilityHypothesis, SingleDistanceHypothesis, SingleHypothesis
from ..prediction import GaussianStatePrediction, GaussianMeasurementPrediction, StatePrediction
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


def test_multi_composite_hypothesis(composite_hypothesis1,
                                    composite_hypothesis2,
                                    composite_probability_hypothesis1,
                                    composite_probability_hypothesis2):
    # default empty list
    assert MultipleCompositeHypothesis().single_hypotheses == list()

    # error on non-composite hypotheses
    with pytest.raises(ValueError, match="Cannot form MultipleHypothesis out of "
                                         "non-CompositeHypothesis inputs!"):
        MultipleCompositeHypothesis([
            composite_hypothesis1,
            SingleHypothesis(prediction=StatePrediction([3]), measurement=Detection([8]))
        ])

    multi_hypothesis = MultipleCompositeHypothesis([composite_probability_hypothesis1,
                                                    composite_probability_hypothesis2],
                                                   normalise=True)

    # Test normalise
    # error on no-probability hypotheses
    with pytest.raises(ValueError, match="MultipleHypothesis not composed of composite hypotheses "
                                         "with probabilities"):
        MultipleCompositeHypothesis([composite_hypothesis1, composite_hypothesis2], normalise=True)

    assert np.isclose(float(multi_hypothesis.single_hypotheses[0].probability),
                      (0.5 ** 2) / (0.5 ** 2 + 0.2 ** 2))
    assert np.isclose(float(multi_hypothesis.single_hypotheses[1].probability),
                      (0.2 ** 2) / (0.5 ** 2 + 0.2 ** 2))
    multi_hypothesis.normalise_probabilities(total_weight=5)
    assert np.isclose(float(multi_hypothesis.single_hypotheses[0].probability),
                      (5 * 0.5 ** 2) / (0.5 ** 2 + 0.2 ** 2))
    assert np.isclose(float(multi_hypothesis.single_hypotheses[1].probability),
                      (5 * 0.2 ** 2) / (0.5 ** 2 + 0.2 ** 2))

    multi_hypothesis = MultipleCompositeHypothesis([composite_probability_hypothesis1,
                                                    composite_probability_hypothesis2],
                                                   normalise=True)

    # Test len
    assert len(multi_hypothesis) == 2

    # Test contains
    assert composite_probability_hypothesis1 in multi_hypothesis
    assert composite_probability_hypothesis2 in multi_hypothesis
    assert composite_hypothesis1 not in MultipleCompositeHypothesis(
        single_hypotheses=[composite_probability_hypothesis1])
    assert 'a' not in multi_hypothesis

    # Test iter
    for actual_hypothesis, exp_hypothesis in zip(iter(multi_hypothesis),
                                                 [composite_probability_hypothesis1,
                                                  composite_probability_hypothesis2]):
        assert actual_hypothesis == exp_hypothesis

    # Test get
    assert multi_hypothesis[0] == composite_probability_hypothesis1
    assert multi_hypothesis[1] == composite_probability_hypothesis2

    # Test get missed detection probability
    assert multi_hypothesis.get_missed_detection_probability() is None

    # create null-composite-hypothesis
    for sub_hypothesis in composite_probability_hypothesis1:
        sub_hypothesis.measurement = MissedDetection()
    # equal to probability of first composite hypothesis which is now null
    assert np.isclose(float(multi_hypothesis.get_missed_detection_probability()),
                      (0.5 ** 2) / (0.5 ** 2 + 0.2 ** 2))
