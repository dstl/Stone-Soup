# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np
import pytest

from ..detection import Detection, MissedDetection
from ..hypothesis import (SingleHypothesis, SingleDistanceHypothesis, SingleProbabilityHypothesis,
                          JointHypothesis, ProbabilityJointHypothesis, DistanceJointHypothesis,
                          CompositeHypothesis, CompositeProbabilityHypothesis)
from ..prediction import StatePrediction, StateMeasurementPrediction
from ..track import Track

prediction = StatePrediction(np.array([[1], [0]]))
measurement_prediction = StateMeasurementPrediction(np.array([[1], [0]]))
detection = Detection(np.array([[1], [0]]))
distance = float(1)


def test_single_hypothesis():
    """Single Measurement Hypothesis type test"""

    hypothesis = SingleHypothesis(prediction, detection)
    assert hypothesis.prediction is prediction
    assert hypothesis.measurement is detection
    assert hypothesis.measurement_prediction is None
    assert hypothesis

    hypothesis = SingleHypothesis(prediction, detection,
                                  measurement_prediction)
    assert hypothesis.prediction is prediction
    assert hypothesis.measurement is detection
    assert hypothesis.measurement_prediction is measurement_prediction
    assert hypothesis

    hypothesis = SingleHypothesis(prediction, None)
    assert hypothesis.prediction is prediction
    assert hypothesis.measurement is None
    assert hypothesis.measurement_prediction is None
    assert not hypothesis


def test_single_distance_hypothesis():
    """Single Measurement Distance Hypothesis type test"""

    hypothesis = SingleDistanceHypothesis(
        prediction, detection, distance, measurement_prediction)

    assert hypothesis.prediction is prediction
    assert hypothesis.measurement is detection
    assert hypothesis.distance is distance
    assert hypothesis.measurement_prediction is measurement_prediction
    assert hypothesis.weight == 1/distance

    hypothesis.distance = 0
    assert hypothesis.weight == float('inf')


def test_single_distance_hypothesis_comparison():
    """Single Measurement Distance Hypothesis comparison test"""

    h1 = SingleDistanceHypothesis(
        prediction, detection, distance, measurement_prediction)
    h2 = SingleDistanceHypothesis(
        prediction, detection, distance + 1, measurement_prediction)

    assert h1 > h2
    assert h2 < h1
    assert h1 <= h1
    assert h1 >= h1
    assert h1 == h1


def test_single_probability_hypothesis_comparison():
    """Single Measurement Probability Hypothesis comparison test"""

    h1 = SingleProbabilityHypothesis(
        prediction, detection, 0.9, measurement_prediction)
    h2 = SingleProbabilityHypothesis(
        prediction, detection, 0.1, measurement_prediction)
    h3 = SingleHypothesis(prediction, detection, measurement_prediction)

    assert h1 > h2
    assert h2 < h1
    assert h1 <= h1
    assert h1 >= h1
    assert h1 == h1
    assert h1 != h3


def test_probability_joint_hypothesis():
    """Probability Joint Hypothesis type test"""

    t1 = Track()
    t2 = Track()
    h1 = SingleProbabilityHypothesis(
        prediction, detection, 0.9, measurement_prediction)
    h2 = SingleProbabilityHypothesis(
        prediction, detection, 0.1, measurement_prediction)

    hypotheses = {t1: h1, t2: h2}
    joint_hypothesis = JointHypothesis(hypotheses)

    assert isinstance(joint_hypothesis,
                      ProbabilityJointHypothesis)
    assert joint_hypothesis[t1] is h1
    assert joint_hypothesis[t2] is h2
    assert joint_hypothesis.probability == h1.probability * h2.probability


def test_probability_joint_hypothesis_comparison():
    """Probability Joint Hypothesis comparison test"""

    t1 = Track()
    t2 = Track()
    h1 = SingleProbabilityHypothesis(
        prediction, detection, 0.75, measurement_prediction)
    h2 = SingleProbabilityHypothesis(
        prediction, detection, 0.75, measurement_prediction)
    h3 = SingleProbabilityHypothesis(
        prediction, detection, 0.25, measurement_prediction)

    hypotheses1 = {t1: h1, t2: h2}
    hypotheses2 = {t1: h1, t2: h3}
    j1 = JointHypothesis(hypotheses1)
    j1.normalise()
    j2 = JointHypothesis(hypotheses2)
    j2.normalise()

    assert j1 > j2
    assert j2 < j1
    assert j1 <= j1
    assert j1 >= j1
    assert j1 == j1


def test_distance_joint_hypothesis():
    """Distance Joint Hypothesis type test"""

    t1 = Track()
    t2 = Track()
    h1 = SingleDistanceHypothesis(
        prediction, detection, distance, measurement_prediction)
    h2 = SingleDistanceHypothesis(
        prediction, detection, distance, measurement_prediction)

    hypotheses = {t1: h1, t2: h2}
    joint_hypothesis = JointHypothesis(hypotheses)

    assert isinstance(joint_hypothesis,
                      DistanceJointHypothesis)
    assert joint_hypothesis[t1] is h1
    assert joint_hypothesis[t2] is h2
    assert joint_hypothesis.distance == distance * 2


def test_distance_joint_hypothesis_comparison():
    """Distance Joint Hypothesis comparison test"""

    t1 = Track()
    t2 = Track()
    h1 = SingleDistanceHypothesis(
        prediction, detection, distance, measurement_prediction)
    h2 = SingleDistanceHypothesis(
        prediction, detection, distance, measurement_prediction)
    h3 = SingleDistanceHypothesis(
        prediction, detection, distance + 1, measurement_prediction)

    hypotheses1 = {t1: h1, t2: h2}
    hypotheses2 = {t1: h1, t2: h3}
    j1 = JointHypothesis(hypotheses1)
    j2 = JointHypothesis(hypotheses2)

    assert j1 > j2
    assert j2 < j1
    assert j1 <= j1
    assert j1 >= j1
    assert j1 == j1


def test_invalid_single_joint_hypothesis():
    """Invalid Single Measurement Joint Hypothesis test"""

    t1 = Track()
    t2 = Track()

    h1 = object()
    h2 = object()

    hypotheses = {t1: h1, t2: h2}

    with pytest.raises(NotImplementedError):
        JointHypothesis(hypotheses)


def test_composite_hypothesis(sub_predictions1, composite_prediction1,
                              sub_measurements1, composite_measurement1,
                              sub_hypotheses1, composite_hypothesis1,
                              sub_predictions2, sub_measurements2, sub_hypotheses2):
    # Test empty composite
    with pytest.raises(ValueError, match="Cannot create an empty composite hypothesis"):
        CompositeHypothesis(prediction=composite_prediction1, measurement=composite_measurement1,
                            sub_hypotheses=list())

    # Test bool
    assert composite_hypothesis1
    temp_sub_hypotheses = deepcopy(sub_hypotheses1)
    temp_sub_hypotheses[0].measurement = MissedDetection()
    # not null as second sub-hypothesis is not null
    assert CompositeHypothesis(prediction=composite_prediction1,
                               measurement=composite_measurement1,
                               sub_hypotheses=temp_sub_hypotheses)
    temp_sub_hypotheses[1].measurement = MissedDetection()
    # null as all sub-hypotheses are null
    assert not CompositeHypothesis(prediction=composite_prediction1,
                                   measurement=composite_measurement1,
                                   sub_hypotheses=temp_sub_hypotheses)
    # in the above, sub-hypotheses and sub-measurements don't have correct correspondence
    # which is not the intended use of CompositeHypothesis, but serves the purpose of testing null
    # instances

    # Test contains
    assert sub_predictions1[0] in composite_hypothesis1
    assert sub_predictions2[0] not in composite_hypothesis1
    assert sub_measurements1[1] in composite_hypothesis1
    assert sub_measurements2[1] not in composite_hypothesis1
    assert sub_hypotheses1[0] in composite_hypothesis1
    assert sub_hypotheses2[0] not in composite_hypothesis1
    assert "a" not in composite_hypothesis1

    # Test get
    assert composite_hypothesis1[0] is sub_hypotheses1[0]
    assert composite_hypothesis1[sub_predictions1[1]] is sub_hypotheses1[1]
    assert composite_hypothesis1[sub_predictions2[1]] is None
    assert composite_hypothesis1[sub_measurements1[0]] is sub_hypotheses1[0]
    assert composite_hypothesis1[sub_measurements2[0]] is None
    assert composite_hypothesis1["a"] is None
    # Test get slice
    hypothesis_slice = composite_hypothesis1[1:]
    assert isinstance(hypothesis_slice, CompositeHypothesis)
    assert hypothesis_slice.sub_hypotheses == sub_hypotheses1[1:]

    # Test iter
    for i, sub_state in enumerate(composite_hypothesis1):
        assert sub_state == sub_hypotheses1[i]

    # Test len
    assert len(composite_hypothesis1) == 2


def test_composite_probability_hypothesis(composite_prediction1,
                                          composite_measurement1,
                                          sub_hypotheses1,
                                          sub_probability_hypotheses1,
                                          composite_probability_hypothesis1):
    # Test non-probability sub-hypotheses
    with pytest.raises(ValueError, match="CompositeProbabilityHypothesis must be composed of "
                                         "SingleProbabilityHypothesis types"):
        # sub_hypotheses1 contains non-probability hypotheses
        CompositeProbabilityHypothesis(prediction=composite_prediction1,
                                       measurement=composite_measurement1,
                                       sub_hypotheses=sub_hypotheses1)

    # Test probability

    # product of sub-hypotheses' probabilities
    assert composite_probability_hypothesis1.probability == 0.5 ** 2

    # sub-null-hypotheses
    sub_probability_hypotheses1[0].measurement = MissedDetection()
    # ignore sub-null-hypotheses for non-null-composite-hypothesis
    assert CompositeProbabilityHypothesis(
        prediction=composite_prediction1,
        measurement=composite_measurement1,
        sub_hypotheses=sub_probability_hypotheses1).probability == 0.5

    # null-composite-hypothesis
    sub_probability_hypotheses1[1].measurement = MissedDetection()
    # consider all sub-hypotheses' probabilities if null
    assert CompositeProbabilityHypothesis(
        prediction=composite_prediction1,
        measurement=composite_measurement1,
        sub_hypotheses=sub_probability_hypotheses1).probability == 0.5 ** 2
    # takes argument value if not `None`
    assert CompositeProbabilityHypothesis(
        prediction=composite_prediction1,
        measurement=composite_measurement1,
        sub_hypotheses=sub_probability_hypotheses1, probability=0.4).probability == 0.4
