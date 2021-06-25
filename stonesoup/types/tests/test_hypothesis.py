# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ..prediction import StatePrediction, StateMeasurementPrediction, CompositePrediction, \
    CompositeMeasurementPrediction
from ..detection import Detection, CompositeDetection, CompositeMissedDetection
from ..track import Track
from ..hypothesis import (
    SingleHypothesis,
    SingleDistanceHypothesis,
    SingleProbabilityHypothesis,
    JointHypothesis,
    ProbabilityJointHypothesis,
    DistanceJointHypothesis, CompositeHypothesis, CompositeProbabilityHypothesis)

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
    assert hypothesis.weight == 1 / distance

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

    assert h1 > h2
    assert h2 < h1
    assert h1 <= h1
    assert h1 >= h1
    assert h1 == h1


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


composite_prediction = CompositePrediction([StatePrediction([0]), StatePrediction([1, 2])])
composite_measurement_prediction = CompositeMeasurementPrediction([StateMeasurementPrediction([6]),
                                                                   StateMeasurementPrediction(
                                                                       [3])])
composite_detection = CompositeDetection(sub_states=[Detection([7]), Detection([3])],
                                         mapping=[1, 0])


def sub_hypotheses():
    hypotheses = list()
    for sub_pred, sub_meas_pred, sub_det in zip(composite_prediction,
                                                composite_measurement_prediction,
                                                composite_detection):
        sub_hypothesis = SingleHypothesis(sub_pred, sub_det, sub_meas_pred)
        hypotheses.append(sub_hypothesis)
    return hypotheses


def sub_prob_hypotheses():
    hypotheses = list()
    for sub_pred, sub_det in zip(composite_prediction, composite_detection):
        sub_hypothesis = SingleProbabilityHypothesis(prediction=sub_pred,
                                                     measurement=sub_det,
                                                     probability=0.5)
        hypotheses.append(sub_hypothesis)
    return hypotheses


def test_composite_hypothesis():
    sub_hyps = sub_hypotheses()

    # Test default sub-hypotheses
    assert CompositeHypothesis(None, None).sub_hypotheses == list()

    composite_hypothesis = CompositeHypothesis(composite_prediction,
                                               composite_measurement_prediction,
                                               sub_hyps)

    assert composite_hypothesis.sub_hypotheses == sub_hyps

    # Test True
    assert composite_hypothesis

    # Test False
    assert not CompositeHypothesis(composite_prediction,
                                   CompositeMissedDetection(default_timestamp=1),
                                   composite_measurement_prediction)

    # Test measurement prediction
    assert composite_hypothesis.measurement_prediction.sub_states == \
           composite_measurement_prediction.sub_states

    # Test contains
    for sub_hyp in sub_hyps:
        assert sub_hyp in composite_hypothesis
    assert 'a' not in composite_hypothesis

    # Test get
    for i in range(len(sub_hypotheses())):
        assert composite_hypothesis[i] == sub_hyps[i]

    # Test iter
    for i, sub_state in enumerate(composite_hypothesis):
        assert sub_state == sub_hyps[i]

    a = SingleHypothesis(StatePrediction([0]), Detection([7]), StateMeasurementPrediction([6]))

    # Test insert
    composite_hypothesis.insert(1, a)
    assert composite_hypothesis[1] == a

    # Test del
    del composite_hypothesis[1]
    assert composite_hypothesis.sub_hypotheses == sub_hyps

    # Test set
    composite_hypothesis[1] = a
    assert composite_hypothesis[1] == a

    # Test len
    assert len(composite_hypothesis) == len(sub_hyps)

    composite_hypothesis = CompositeHypothesis(composite_prediction,
                                               composite_detection,
                                               sub_hyps)

    # Test append
    composite_hypothesis.append(a)
    assert composite_hypothesis[-1] == a
    # 'a' is appended to 'sub_hyps' too
    assert composite_hypothesis.sub_hypotheses == sub_hyps


def test_composite_probability_hypothesis():
    with pytest.raises(ValueError, match="CompositeProbabilityHypothesis must be comprised of "
                                         "SingleProbabilityHypothesis types"):
        hypotheses = sub_hypotheses()
        CompositeProbabilityHypothesis(composite_prediction,
                                       composite_detection,
                                       sub_hypotheses=hypotheses)
    hypotheses = sub_prob_hypotheses()
    composite_hypothesis = CompositeProbabilityHypothesis(composite_prediction,
                                                          composite_detection,
                                                          probability=1,
                                                          sub_hypotheses=hypotheses)
    assert composite_hypothesis.probability == 1

    composite_hypothesis = CompositeProbabilityHypothesis(composite_prediction,
                                                          composite_detection,
                                                          sub_hypotheses=hypotheses)
    assert composite_hypothesis.probability == 0.5 ** len(hypotheses)
