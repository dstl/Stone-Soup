# -*- coding: utf-8 -*-
import numpy as np

from ..detection import Detection
from ..multimeasurementhypothesis import MultipleMeasurementHypothesis, \
    ProbabilityMultipleMeasurementHypothesis
from ..prediction import GaussianStatePrediction, GaussianMeasurementPrediction
from ..numeric import Probability
import pytest


def test_multiplemeasurementhypothesis():
    """MultipleMeasurementHypothesis type test"""
    timestamp = 1
    prediction = GaussianStatePrediction([[1], [1]],
                                         np.diag([10, 10]),
                                         timestamp=timestamp)
    measurement_prediction = \
        GaussianMeasurementPrediction(np.array([[0.991], [1.062]]),
                                      np.diag([2, 2]),
                                      timestamp=timestamp,
                                      cross_covar=np.diag([10, 10]))

    weighted_detections = [Detection([[2.26], [3.03]], timestamp=timestamp),
                           Detection([[1.21], [0.95]], timestamp=timestamp)]
    weights = [6, 14]

    hypothesis = MultipleMeasurementHypothesis(prediction,
                                               measurement_prediction)
    hypothesis.add_weighted_detections(weighted_detections, weights,
                                       normalize=True)

    # check the basic Properties of the MultipleDetectionHypothesis
    assert hypothesis.prediction is prediction
    assert hypothesis.measurement_prediction is measurement_prediction
    assert len(hypothesis.weighted_measurements) == 2
    assert hypothesis.weighted_measurements[0]["measurement"] is \
        weighted_detections[0]
    assert hypothesis.weighted_measurements[0]["weight"] == \
        (weights[0]/sum(weights))
    assert hypothesis.weighted_measurements[1]["measurement"] is \
        weighted_detections[1]
    assert hypothesis.weighted_measurements[1]["weight"] == \
        (weights[1]/sum(weights))

    # check that the MultipleDetectionHypothesis selected_measurement cannot
    # be returned, as it has not been set yet
    with pytest.raises(Exception):
        if hypothesis:
            assert hypothesis.get_selected_measurement() is not None
    with pytest.raises(Exception):
        hypothesis.get_selected_measurement()
    with pytest.raises(Exception):
        a = hypothesis.measurement
        a.clear()

    # try to set hypothesis.selected_measurement with an invalid value
    with pytest.raises(Exception):
        hypothesis.set_selected_measurement(Detection([[5], [12]]))

    # set hypothesis.selected_measurement, verify that it has been set
    hypothesis.set_selected_measurement(weighted_detections[1])
    assert hypothesis.get_selected_measurement() is weighted_detections[1]
    assert hypothesis.measurement is weighted_detections[1]
    if hypothesis:
        assert True

    # create MultipleMeasurementHypothesis with incorrect length/type of
    # weighted_detections and weights
    hypothesis = MultipleMeasurementHypothesis(prediction,
                                               measurement_prediction)

    # incorrect types in 'weighted_detections'
    weighted_detections = [Detection([[2.26], [3.03]], timestamp=timestamp),
                           None]
    weights = [6, 14]
    with pytest.raises(Exception):
        hypothesis.add_weighted_detections(weighted_detections, weights,
                                           normalize=True)

    # incorrect types in 'weights'
    weighted_detections = [Detection([[2.26], [3.03]], timestamp=timestamp),
                           Detection([[1.21], [0.95]], timestamp=timestamp)]
    weights = ['s', 14]
    with pytest.raises(Exception):
        hypothesis.add_weighted_detections(weighted_detections, weights,
                                           normalize=True)

    # mismatching lengths between 'weighted_detections' and 'weights'
    weights = [6, 14, 5]
    with pytest.raises(Exception):
        hypothesis.add_weighted_detections(weighted_detections, weights,
                                           normalize=True)


def test_probabilitymultiplemeasurementhypothesis():
    """ProbabilityMultipleMeasurementHypothesis type test"""

    timestamp = 1
    prediction = GaussianStatePrediction([[1], [1]],
                                         np.diag([10, 10]),
                                         timestamp=timestamp)
    measurement_prediction = \
        GaussianMeasurementPrediction(np.array([[0.991], [1.062]]),
                                      np.diag([2, 2]),
                                      timestamp=timestamp,
                                      cross_covar=np.diag([10, 10]))

    weighted_detections = [Detection([[2.26], [3.03]], timestamp=timestamp),
                           Detection([[1.21], [0.95]], timestamp=timestamp)]
    probabilities = [Probability(6), Probability(14)]

    hypothesis = \
        ProbabilityMultipleMeasurementHypothesis(prediction,
                                                 measurement_prediction)
    hypothesis.add_weighted_detections(weighted_detections,
                                       probabilities, normalize=True)

    # check the basic Properties of the MultipleDetectionHypothesis
    assert hypothesis.prediction is prediction
    assert hypothesis.measurement_prediction is measurement_prediction
    assert len(hypothesis.weighted_measurements) == 2
    assert hypothesis.weighted_measurements[0]["measurement"] is \
        weighted_detections[0]
    assert hypothesis.weighted_measurements[0]["weight"] == \
        (probabilities[0]/sum(probabilities))
    assert hypothesis.weighted_measurements[1]["measurement"] is \
        weighted_detections[1]
    assert hypothesis.weighted_measurements[1]["weight"] == \
        (probabilities[1]/sum(probabilities))
    assert hypothesis.get_missed_detection_probability() is None

    # create MultipleMeasurementHypothesis with incorrect length/type of
    # weighted_detections and weights
    hypothesis = ProbabilityMultipleMeasurementHypothesis(
        prediction, measurement_prediction)

    # incorrect types in 'weighted_detections'
    weighted_detections = [Detection([[2.26], [3.03]], timestamp=timestamp),
                           None]
    probabilities = [Probability(6), Probability(14)]
    with pytest.raises(Exception):
        hypothesis.add_weighted_detections(weighted_detections,
                                           probabilities, normalize=True)

    # incorrect types in 'probabilities'
    weighted_detections = [Detection([[2.26], [3.03]], timestamp=timestamp),
                           Detection([[1.21], [0.95]], timestamp=timestamp)]
    probabilities = [Probability(6), 'g']
    with pytest.raises(Exception):
        hypothesis.add_weighted_detections(weighted_detections,
                                           probabilities, normalize=True)

    # mismatching lengths between 'weighted_detections' and 'probabilities'
    probabilities = [Probability(6), Probability(14), Probability(5)]
    with pytest.raises(Exception):
        hypothesis.add_weighted_detections(weighted_detections,
                                           probabilities, normalize=True)
