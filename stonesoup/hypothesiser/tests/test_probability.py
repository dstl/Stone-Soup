# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pytest
from ...models.measurement.categorical import CategoricalMeasurementModel
from ...models.measurement.nonlinear import CartesianToElevationBearingRange
from ..probability import PDAHypothesiser, CategoricalHypothesiser
from ...types.detection import Detection, MissedDetection, CategoricalDetection
from ...types.numeric import Probability
from ...types.state import GaussianState, State
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


def test_categorical(class_predictor, class_updater):
    now = datetime.datetime.now()
    future = now + datetime.timedelta(seconds=5)
    track = Track([State([0.2, 0.3, 0.5], now)])

    # generate random emission matrix and normalise
    E = np.random.rand(3, 3)
    sum_of_rows = E.sum(axis=1)
    E = E / sum_of_rows[:, np.newaxis]

    # create observation-based measurement model
    measurement_model = CategoricalMeasurementModel(ndim_state=3, emission_matrix=E)

    detection1 = CategoricalDetection(state_vector=[1, 0, 0], timestamp=future, measurement_model=measurement_model)
    detection2 = CategoricalDetection(state_vector=[0, 1, 0], timestamp=future, measurement_model=measurement_model)
    detection3 = CategoricalDetection(state_vector=[0, 0, 1], timestamp=future, measurement_model=measurement_model)
    detections = [detection1, detection2, detection3]

    hypothesiser = CategoricalHypothesiser(class_predictor, class_updater,
                                           clutter_spatial_density=1.2e-2,
                                           prob_detect=0.9, prob_gate=0.99)

    multihypothesis = hypothesiser.hypothesise(track, detections, future)

    # test hypotheses - 4 detections, 1 missed
    assert len(multihypothesis) == 4

    # test each hypothesis has a probability/weight attribute
    assert all(hypothesis.probability >= 0 and isinstance(hypothesis.probability, Probability) for
               hypothesis in multihypothesis)

    # test all detections present
    hypothesis_detections = {hypothesis.measurement for hypothesis in multihypothesis}
    for detection in detections:
        assert detection in hypothesis_detections

    # test missed detection present
    assert any(isinstance(measurement, MissedDetection) for measurement in hypothesis_detections)
