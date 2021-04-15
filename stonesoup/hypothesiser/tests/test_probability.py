# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pytest
from stonesoup.measures import ObservationAccuracy
from stonesoup.models.measurement.classification import BasicTimeInvariantObservationModel
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from ..probability import PDAHypothesiser, ClassificationHypothesiser
from ...types.detection import Detection, MissedDetection
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


def test_classification(class_predictor, class_updater):
    timestamp = datetime.datetime.now()
    track = Track([State([0.2, 0.3, 0.5], timestamp)])

    # generate random emission matrix
    E = np.random.rand(3, 4)

    # create observation-based measurement model
    measurement_model = BasicTimeInvariantObservationModel(E)

    detection1 = Detection([1, 0, 0, 0], timestamp, measurement_model)
    detection2 = Detection([0, 1, 0, 0], timestamp, measurement_model)
    detection3 = Detection([0, 0, 1, 0], timestamp, measurement_model)
    detection4 = Detection([0, 0, 0, 1], timestamp, measurement_model)
    detections = {detection1, detection2, detection3, detection4}

    hypothesiser = ClassificationHypothesiser(class_predictor, class_updater,
                                              clutter_spatial_density=1.2e-2,
                                              prob_detect=0.9, prob_gate=0.99)

    # test default measure
    assert isinstance(hypothesiser.measure, ObservationAccuracy)

    multihypothesis = hypothesiser.hypothesise(track, detections, timestamp)
    # test hypotheses - 4 detections, 1 missed
    assert len(multihypothesis) == 5

    # test each hypothesis has a probability/weight attribute
    assert all(hypothesis.probability >= 0 and isinstance(hypothesis.probability, Probability) for
               hypothesis in multihypothesis)

    # test all detections present
    hypothesis_detections = {hypothesis.measurement for hypothesis in multihypothesis}
    for detection in detections:
        assert detection in hypothesis_detections

    # test missed detection present
    assert any(isinstance(measurement, MissedDetection) for measurement in hypothesis_detections)

    # test errors

    detection1.measurement_model = None
    with pytest.raises(ValueError,
                       match="All observation-based measurements must have corresponding "
                             "measurement models"):
        hypothesiser.hypothesise(track, detections, timestamp)

    detection1.measurement_model = CartesianToElevationBearingRange(mapping=[0, 1],
                                                                    noise_covar=np.eye(4),
                                                                    ndim_state=2)
    with pytest.raises(ValueError,
                       match="ClassificationHypothesiser can only hypothesise observation-based "
                             "measurements with corresponding state space emissions. Therefore an "
                             "emission matrix must be defined in the measurement's corresponding "
                             "measurement model"):
        hypothesiser.hypothesise(track, detections, timestamp)
