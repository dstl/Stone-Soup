# -*- coding: utf-8 -*-
import datetime
from operator import attrgetter

import numpy as np
import pytest

from ..distance import DistanceHypothesiser, MultiDistanceHypothesiser
from ... import measures
from ...measures import Euclidean
from ...models.measurement.linear import LinearGaussian
from ...models.measurement.nonlinear import CartesianToBearingRange
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from ...predictor.kalman import KalmanPredictor
from ...types.detection import Detection
from ...types.state import GaussianState
from ...types.track import Track
from ...updater.kalman import ExtendedKalmanUpdater


def test_mahalanobis(predictor, updater):

    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    detection1 = Detection(np.array([[2]]))
    detection2 = Detection(np.array([[3]]))
    detection3 = Detection(np.array([[10]]))
    detections = {detection1, detection2, detection3}

    measure = measures.Mahalanobis()
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=measure, missed_distance=3)

    hypotheses = hypothesiser.hypothesise(track, detections, timestamp)

    # There are 3 hypotheses - Detection 1, Detection 2, Missed Detection
    assert len(hypotheses) == 3

    # And not detection3
    assert detection3 not in {hypothesis.measurement
                              for hypothesis in hypotheses}

    # There is a missed detection hypothesis
    assert any(not hypothesis.measurement for hypothesis in hypotheses)

    # Each hypothesis has a distance attribute
    assert all(hypothesis.distance >= 0 for hypothesis in hypotheses)

    # The hypotheses are sorted correctly
    assert min(hypotheses, key=attrgetter('distance')) is hypotheses[0]


def test_distance_include_all(predictor, updater):

    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    detection1 = Detection(np.array([[2]]))
    detection2 = Detection(np.array([[3]]))
    detection3 = Detection(np.array([[10]]))
    detections = {detection1, detection2, detection3}

    measure = measures.Mahalanobis()
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=measure, missed_distance=1,
        include_all=True)

    hypotheses = hypothesiser.hypothesise(track, detections, timestamp)

    # There are 4 hypotheses - Detections and Missed Detection
    assert len(hypotheses) == 4

    # detection3 is beyond missed distance and largest distance (last
    # hypothesis in list)
    last_hypothesis = hypotheses[-1]
    assert last_hypothesis.measurement is detection3
    assert last_hypothesis.distance > hypothesiser.missed_distance


def test_multi_distance(predictor, updater):
    measurement_models = [
        LinearGaussian(ndim_state=4, mapping=(0, 2), noise_covar=np.eye(2)),
        CartesianToBearingRange(ndim_state=4, mapping=(0, 2),
                                noise_covar=np.diag([np.radians(1) ** 2, 10 ** 2]))
    ]

    predictor = KalmanPredictor(CombinedLinearGaussianTransitionModel(2 * [ConstantVelocity(0.1)]))

    hypothesisers = [
        DistanceHypothesiser(predictor,
                             updater=ExtendedKalmanUpdater(),
                             measure=Euclidean((0, 1)),  # consider all elements of linear Gaussian
                             missed_distance=2,
                             include_all=True),  # include missed detections
        DistanceHypothesiser(predictor,
                             updater=ExtendedKalmanUpdater(),
                             measure=Euclidean(0),  # consider bearing element of bearing-range
                             missed_distance=np.radians(30),
                             include_all=False)  # don't include missed detections
    ]

    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[1, 0, 1, 0]]), np.eye(4), timestamp)])

    detection1 = Detection(np.array([[1, 1]]), measurement_model=measurement_models[0],
                           timestamp=timestamp)
    detection2 = Detection(np.array([[3, 3]]), measurement_model=measurement_models[0],
                           timestamp=timestamp)
    detection3 = Detection(np.array([[np.radians(45)]]), measurement_model=measurement_models[1],
                           timestamp=timestamp)
    detection4 = Detection(np.array([[np.radians(120)]]), measurement_model=measurement_models[1],
                           timestamp=timestamp)
    detection5 = Detection(np.array([8, 8]),
                           measurement_model=LinearGaussian(ndim_state=4,
                                                            mapping=(0, 1),
                                                            noise_covar=np.eye(1)),
                           timestamp=timestamp)

    detections = [detection1, detection2, detection3, detection4, detection5]

    # Test mismatch sequence length error
    with pytest.raises(ValueError, match="Number of hypothesisers must equal number of "
                                         "measurement models in MultiDistanceHypothesiser"):
        MultiDistanceHypothesiser(hypothesisers[1:], measurement_models)

    hypothesiser = MultiDistanceHypothesiser(hypothesisers, measurement_models)

    with pytest.warns(UserWarning, match="Defaulting to first hypothesiser"):
        hypotheses = hypothesiser.hypothesise(track, detections, timestamp)

    # There are 4 hypotheses - Detections and Missed Detection but not for detection4
    assert len(hypotheses) == 5

    for i, detection in enumerate(detections, 1):
        detection_hypothesis = [hypothesis
                                for hypothesis in hypotheses
                                if hypothesis.measurement is detection]
        # Test all detections except for detection4 are accounted for
        if i == 4:
            assert len(detection_hypothesis) == 0
            continue
        assert len(detection_hypothesis) == 1

        detection_hypothesis = detection_hypothesis.pop()

        if i == 1:
            # detection1 within hypothesiser1 missed distance
            assert detection_hypothesis.distance < hypothesisers[0].missed_distance
        elif i in (2, 5):
            # detection2 outside of hypothesiser1 missed distance
            assert detection_hypothesis.distance > hypothesisers[0].missed_distance
        else:
            # detection3 within hypothesiser2 missed distance
            assert detection_hypothesis.distance < hypothesisers[1].missed_distance
