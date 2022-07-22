# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pytest

from ..distance import DistanceHypothesiser
from ..gaussianmixture import GaussianMixtureHypothesiser, GaussianMixtureKDTreeHypothesiser
from ...types.detection import Detection
from ...types.state import WeightedGaussianState
from ...types.hypothesis import SingleHypothesis
from ...types.multihypothesis import MultipleHypothesis
from ...types.mixture import GaussianMixture
from ... import measures


@pytest.fixture(params=(GaussianMixtureHypothesiser, GaussianMixtureKDTreeHypothesiser))
def hypothesiser(updater, predictor, request):
    measure = measures.Mahalanobis()
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=measure, missed_distance=20)
    kwargs = dict(
        hypothesiser=hypothesiser,
        order_by_detection=True
    )
    if request.param is GaussianMixtureKDTreeHypothesiser:
        kwargs.update(
            predictor=predictor,
            updater=updater,
            max_distance_covariance_multiplier=1,
        )
    return request.param(**kwargs)


def test_gm_ordered_by_measurement(hypothesiser):

    timestamp = datetime.datetime.now()
    gaussian_mixture = GaussianMixture(
        [WeightedGaussianState(
            np.array([[0.3]]), np.array([[1]]), timestamp, 0.4),
            WeightedGaussianState(
                np.array([[5]]), np.array([[0.5]]), timestamp, 0.3)])
    detection1 = Detection(np.array([[1]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detection2 = Detection(np.array([[6.2]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detections = {detection1, detection2}

    hypotheses = hypothesiser.hypothesise(gaussian_mixture,
                                          detections, timestamp)

    # There are 4 hypotheses - 2 each associated with detection1/detection2
    assert all(isinstance(multi_hyp, MultipleHypothesis)
               for multi_hyp in hypotheses)
    assert all(isinstance(hyp, SingleHypothesis)
               for multi_hyp in hypotheses for hyp in multi_hyp)
    # Last element is the miss detected components
    assert len(hypotheses) == 3
    if isinstance(hypothesiser, GaussianMixtureKDTreeHypothesiser):
        assert len(hypotheses[0]) == 1
        assert len(hypotheses[1]) == 1
    else:
        assert len(hypotheses[0]) == 2
        assert len(hypotheses[1]) == 2

    # each SingleHypothesis has a distance attribute
    assert all(hyp.distance >= 0
               for multi_hyp in hypotheses for hyp in multi_hyp)

    # sanity-check the values returned by the hypothesiser
    assert hypotheses[0][0].distance < 10
    assert hypotheses[1][0].distance > 0
    if not isinstance(hypothesiser, GaussianMixtureKDTreeHypothesiser):
        assert hypotheses[0][1].distance > 0
        assert hypotheses[1][1].distance < 10

    # Check the measurements are the same
    assert len(set(hypothesis.measurement for hypothesis in hypotheses[0])) == 1
    assert len(set(hypothesis.measurement for hypothesis in hypotheses[1])) == 1


def test_gm_ordered_by_component(hypothesiser):
    hypothesiser.order_by_detection = False

    timestamp = datetime.datetime.now()
    gaussian_mixture = GaussianMixture(
        [WeightedGaussianState(
            np.array([[0.3]]), np.array([[1]]), timestamp, 0.4),
            WeightedGaussianState(
                np.array([[5]]), np.array([[0.5]]), timestamp, 0.3)])
    detection1 = Detection(np.array([[1]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detection2 = Detection(np.array([[6.2]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detections = {detection1, detection2}

    hypotheses = hypothesiser.hypothesise(gaussian_mixture,
                                          detections, timestamp)

    # There are 6 hypotheses - 2 each associated with detection1/detection
    # then 2 miss detected components
    assert all(isinstance(multi_hyp, MultipleHypothesis)
               for multi_hyp in hypotheses)
    assert all(isinstance(hyp, SingleHypothesis)
               for multi_hyp in hypotheses for hyp in multi_hyp)
    assert len(hypotheses) == 2
    # Last element in each array is the miss detected component
    if isinstance(hypothesiser, GaussianMixtureKDTreeHypothesiser):
        assert len(hypotheses[0]) == 2
        assert len(hypotheses[1]) == 2
    else:
        assert len(hypotheses[0]) == 3
        assert len(hypotheses[1]) == 3

    # each SingleHypothesis has a distance attribute
    assert all(hyp.distance >= 0
               for multi_hyp in hypotheses for hyp in multi_hyp)

    # sanity-check the values returned by the hypothesiser
    assert hypotheses[0][0].distance < 10
    assert hypotheses[1][0].distance > 0
    if not isinstance(hypothesiser, GaussianMixtureKDTreeHypothesiser):
        assert hypotheses[0][1].distance > 0
        assert hypotheses[1][1].distance < 10

    # Check the components are the same
    assert hypotheses[0][0].prediction.state_vector == np.array([[1.3]])
    assert hypotheses[1][0].prediction.state_vector == np.array([[6]])
    if not isinstance(hypothesiser, GaussianMixtureKDTreeHypothesiser):
        assert hypotheses[0][1].prediction.state_vector == np.array([[1.3]])
        assert hypotheses[1][1].prediction.state_vector == np.array([[6]])
