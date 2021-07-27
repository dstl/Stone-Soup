# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pytest

from ..categorical import CategoricalHypothesiser
from ..composite import CompositeHypothesiser
from ..probability import PDAHypothesiser
from ...models.measurement.categorical import CategoricalMeasurementModel
from ...models.transition.tests.test_categorical import create_categorical, \
    create_categorical_matrix
from ...predictor.composite import CompositePredictor
from ...types.array import CovarianceMatrix
from ...types.detection import Detection, MissedDetection, CompositeDetection, \
    CompositeMissedDetection, CategoricalDetection
from ...types.hypothesis import CompositeHypothesis, CompositeProbabilityHypothesis
from ...types.multihypothesis import MultipleHypothesis
from ...types.state import GaussianState, CompositeState, CategoricalState
from ...types.track import Track


def make_categorical_measurement_model(ndim_state, ndim_meas, mapping):
    E = create_categorical_matrix(ndim_state, ndim_meas)
    Ecov = CovarianceMatrix(0.1 * np.eye(ndim_meas))

    model = CategoricalMeasurementModel(ndim_state=ndim_state,
                                        emission_matrix=E,
                                        emission_covariance=Ecov,
                                        mapping=mapping)
    return model


def test_composite(predictor, updater, class_predictor, class_updater):
    sub_hypothesisers = [
        PDAHypothesiser(predictor, updater, clutter_spatial_density=1.2e-2, prob_detect=0.9,
                        prob_gate=0.99),
        CategoricalHypothesiser(class_predictor, class_updater,
                                prob_detect=0.7, prob_gate=0.95),
        PDAHypothesiser(predictor, updater, clutter_spatial_density=1.4e-2, prob_detect=0.5,
                        prob_gate=0.98),
        CategoricalHypothesiser(class_predictor, class_updater,
                                prob_detect=0.8, prob_gate=0.97)
    ]

    hypothesiser = CompositeHypothesiser(sub_hypothesisers=sub_hypothesisers)

    # test composite predictor and updater
    expected_predictors = [sub_hyp.predictor for sub_hyp in sub_hypothesisers]

    assert isinstance(hypothesiser.predictor, CompositePredictor)
    assert hypothesiser.predictor.sub_predictors == expected_predictors

    now = datetime.datetime.now()

    track = Track([CompositeState([GaussianState([0, 1, 0, 1], 0.1 * np.eye(4)),
                                   CategoricalState([0.2, 0.2, 0.2, 0.2, 0.2]),
                                   GaussianState([3, 4, 5], 0.2 * np.eye(3)),
                                   CategoricalState([0.3, 0.4, 0.3])])])

    detection1 = CompositeDetection([Detection([3, 3, 3, 3], timestamp=now),
                                     CategoricalDetection(create_categorical(2), timestamp=now),
                                     Detection([2, 4, 6], timestamp=now),
                                     CategoricalDetection(create_categorical(2), timestamp=now)],
                                    mapping=[0, 1, 2, 3])
    detection2 = CompositeDetection([Detection([4, 4, 4, 4], timestamp=now),
                                     CategoricalDetection(create_categorical(2), timestamp=now),
                                     CategoricalDetection(create_categorical(2), timestamp=now)],
                                    mapping=[0, 1, 3])
    detection3 = CompositeDetection([CategoricalDetection(create_categorical(2), timestamp=now),
                                     CategoricalDetection(create_categorical(2), timestamp=now)],
                                    mapping=[3, 1])

    detections = {detection1, detection2, detection3}

    multi_hypothesis = hypothesiser.hypothesise(track, detections, now)

    # Test all hypotheses are composite
    assert all({isinstance(hypothesis, CompositeHypothesis) for hypothesis in multi_hypothesis})

    # Test all detections considered
    hyp_detections = {hypothesis.measurement for hypothesis in multi_hypothesis}
    for detection in detections:
        assert detection in hyp_detections

    # Test composite null hypothesis present (no detections in any sub-state)
    assert any({isinstance(meas, CompositeMissedDetection) for meas in hyp_detections})

    # Test hypothesis for every detection
    assert len(multi_hypothesis) == len(detections) + 1

    # Rest detections completely present in corresponding hypotheses
    for detection in detections:
        det_hyp = None
        count = 0
        for hyp in multi_hypothesis:
            if hyp.measurement == detection:
                det_hyp = hyp
                count += 1
        assert det_hyp is not None
        assert count == 1
        assert isinstance(det_hyp, CompositeProbabilityHypothesis)

        # test mapping correct
        for i, sub_det_hyp in enumerate(det_hyp):
            try:
                mapping_index = detection.mapping.index(i)
            except ValueError:
                assert isinstance(sub_det_hyp.measurement, MissedDetection)
            else:
                assert sub_det_hyp.measurement == detection[mapping_index]

    # Test normalised
    assert pytest.approx(sum({hyp.probability for hyp in multi_hypothesis}), 1)

    sub_hyps_prob_sum = 0
    for hyp in multi_hypothesis:
        prod = 1
        for sub_hyp in hyp:
            prod *= sub_hyp.probability
        sub_hyps_prob_sum += prod

    # Test no detections
    empty_hypotheses = hypothesiser.hypothesise(track, set(), now)
    assert isinstance(empty_hypotheses, MultipleHypothesis)
    assert len(empty_hypotheses) == 1
    null_hyp = next(iter(empty_hypotheses))
    assert not null_hyp
    assert isinstance(null_hyp.measurement, CompositeMissedDetection)
