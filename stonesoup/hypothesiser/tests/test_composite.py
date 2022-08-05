import datetime

import numpy as np
import pytest

from ..categorical import HMMHypothesiser
from ..composite import CompositeHypothesiser
from ..probability import PDAHypothesiser
from ...models.measurement.categorical import MarkovianMeasurementModel
from ...predictor.composite import CompositePredictor
from ...types.detection import Detection, MissedDetection, CompositeDetection, CategoricalDetection
from ...types.hypothesis import CompositeHypothesis, CompositeProbabilityHypothesis
from ...types.multihypothesis import MultipleHypothesis
from ...types.state import GaussianState, CompositeState, CategoricalState
from ...types.track import Track


def make_categorical_measurement_model(ndim_state, ndim_meas):
    E = np.random.rand(ndim_state, ndim_meas)

    model = MarkovianMeasurementModel(emission_matrix=E)
    return model


def test_composite(predictor, updater, dummy_category_predictor, dummy_category_updater):
    sub_hypothesisers = [
        PDAHypothesiser(predictor, updater, clutter_spatial_density=1.2e-2, prob_detect=0.9,
                        prob_gate=0.99),
        HMMHypothesiser(dummy_category_predictor, dummy_category_updater,
                        prob_detect=0.7, prob_gate=0.95),
        PDAHypothesiser(predictor, updater, clutter_spatial_density=1.4e-2, prob_detect=0.5,
                        prob_gate=0.98),
        HMMHypothesiser(dummy_category_predictor, dummy_category_updater,
                        prob_detect=0.8, prob_gate=0.97)
    ]

    # Test instantiation errors
    with pytest.raises(ValueError, match="Cannot create an empty composite hypothesiser"):
        CompositeHypothesiser(sub_hypothesisers=list())

    with pytest.raises(ValueError, match="All sub-hypothesisers must be a hypothesiser type"):
        CompositeHypothesiser(sub_hypothesisers + [1, 2, 3])

    hypothesiser = CompositeHypothesiser(sub_hypothesisers=sub_hypothesisers)

    # Test composite predictor and updater
    expected_predictors = [sub_hyp.predictor for sub_hyp in sub_hypothesisers]

    assert isinstance(hypothesiser.predictor, CompositePredictor)
    assert hypothesiser.predictor.sub_predictors == expected_predictors

    then = datetime.datetime.now()
    now = then + datetime.timedelta(seconds=5)

    track = Track([CompositeState([GaussianState([0, 1, 0, 1], 0.1 * np.eye(4)),
                                   CategoricalState([0.2, 0.2, 0.2, 0.2, 0.2]),
                                   GaussianState([3, 4, 5], 0.2 * np.eye(3)),
                                   CategoricalState([0.3, 0.4, 0.3])],
                                  default_timestamp=then)])

    detection1 = CompositeDetection([Detection([3, 3, 3, 3], timestamp=now),
                                     CategoricalDetection(np.random.rand(2), timestamp=now),
                                     Detection([2, 4, 6], timestamp=now),
                                     CategoricalDetection(np.random.rand(2), timestamp=now)],
                                    mapping=[0, 1, 2, 3])
    detection2 = CompositeDetection([Detection([4, 4, 4, 4], timestamp=now),
                                     CategoricalDetection(np.random.rand(2), timestamp=now),
                                     CategoricalDetection(np.random.rand(2), timestamp=now)],
                                    mapping=[0, 1, 3])
    detection3 = CompositeDetection([CategoricalDetection(np.random.rand(2), timestamp=now),
                                     CategoricalDetection(np.random.rand(2), timestamp=now)],
                                    mapping=[3, 1])

    detections = {detection1, detection2, detection3}

    multi_hypothesis = hypothesiser.hypothesise(track, detections, now)

    # Test all hypotheses are composite
    assert all({isinstance(hypothesis, CompositeHypothesis) for hypothesis in multi_hypothesis})

    # Test all detections considered
    hyp_detections = {hypothesis.measurement for hypothesis in multi_hypothesis}
    for detection in detections:
        assert detection in hyp_detections

    # Test hypothesis for every detection
    assert len(multi_hypothesis) == len(detections) + 1

    # Test detections completely present in corresponding hypotheses
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
    assert pytest.approx(sum({hyp.probability for hyp in multi_hypothesis})) == 1

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

    # Test contains
    for sub_hypothesiser in sub_hypothesisers:
        assert sub_hypothesiser in hypothesiser
    assert PDAHypothesiser(predictor, updater, clutter_spatial_density=1, prob_detect=1,
                           prob_gate=1) not in hypothesiser
    assert 'a' not in hypothesiser

    # Test get
    for i, expected_hypothesiser in enumerate(sub_hypothesisers):
        assert hypothesiser[i] == expected_hypothesiser

    # Test get slice
    hypothesiser_slice = hypothesiser[1:]
    assert isinstance(hypothesiser_slice, CompositeHypothesiser)
    assert hypothesiser_slice.sub_hypothesisers == sub_hypothesisers[1:]

    # Test iter
    for i, exp_sub_hypothesiser in enumerate(hypothesiser):
        assert exp_sub_hypothesiser == sub_hypothesisers[i]

    # Test len
    assert len(hypothesiser) == 4
