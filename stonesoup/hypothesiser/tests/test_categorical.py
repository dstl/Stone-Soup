from datetime import datetime

from ..categorical import HMMHypothesiser
from ...types.array import StateVector
from ...types.detection import CategoricalDetection
from ...types.multihypothesis import MultipleHypothesis
from ...types.state import CategoricalState
from ...types.track import Track


def test_hmm_hypothesiser(dummy_category_predictor, dummy_category_updater):
    now = datetime.now()

    track = Track([CategoricalState(state_vector=[80, 10, 10], timestamp=now)])

    measurement_categories = ['red', 'green', 'blue', 'yellow']
    detection1 = CategoricalDetection(StateVector([10, 20, 30, 40]),
                                      timestamp=now,
                                      categories=measurement_categories)
    detection2 = CategoricalDetection(StateVector([40, 30, 20, 10]),
                                      timestamp=now,
                                      categories=measurement_categories)
    detection3 = CategoricalDetection(StateVector([10, 40, 40, 10]),
                                      timestamp=now,
                                      categories=measurement_categories)
    detections = {detection1, detection2, detection3}

    hypothesiser = HMMHypothesiser(dummy_category_predictor, dummy_category_updater)

    hypotheses = hypothesiser.hypothesise(track, detections, now)

    assert isinstance(hypotheses, MultipleHypothesis)

    # 4 hypotheses: detections 1, 2 and 3, and missed detection
    assert len(hypotheses) == 4

    for hypothesis in hypotheses:
        if hypothesis.measurement:
            assert hypothesis
            assert hypothesis.measurement in detections
        else:
            assert not hypothesis
        assert hypothesis.probability >= 0
