import datetime
from itertools import product

import numpy as np
import pytest

from ...gater.distance import DistanceGater
from ...hypothesiser.mfa import MFAHypothesiser
from ...measures import Mahalanobis
from ...types.detection import Detection
from ...types.mixture import GaussianMixture
from ...types.numeric import Probability
from ...types.state import TaggedWeightedGaussianState, GaussianState
from ...types.track import Track
from ...types.update import GaussianMixtureUpdate
try:
    from ..mfa import MFADataAssociator
except ImportError:
    pytest.skip("ortools not available", allow_module_level=True)


def update_tracks(associations, updater):
    for track, hypotheses in associations.items():
        components = []
        for hypothesis in hypotheses:
            if not hypothesis:
                components.append(hypothesis.prediction)
            else:
                update = updater.update(hypothesis)
                components.append(update)
        track.append(GaussianMixtureUpdate(components=components, hypothesis=hypotheses))


def generate_detections(tracks, timestamp, predictor, measurement_model, n=2):
    return {
        Detection(
            measurement_model.function(predictor.predict(
                GaussianState(
                    track.mean,
                    track.covar,
                    track.timestamp), timestamp), noise=True),
            timestamp=timestamp)
        for track in tracks for _ in range(n)}  # n detections per track; pseudo clutter


@pytest.fixture(scope='function', params=[2, 3, 6])
def data_associator(request, probability_hypothesiser):
    # Hypothesiser and Data Associator
    hypothesiser = MFAHypothesiser(probability_hypothesiser)

    return MFADataAssociator(hypothesiser, slide_window=request.param)


def test_mfa(predictor, updater, measurement_model, data_associator):
    start_time = datetime.datetime.now()
    slide_window = data_associator.slide_window

    prior1 = GaussianMixture([TaggedWeightedGaussianState([[0], [1], [0], [1]],
                                                          np.diag([1.5, 0.5, 1.5, 0.5]),
                                                          timestamp=start_time,
                                                          weight=Probability(1), tag=[])])
    prior2 = GaussianMixture([TaggedWeightedGaussianState([[0], [1], [40], [-1]],
                                                          np.diag([1.5, 0.5, 1.5, 0.5]),
                                                          timestamp=start_time,
                                                          weight=Probability(1), tag=[])])
    tracks = {Track([prior1]), Track([prior2])}

    timestamp = start_time + datetime.timedelta(seconds=1)
    detections = generate_detections(tracks, timestamp, predictor, measurement_model)

    associations = data_associator.associate(tracks, detections, timestamp)

    # Association for each track
    assert tracks == associations.keys()

    # All measurements should feature
    assert detections == {hyp.measurement for mhyp in associations.values() for hyp in mhyp if hyp}

    # Missed detections
    assert len([hyp for mhyp in associations.values() for hyp in mhyp if not hyp]) == 2

    for track, mhyp in associations.items():
        assert len(mhyp) == 5

    # Let's add a gate
    data_associator.hypothesiser = DistanceGater(data_associator.hypothesiser, Mahalanobis(), 6)
    associations = data_associator.associate(tracks, detections, timestamp)

    assert tracks == associations.keys()
    assert detections == {hyp.measurement for mhyp in associations.values() for hyp in mhyp if hyp}
    assert len([hyp for mhyp in associations.values() for hyp in mhyp if not hyp]) == 2

    for track, mhyp in associations.items():
        assert len(mhyp) == 3  # One missed, and detections near track
        for hyp in mhyp:
            if not hyp:
                assert hyp.prediction.tag == [0]
            else:
                assert hyp.prediction.tag in [[n] for n in range(1, len(detections) + 1)]

    step = 2
    timestamp += datetime.timedelta(seconds=1)
    update_tracks(associations, updater)
    detections = generate_detections(tracks, timestamp, predictor, measurement_model)
    associations = data_associator.associate(tracks, detections, timestamp)

    for track, mhyp in associations.items():
        assert len(mhyp) == 3**step if slide_window > step else 3  # Pruned
        for hyp in mhyp:
            assert len(hyp.prediction.tag) == step
            if not hyp:
                assert hyp.prediction.tag[-1] == 0
                assert hyp.prediction.tag[-2] in range(5)
            else:
                assert tuple(hyp.prediction.tag) in list(product(range(0, 5),
                                                                 range(1, 5)))

    step = 3
    timestamp += datetime.timedelta(seconds=1)
    update_tracks(associations, updater)
    detections = generate_detections(tracks, timestamp, predictor, measurement_model)
    associations = data_associator.associate(tracks, detections, timestamp)

    for track, mhyp in associations.items():
        assert len(mhyp) == 3**step if slide_window > step else 3**(slide_window-1)
        for hyp in mhyp:
            assert len(hyp.prediction.tag) == step
            if not hyp:
                assert hyp.prediction.tag[-1] == 0
                assert hyp.prediction.tag[-2] in range(5)
                assert hyp.prediction.tag[-3] in range(5)
            else:
                assert tuple(hyp.prediction.tag) in list(product(range(0, 5),
                                                                 range(0, 5),
                                                                 range(1, 5)))


def test_mfa_no_tracks(data_associator):
    associations = data_associator.associate(set(), set(), datetime.datetime.now())
    assert not associations

    associations = data_associator.associate(set(), {Detection([0, 1])}, datetime.datetime.now())
    assert not associations
