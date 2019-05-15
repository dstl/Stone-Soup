import datetime
from operator import attrgetter

import numpy as np

from ..distance import DistanceHypothesiser
from ..filtered import FilteredDetectionsHypothesiser
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.track import Track
from ...types.update import GaussianStateUpdate
from ... import measures as measures

measure = measures.Mahalanobis()


def test_filtereddetections(predictor, updater):
    # CASE 1
    # there is one track with associated metadata,
    # two detections where one has matching metadata and one does not

    timestamp = datetime.datetime.now()

    hypothesiser = DistanceHypothesiser(predictor, updater,
                                        measure=measure, missed_distance=0.2)
    hypothesiser_wrapper = FilteredDetectionsHypothesiser(
        hypothesiser, "MMSI", match_missing=True)

    track = Track([GaussianStateUpdate(
                    np.array([[0]]),
                    np.array([[1]]),
                    SingleHypothesis(
                        None,
                        Detection(np.array([[0]]), metadata={"MMSI": 12345})),
                    timestamp=timestamp)])
    detection1 = Detection(np.array([[2]]), metadata={"MMSI": 12345})
    detection2 = Detection(np.array([[3]]), metadata={"MMSI": 99999})
    detections = {detection1, detection2}

    hypotheses = hypothesiser_wrapper.hypothesise(track, detections, timestamp)

    # There are 2 hypotheses - Detection 1,  Missed Dectection
    # - Detection 2 has different metadata, so no hypothesis
    assert len(hypotheses) == 2
    assert all(not hypothesis.measurement or
               hypothesis.measurement.metadata['MMSI'] == 12345
               for hypothesis in hypotheses)

    # There is a missed detection hypothesis
    assert any(not hypothesis.measurement for hypothesis in hypotheses)

    # Each hypothesis has a distance attribute
    assert all(hypothesis.distance >= 0 for hypothesis in hypotheses)

    # The hypotheses are sorted correctly
    assert min(hypotheses, key=attrgetter('distance')) is hypotheses[0]


def test_filtereddetections_empty_detections(predictor, updater):
    # CASE 3
    # 'detections' is empty
    timestamp = datetime.datetime.now()
    hypothesiser = DistanceHypothesiser(predictor, updater,
                                        measure=measure, missed_distance=0.2)
    hypothesiser_wrapper = FilteredDetectionsHypothesiser(
        hypothesiser, "MMSI", match_missing=False)

    track = Track([GaussianStateUpdate(
        np.array([[0]]),
        np.array([[1]]),
        SingleHypothesis(
            None,
            Detection(np.array([[0]]), metadata={"MMSI": 12345})),
        timestamp=timestamp)])
    detections = {}

    hypotheses = hypothesiser_wrapper.hypothesise(track, detections, timestamp)

    # one hypothesis - Missed Detection
    assert len(hypotheses) == 1

    # There is a missed detection hypothesis
    assert any(not hypothesis.measurement for hypothesis in hypotheses)


def test_filtereddetections_no_track_metadata(predictor, updater):
    # CASE 2
    # there is one track with NO associated metadata, two detections
    # with metadata, hypothesiser.match_missing is True (default), so
    # detections with any metadata can be associated with the track

    timestamp = datetime.datetime.now()
    hypothesiser = DistanceHypothesiser(predictor, updater,
                                        measure=measure, missed_distance=0.2)
    hypothesiser_wrapper = FilteredDetectionsHypothesiser(
        hypothesiser, "MMSI", match_missing=True)

    track = Track([GaussianStateUpdate(
        np.array([[0]]),
        np.array([[1]]),
        SingleHypothesis(
            None,
            Detection(np.array([[0]]), metadata={})),
        timestamp=timestamp)])
    detection1 = Detection(np.array([[2]]), metadata={"MMSI": 12345})
    detection2 = Detection(np.array([[3]]), metadata={"MMSI": 99999})
    detections = {detection1, detection2}

    hypotheses = hypothesiser_wrapper.hypothesise(track, detections, timestamp)

    # There are 3 hypotheses - Detection 1, Detection 2, Missed Detection
    assert len(hypotheses) == 3
    assert all(not hypothesis.measurement or
               hypothesis.measurement.metadata['MMSI'] == 12345 or
               hypothesis.measurement.metadata['MMSI'] == 99999
               for hypothesis in hypotheses)

    # There is a missed detection hypothesis
    assert any(not hypothesis.measurement for hypothesis in hypotheses)

    # Each hypothesis has a distance attribute
    assert all(hypothesis.distance >= 0 for hypothesis in hypotheses)

    # The hypotheses are sorted correctly
    assert min(hypotheses, key=attrgetter('distance')) is hypotheses[0]


def test_filtereddetections_no_matching_metadata(predictor, updater):
    # CASE 4
    # there is one track with associated metadata,
    # two detections where neither has matching metadata

    timestamp = datetime.datetime.now()
    hypothesiser = DistanceHypothesiser(predictor, updater,
                                        measure=measure, missed_distance=0.2)
    hypothesiser_wrapper = FilteredDetectionsHypothesiser(
        hypothesiser, "MMSI", match_missing=True)

    track = Track([GaussianStateUpdate(
                    np.array([[0]]),
                    np.array([[1]]),
                    SingleHypothesis(
                        None,
                        Detection(np.array([[0]]), metadata={"MMSI": 12345})),
                    timestamp=timestamp)])
    detection1 = Detection(np.array([[2]]), metadata={"MMSI": 45678})
    detection2 = Detection(np.array([[3]]), metadata={"MMSI": 99999})
    detections = {detection1, detection2}

    hypotheses = hypothesiser_wrapper.hypothesise(track, detections, timestamp)

    # one hypothesis - Missed Detection
    assert len(hypotheses) == 1

    # There is a missed detection hypothesis
    assert any(not hypothesis.measurement for hypothesis in hypotheses)
