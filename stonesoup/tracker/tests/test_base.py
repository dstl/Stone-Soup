import datetime
import heapq
from typing import Tuple, Set, List

import pytest

from ..base import Tracker
from ...base import Property
from ...types.detection import Detection
from ...types.track import Track


class TrackerWithoutDetector(Tracker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()

    def update_tracker(self, time: datetime.datetime, detections: Set[Detection]) \
            -> Tuple[datetime.datetime, Set[Track]]:

        self._tracks = {Track(detection) for detection in detections}
        return time, self.tracks

    @property
    def tracks(self):
        return self._tracks


class TrackerWithDetector(TrackerWithoutDetector):
    detector: list = Property(default=[])


@pytest.fixture
def detector() -> List[Tuple[datetime.datetime, Set[Detection]]]:
    detections = [Detection(timestamp=datetime.datetime(2023, 11, i),
                            state_vector=[i])
                  for i in range(1, 10)
                  ]

    detector = [(det.timestamp, {det}) for det in detections]

    return detector


@pytest.mark.parametrize("tracker_class", [TrackerWithoutDetector, TrackerWithDetector])
def test_tracker_update_tracker(tracker_class, detector):
    tracker_without_detector = tracker_class()
    for input_time, detections in detector:
        time, tracks = tracker_without_detector.update_tracker(input_time, detections)

        assert time == input_time
        assert tracks == tracker_without_detector.tracks
        tracks_state = {track.state for track in tracks}
        assert tracks_state == detections


def test_tracker_without_detector_iter_error():
    tracker_without_detector = TrackerWithoutDetector()
    with pytest.raises(AttributeError):
        iter(tracker_without_detector)

    with pytest.raises(TypeError):
        next(tracker_without_detector)


def test_tracker_with_detector_iter():
    tracker = TrackerWithDetector()
    assert iter(tracker) is tracker
    assert tracker.detector_iter is not None

    with pytest.raises(StopIteration):
        next(tracker)


def test_tracker_with_detector_for_loop(detector):
    tracker = TrackerWithDetector(detector=detector)

    for (tracker_time, tracks), (detect_time, detections) in zip(tracker, detector):
        assert tracker_time == detect_time
        assert tracks == tracker.tracks
        tracks_state = {track.state for track in tracks}
        assert tracks_state == detections


def test_tracker_with_detector_next(detector):
    tracker = TrackerWithDetector(detector=detector)
    assert iter(tracker) is tracker

    for detect_time, detections in detector:
        tracker_time, tracks = next(tracker)
        assert tracker_time == detect_time
        assert tracks == tracker.tracks
        tracks_state = {track.state for track in tracks}
        assert tracks_state == detections

    with pytest.raises(StopIteration):
        _ = next(tracker)


def test_tracker_wont_restart(detector):
    tracker = TrackerWithDetector(detector=detector)
    for _ in tracker:
        pass

    iter(tracker)
    with pytest.raises(StopIteration):
        next(tracker)


def test_heapq_merge_with_tracker(detector):
    merge_output = list(heapq.merge(TrackerWithDetector(detector=detector),
                                    TrackerWithDetector(detector=detector)))

    assert len(merge_output) == len(detector)*2

    for idx, (tracker_time, tracks) in enumerate(merge_output):
        detect_time, detections = detector[int(idx/2)]
        assert tracker_time == detect_time
        tracks_state = {track.state for track in tracks}
        assert tracks_state == detections


# The next two tests are 'Tracker' implementation specific.
@pytest.mark.parametrize("tracker_class, expected",
                         [(TrackerWithoutDetector, None),
                          (TrackerWithDetector, [])
                          ])
def test_tracker_detector_creation(tracker_class, expected):
    tracker_without_detector = tracker_class()
    assert tracker_without_detector.detector == expected


@pytest.mark.parametrize("tracker_class", [TrackerWithoutDetector, TrackerWithDetector])
def test_tracker_detector_iter_creation(tracker_class):
    tracker_without_detector = tracker_class()
    assert tracker_without_detector.detector_iter is None
