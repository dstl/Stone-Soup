import datetime
import heapq

import pytest

from ..base import Tracker, _TrackerMixInUpdate, _TrackerMixInNext
from ...base import Property
from ...types.detection import Detection
from ...types.track import Track


class TrackerNextWithoutDetector(_TrackerMixInNext, Tracker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()

    def __next__(self) -> tuple[datetime.datetime, set[Track]]:
        time, detections = next(self.detector_iter)
        self._tracks = {Track(detection) for detection in detections}
        return time, self.tracks

    @property
    def tracks(self):
        return self._tracks


class TrackerNextWithDetector(TrackerNextWithoutDetector):
    detector: list = Property(default=[])


class TrackerUpdateWithoutDetector(_TrackerMixInUpdate, Tracker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()

    def update_tracker(self, time: datetime.datetime, detections: set[Detection]) \
            -> tuple[datetime.datetime, set[Track]]:

        self._tracks = {Track(detection) for detection in detections}
        return time, self.tracks

    @property
    def tracks(self):
        return self._tracks


class TrackerUpdateWithDetector(TrackerUpdateWithoutDetector):
    detector: list = Property(default=[])


@pytest.fixture
def detector() -> list[tuple[datetime.datetime, set[Detection]]]:
    detections = [
        Detection(timestamp=datetime.datetime(2023, 11, i), state_vector=[i])
        for i in range(1, 10)
    ]

    detector = [(det.timestamp, {det}) for det in detections]

    return detector


@pytest.mark.parametrize("tracker_class",
                         [TrackerNextWithoutDetector, TrackerNextWithDetector,
                          TrackerUpdateWithoutDetector, TrackerUpdateWithDetector])
def test_tracker_update_tracker(tracker_class, detector):
    tracker = tracker_class()
    for input_time, detections in detector:
        time, tracks = tracker.update_tracker(input_time, detections)

        assert time == input_time
        assert tracks == tracker.tracks
        tracks_state = {track.state for track in tracks}
        assert tracks_state == detections


@pytest.mark.parametrize("tracker_class",
                         [TrackerNextWithoutDetector, TrackerUpdateWithoutDetector])
def test_tracker_without_detector_iter_error(tracker_class):
    tracker_without_detector = tracker_class()
    with pytest.raises(AttributeError):
        iter(tracker_without_detector)

    with pytest.raises(TypeError):
        next(tracker_without_detector)


@pytest.mark.parametrize("tracker_class", [TrackerNextWithDetector, TrackerUpdateWithDetector])
def test_tracker_detector_none_iter_error(tracker_class):
    tracker = tracker_class(detector=None)
    with pytest.raises(AttributeError):
        iter(tracker)


@pytest.mark.parametrize("tracker_class", [TrackerNextWithDetector, TrackerUpdateWithDetector])
def test_tracker_with_detector_iter(tracker_class):
    tracker = tracker_class()
    assert iter(tracker) is tracker
    assert tracker.detector_iter is not None

    with pytest.raises(StopIteration):
        next(tracker)


@pytest.mark.parametrize("tracker_class", [TrackerNextWithDetector, TrackerUpdateWithDetector])
def test_tracker_with_detector_for_loop(tracker_class, detector):
    tracker = tracker_class(detector=detector)

    for (tracker_time, tracks), (detect_time, detections) in zip(tracker, detector):
        assert tracker_time == detect_time
        assert tracks == tracker.tracks
        tracks_state = {track.state for track in tracks}
        assert tracks_state == detections


@pytest.mark.parametrize("tracker_class", [TrackerNextWithDetector, TrackerUpdateWithDetector])
def test_tracker_with_detector_next(tracker_class, detector):
    tracker = tracker_class(detector=detector)
    assert iter(tracker) is tracker

    for detect_time, detections in detector:
        tracker_time, tracks = next(tracker)
        assert tracker_time == detect_time
        assert tracks == tracker.tracks
        tracks_state = {track.state for track in tracks}
        assert tracks_state == detections

    with pytest.raises(StopIteration):
        _ = next(tracker)


@pytest.mark.parametrize("tracker_class", [TrackerNextWithDetector, TrackerUpdateWithDetector])
def test_tracker_wont_restart(tracker_class, detector):
    tracker = tracker_class(detector=detector)
    for _ in tracker:
        pass

    iter(tracker)
    with pytest.raises(StopIteration):
        next(tracker)


@pytest.mark.parametrize("tracker_class", [TrackerNextWithDetector, TrackerUpdateWithDetector])
def test_heapq_merge_with_tracker(tracker_class, detector):
    merge_output = list(heapq.merge(tracker_class(detector=detector),
                                    tracker_class(detector=detector)))

    assert len(merge_output) == len(detector)*2

    for idx, (tracker_time, tracks) in enumerate(merge_output):
        detect_time, detections = detector[int(idx/2)]
        assert tracker_time == detect_time
        tracks_state = {track.state for track in tracks}
        assert tracks_state == detections


@pytest.mark.parametrize("tracker_class",
                         [TrackerNextWithoutDetector, TrackerNextWithDetector,
                          TrackerUpdateWithoutDetector, TrackerUpdateWithDetector])
def test_tracker_detector_iter_creation(tracker_class):
    tracker_without_detector = tracker_class()
    assert tracker_without_detector.detector_iter is None


@pytest.mark.parametrize("tracker_class", [TrackerNextWithDetector, TrackerUpdateWithDetector])
def test_tracker_with_detections_mid_iter(tracker_class, detector):
    tracker = tracker_class(detector=detector)
    for i, ((tracker_time, tracks), (detect_time, detections)) in enumerate(zip(tracker,
                                                                                detector)):
        assert tracker_time == detect_time
        assert tracks == tracker.tracks
        tracks_state = {track.state for track in tracks}
        assert tracks_state == detections

        interrupt_time = datetime.datetime(2024, 4, 1, i)
        interrupt_detections = {Detection(timestamp=interrupt_time, state_vector=[i])}
        time, interrupt_tracks = tracker.update_tracker(interrupt_time, interrupt_detections)
        assert time == interrupt_time
        assert interrupt_tracks == tracker.tracks
        interrupt_tracks_state = {track.state for track in interrupt_tracks}
        assert interrupt_tracks_state == interrupt_detections
