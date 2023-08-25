from abc import ABC

from stonesoup.architecture.edge import FusionQueue
from .base import Tracker
from ..base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.base import DetectionReader
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track


class FusionTracker(Tracker, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()
        self._current_time = None

    def set_time(self, time):
        self._current_time = time


class DummyDetector(DetectionReader):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.current = kwargs['current']

    @BufferedGenerator.generator_method
    def detections_gen(self):
        yield self.current


class SimpleFusionTracker(FusionTracker):  # implement tracks method
    """Presumes data from this node are detections, and from every other node are tracks
    Acts as a wrapper around a base tracker. Track is fixed after the sliding window.
    It exists within it, but the States may change. """
    base_tracker: Tracker = Property(doc="Tracker given to the fusion node")
    sliding_window: int = Property(default=30,
                                   doc="The number of time steps before the result is fixed")
    queue: FusionQueue = Property(default=None,
                                  doc="Queue which feeds in data")
    track_fusion_tracker: Tracker = Property(doc="Tracker for fusing of multiple tracks together")

    @property
    def tracks(self):
        return self._tracks

    def __next__(self):
        data_piece = self.queue.get()
        if (self._current_time - data_piece.time_arrived).total_seconds() > self.sliding_window:
            # data not in window
            return
        if data_piece.time_arrived > self._current_time:
            raise ValueError("Not sure this should happen... check this")

        if isinstance(data_piece.data, Detection):
            return next(self.base_tracker)  # this won't work probably :(
            # Need to feed in self.queue as base_tracker.detector_iter
            # Also need to give the base tracker our tracks to treat as its own
        elif isinstance(data_piece.data, Track):
            pass
            # Must take account if one track has been fused together already

            # like this?
            # for tracks in [ctracks]:
            #     dummy_detector = DummyDetector(current=[time, tracks])
            #     self.track_fusion_tracker.detector =
            #     Tracks2GaussianDetectionFeeder(dummy_detector)
            #     self.track_fusion_tracker.__iter__()
            #     _, tracks = next(self.track_fusion_tracker)
            #     self.track_fusion_tracker.update(tracks)
            #
            # return time, self.tracks
        else:
            raise TypeError(f"Data piece contained an incompatible type: {type(data_piece.data)}")
