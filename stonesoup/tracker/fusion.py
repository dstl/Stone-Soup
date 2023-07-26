import datetime

from .base import Tracker
from ..base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.dataassociator.tracktotrack import TrackToTrackCounting
from stonesoup.reader.base import DetectionReader
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import Hypothesis
from stonesoup.types.track import Track
from stonesoup.tracker.pointprocess import PointProcessMultiTargetTracker
from stonesoup.feeder.track import Tracks2GaussianDetectionFeeder


class DummyDetector(DetectionReader):
    def __init__(self, *args, **kwargs):
        self.current = kwargs['current']

    @BufferedGenerator.generator_method
    def detections_gen(self):
        yield self.current


class SimpleFusionTracker(Tracker):  # implement tracks method
    """Presumes data from this node are detections, and from every other node are tracks
    Acts as a wrapper around a base tracker. Track is fixed after the sliding window.
    It exists within it, but the States may change. """
    base_tracker: Tracker = Property(doc="Tracker given to the fusion node")
    sliding_window = Property(default=30,
                              doc="The number of time steps before the result is fixed")
    queue = Property(default=None, doc="Queue which feeds in data")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()
        self._current_time = None

    def set_time(self, time):
        self._current_time = time

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
            # Must take account if one track has been fused together already

            # like this?
            # for tracks in [ctracks]:
            #     dummy_detector = DummyDetector(current=[time, tracks])
            #     self.track_fusion_tracker.detector = Tracks2GaussianDetectionFeeder(dummy_detector)
            #     self.track_fusion_tracker.__iter__()
            #     _, tracks = next(self.track_fusion_tracker)
            #     self.track_fusion_tracker.update(tracks)
            #
            # return time, self.tracks
        elif isinstance(data_piece.data, Hypothesis):
            # do something
        else:
            raise TypeError(f"Data piece contained an incompatible type: {type(data_piece.data)}")


    # Don't edit anything that comes before the current time - the sliding window. That's fixed forever.
