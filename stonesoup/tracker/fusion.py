import datetime

from .base import Tracker
from ..base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.dataassociator.tracktotrack import TrackToTrackCounting
from stonesoup.reader.base import DetectionReader
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
from stonesoup.tracker.pointprocess import PointProcessMultiTargetTracker
from stonesoup.feeder.track import Tracks2GaussianDetectionFeeder
class DummyDetector(DetectionReader):
    def __init__(self, *args, **kwargs):
        self.current = kwargs['current']

    @BufferedGenerator.generator_method
    def detections_gen(self):
        yield self.current


class SimpleFusionTracker(Tracker):
    """Presumes data from this node are detections, and from every other node are tracks"""
    tracker: Tracker = Property(
        doc="Tracker given to the fusion node")
    track_fusion_tracker: Tracker = Property(
        doc="Tracker for associating tracks at the node")
    sliding_window = Property(
        doc="The number of time steps before the result is fixed")
    data = Property(
        doc="data received from queue and sensor")
    current_time: datetime.datetime = Property(
        doc='Current time in simulation')

    def __next__(self):
        # Get data in time window
        data_in_window = set()
        data_not_in_window = set()
        for data_piece in self.data:
            if (self.current_time - datetime.timedelta(seconds=self.sliding_window) <
                    data_piece.time_arrived <= self.current_time):
                data_in_window.add(data_piece)
            else:
                # Not sure what we want to do with this
                data_not_in_window.add(data_piece)

        cdets = set()
        ctracks = set()
        for data_piece in data_in_window:
            if isinstance(data_piece, Detection):
                cdets.add(data_piece)
            elif isinstance(data_piece, Track):
                ctracks.add(data_piece)

        # run our tracker on our detections
        tracks = set()
        for time, ctracks in self.tracker:
            tracks.update(ctracks)

        # Take the tracks and combine them, accounting for the fact that they might not all be as
        # up-to-date as one another
        for tracks in [ctracks]:
            dummy_detector = DummyDetector(current=[time, tracks])
            self.track_fusion_tracker.detector = Tracks2GaussianDetectionFeeder(dummy_detector)
            self.track_fusion_tracker.__iter__()
            _, tracks = next(self.track_fusion_tracker)
            self.track_fusion_tracks.update(tracks)


    # Don't edit anything that comes before the current time - the sliding window. That's fixed forever.