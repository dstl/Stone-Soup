import datetime
import numpy as np

from .base import Tracker
from ..base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.dataassociator.tracktotrack import TrackToTrackCounting
from stonesoup.reader.base import DetectionReader
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
from stonesoup.tracker.pointprocess import PointProcessMultiTargetTracker
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.feeder.track import Tracks2GaussianDetectionFeeder
from stonesoup.architecture.edge import FusionQueue

from stonesoup.models.transition.categorical import MarkovianTransitionModel
from stonesoup.types.state import CategoricalState
from stonesoup.predictor.categorical import HMMPredictor
from stonesoup.updater.categorical import HMMUpdater
from stonesoup.hypothesiser.categorical import HMMHypothesiser
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.initiator.categorical import SimpleCategoricalMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.tracker.tests.conftest import detector


class DummyDetector(DetectionReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current = kwargs['current']

    @BufferedGenerator.generator_method
    def detections_gen(self):
        yield self.current


class SimpleFusionTracker(Tracker):  # implement tracks method
    """Presumes data from this node are detections, and from every other node are tracks
    Acts as a wrapper around a base tracker. Track is fixed after the sliding window.
    It exists within it, but the States may change. """
    base_tracker: Tracker = Property(
        default=MultiTargetTracker(initiator=SimpleCategoricalMeasurementInitiator(
            prior_state=CategoricalState([1 / 2, 1 / 2],
                                         categories=['Friendly', 'Enemy']),
            updater=HMMUpdater()),
                                   deleter=UpdateTimeStepsDeleter(2),
                                   detector=detector,
                                   data_associator=GNNWith2DAssignment(
                                       HMMHypothesiser(
                                           predictor=HMMPredictor(MarkovianTransitionModel(
                                               transition_matrix=np.array([[0.6, 0.4],
                                                                           [0.39, 0.61]]))),
                                           updater=HMMUpdater())),
                                   updater=HMMUpdater()),
        doc="Tracker given to the fusion node")
    sliding_window: int = Property(default=30,
                              doc="The number of time steps before the result is fixed")
    queue: FusionQueue = Property(default=None, doc="Queue which feeds in data")
    current_time: datetime.datetime = Property(default=datetime.datetime(2000, 1, 1, 0, 0, 0, 1))
    detector: DetectionReader = Property(doc= "Detection reader to read detections from the queue")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()

    @property
    def tracks(self):
        return self._tracks

    def __iter__(self):
        self.detector_iter = iter(self.detector)
        return super().__iter__()

    def __next__(self):
        if self.current_time == datetime.datetime(2000, 1, 1, 0, 0, 0, 1):
            raise ValueError("current_time tracker property hasn't updated to match Architecture "
                             "current_time")
        # Get data in time window

        # I think this block of code is equivalent to a detection reader?
        data_in_window = list()
        for data_piece in self.queue:
            if (self.current_time - datetime.timedelta(seconds=self.sliding_window) <
                    data_piece.time_arrived <= self.current_time):
                data_in_window.append(data_piece)
        cdets = set()
        ctracks = set()
        for data_piece in data_in_window:
            if isinstance(data_piece, Detection):
                cdets.add(data_piece)
            elif isinstance(data_piece, Track):
                ctracks.add(data_piece)

        # run our tracker on our detections
        tracks = set()
        for time, ctracks in self.base_tracker:
            tracks.update(ctracks)

        # Take the tracks and combine them, accounting for the fact that they might not all be as
        # up-to-date as one another
        # for tracks in [ctracks]:
        #     dummy_detector = DummyDetector(current=[time, tracks])
        #     self.track_fusion_tracker.detector = Tracks2GaussianDetectionFeeder(dummy_detector)
        #     self.track_fusion_tracker.__iter__()
        #     _, tracks = next(self.track_fusion_tracker)
        #     self.track_fusion_tracker.update(tracks)

        return time, self.tracks


    # Don't edit anything that comes before the current time - the sliding window. That's fixed forever.
