from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Tuple, Set, Iterator

import numpy as np

from .modify import DelayedFeeder, CopyFeeder
from .base import DetectionFeeder, TrackFeeder
from .multi import FeederToMultipleFeeders
from .track_continuity_buffer import BetaTrackContinuityBuffer, TrackerWithContinuityBuffer
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..models.measurement.linear import LinearGaussian
from ..predictor.base import Predictor
from ..tracker import Tracker
from ..types.detection import GaussianDetection
from ..types.track import Track


class Tracks2GaussianDetectionFeeder(DetectionFeeder):
    '''
    Feeder consumes Track objects and outputs GaussianDetection objects.

    At each time step, the :attr:`Reader` feeds in a set of live tracks. The feeder takes the most
    recent state from each of those tracks, and turn them into a set of
    :class:`~.GaussianDetection` objects. Each detection is given a :class:`~.LinearGaussian`
    measurement model whose covariance is equal to the state covariance. The feeder assumes that
    the tracks are all live, that is each track has a state at the most recent time step.
    '''
    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, tracks in self.reader:
            detections = set()
            for track in tracks:
                dim = len(track.state.state_vector)
                metadata = track.metadata.copy()
                metadata['track_id'] = track.id
                detections.add(
                    GaussianDetection.from_state(
                        track.state,
                        measurement_model=LinearGaussian(
                            dim, list(range(dim)), np.asarray(track.covar)),
                        metadata=metadata,
                        target_type=GaussianDetection)
                )

            yield time, detections


class TrackFeederIter(TrackFeeder):
    def __iter__(self) -> Iterator[Tuple[datetime, Set[Track]]]:
        return self

    @abstractmethod
    def __next__(self) -> Tuple[datetime, Set[Track]]:
        raise NotImplementedError


class ActiveTrackerTrackFeeder(TrackFeeder):
    tracker: Tracker = Property()

    def __iter__(self) -> Iterator[Tuple[datetime, Set[Track]]]:
        for time, tracks in self.tracker:
            yield time, tracks
        return


class PassiveTrackerTrackFeeder(TrackFeederIter):
    """
    This track feeder takes the :attr:`~Tracker.tracks` property from a tracks to feed tracks.

    Notes
    -----
    The time output of next is likely to be wrong when the tracker does not contain tracks. This
    class does not work well the tracker isn't producing tracks as the time is drawn from the
    tracks.
    """

    tracker: Tracker = Property()
    time: datetime = Property(default=datetime.min)

    def __next__(self) -> Tuple[datetime, Set[Track]]:
        tracks = self.tracker.tracks
        if len(tracks) > 0:
            self.time = max(track.state.timestamp for track in tracks)
        else:
            # Have to use previous/preset time
            pass

        return self.time, tracks


class SyncMultiTrackFeeder(TrackFeeder, FeederToMultipleFeeders):
    """
    Todo
    """

    reader: TrackFeeder = Property()

    def create_track_feeder(self) -> TrackFeeder:
        return self.create_feeder()

    def __iter__(self) -> Iterator[Tuple[datetime, Set[Track]]]:
        yield from self.create_feeder()


class AsyncMultiTrackFeeder(TrackFeeder):
    """
    Track objects are often edited (appended with more states). If the track feeders are ran
    asynchronously this can lead to track objects being fed with additional states that they
    shouldn't have at that time

    This feeder breaks up the track objects with a CopyFeeder and reassembles them with a
    ContinuityBuffer. This breaking and reassembling ensure the tracks that are fed are aligned
    with the correct time

    To enable async ability. The track objects will different track objects but will have the same
    information
    """

    reader: TrackFeeder = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_feeder = FeederToMultipleFeeders(CopyFeeder(self.reader))

    def __iter__(self) -> Iterator[Tuple[datetime, Set[Track]]]:
        return iter(self.create_track_feeder())

    def create_track_feeder(self) -> TrackFeeder:
        return TrackerWithContinuityBuffer(self.multi_feeder.create_feeder(),
                                           BetaTrackContinuityBuffer())


class DelayedTrackFeeder(TrackFeeder):
    """
    This track feeder takes input of (`time', Set[Track]) and appends a prediction to any tracks
    that don't have a state equal to `time'
    """
    reader: TrackFeeder = Property(doc="Object to recieve the ")
    predictor: Predictor = Property()
    delay: timedelta = Property(default=timedelta(seconds=0))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delayed_feeder = DelayedFeeder(reader=CopyFeeder(self.reader),
                                            delay=self.delay)

    def __iter__(self) -> Iterator[Tuple[datetime, Set[Track]]]:
        for time, tracks in self.delayed_feeder:
            predicted_tracks = self.predict_tracks(tracks, time)
            yield time, predicted_tracks
        return

    def predict_tracks(self, tracks: Set[Track], timestamp):
        for track in tracks:
            prior = track.state
            if prior.timestamp < timestamp:
                new_state = self.predictor.predict(prior, timestamp=timestamp)
                track.append(new_state)
        return tracks
