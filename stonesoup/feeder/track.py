from datetime import datetime
from typing import Tuple, Set, Iterator

import numpy as np

from .base import DetectionFeeder, TrackFeeder
from .multi import FeederToMultipleFeeders
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..models.measurement.linear import LinearGaussian
from ..types.detection import GaussianDetection
from ..types.track import Track


class Tracks2GaussianDetectionFeeder(DetectionFeeder):
    """
    Feeder consumes Track objects and outputs GaussianDetection objects.

    At each time step, the :attr:`Reader` feeds in a set of live tracks. The feeder takes the most
    recent state from each of those tracks, and turn them into a set of
    :class:`~.GaussianDetection` objects. Each detection is given a :class:`~.LinearGaussian`
    measurement model whose covariance is equal to the state covariance. The feeder assumes that
    the tracks are all live, that is each track has a state at the most recent time step.
    """
    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, tracks in self.reader:
            detections = []
            for track in tracks:
                dim = len(track.state.state_vector)
                detections.append(
                    GaussianDetection.from_state(
                        track.state,
                        measurement_model=LinearGaussian(
                            dim, list(range(dim)), np.asarray(track.covar)),
                        target_type=GaussianDetection)
                )

            yield time, detections


class SyncMultiTrackFeeder(TrackFeeder, FeederToMultipleFeeders):
    """
    Todo
    """

    reader: TrackFeeder = Property()

    def create_track_feeder(self) -> TrackFeeder:
        return self.create_feeder()

    def __iter__(self) -> Iterator[Tuple[datetime, Set[Track]]]:
        yield from self.create_feeder()
