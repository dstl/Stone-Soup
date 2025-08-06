from collections.abc import Collection
import datetime
import numpy as np

from . import DetectionFeeder, TrackFeeder
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..models.measurement.linear import LinearGaussian
from ..types.detection import GaussianDetection, Detection
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
            detections = set()
            for track in tracks:
                if isinstance(track, Track):
                    dim = len(track.state.state_vector)
                    metadata = track.metadata.copy()
                    metadata['track_id'] = track.id
                    detections.add(
                        GaussianDetection.from_state(
                            track.state,
                            state_vector=track.mean,
                            covar=track.covar,
                            measurement_model=LinearGaussian(
                                dim, list(range(dim)), np.asarray(track.covar)),
                            metadata=metadata,
                            target_type=GaussianDetection)
                    )
                elif isinstance(track, Detection):
                    detections.add(track)
                else:
                    raise TypeError(f"track is of type {type(track)}. Expected Track or Detection")

            yield time, detections


class ReplayTrackFeeder(TrackFeeder):
    """
    Feeder outputs Track objects from an input of tracks.
    This allows an already produced set of tracks to be used as a reader.

    At each timestep, the states of each track that existed at that point are output.
    Any tracks which have ended are removed
    """

    reader: Collection[Track] = Property(doc="A collection of tracks to be replayed")
    times: list[datetime.datetime] = Property(
        default=None, doc="The timesteps at which the tracks should be replayed. "
        "The default `None` will use all timestamps")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.times is None:
            times = {state.timestamp for track in self.reader for state in track}
            self.times = sorted(times)

    @BufferedGenerator.generator_method
    def data_gen(self):
        output_tracks = {}
        last_time = None

        for time in self.times:
            for track in self.reader:
                if not track or track[0].timestamp > time:
                    continue
                if last_time is not None and last_time >= track.timestamp:
                    if track in output_tracks:
                        del output_tracks[track]
                    continue

                track_states = [state for state in track if state.timestamp <= time]
                current_track = output_tracks.get(track, Track())
                current_track.states = track_states
                output_tracks[track] = current_track
            last_time = time
            yield time, set(output_tracks.values())
