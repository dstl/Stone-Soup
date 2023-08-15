from abc import abstractmethod, ABC
from datetime import datetime
from typing import Set, Dict, Tuple, Iterator

from .base import TrackFeeder
from ..base import Base, Property
from ..tracker.base import Tracker
from ..types.track import Track, CompositeTrack
from ..writer.base import TrackWriter


class TrackContinuityBuffer(Base):
    """Track Continuity Buffer Base Class

    Subclasses must consumes and output a set :class:`.Track` via the `update_tracks` method
    """

    @abstractmethod
    def update_tracks(self, input_tracks: Set[Track]) -> Set[Track]:
        raise NotImplementedError

    @property
    def tracks(self) -> Set[Track]:
        raise NotImplementedError


class TrackerWithContinuityBuffer(TrackWriter, TrackFeeder):
    """
    This class is a wrapper around a :class:`.Tracker` to filter the output of the `Tracker`
    """
    tracker: Tracker = Property(doc="Tracker to supply tracks")
    continuity_buffer: TrackContinuityBuffer = Property(
        doc="Continuity buffer to filter the tracks"
    )

    @property
    def tracks(self) -> Set[Track]:
        return self.continuity_buffer.tracks

    def __iter__(self) -> Iterator[Tuple[datetime, Set[Track]]]:
        for time, tracks in self.tracker:
            yield time, self.continuity_buffer.update_tracks(tracks)
        return


class AlphaTrackContinuityBuffer(TrackContinuityBuffer, ABC):
    """This is an abstract TrackContinuityBuffer class. It contains a dictionary to store track
    and `tracks` to retrieve tracks from"""

    track_library: Dict[str, Track] = Property(
        default=None,
        doc="This dictionary is store tracks with str keys. This property should not be "
            "supplied as it will be overwritten on initialisation")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_library = dict()

    @property
    def tracks(self):
        return set(self.track_library.values())


class BetaTrackContinuityBuffer(AlphaTrackContinuityBuffer):
    """This Track Continuity Buffer will combine tracks if they have the same
    track id for subsequent time-steps. Tracks that aren't updated are deleted"""

    def update_tracks(self, input_tracks: Set[Track]) -> Set[Track]:

        new_track_library = dict()

        for input_track in input_tracks:
            output_track = self.track_library.get(input_track.id, None)
            if output_track is None:
                output_track = CompositeTrack(id=input_track.id)

            output_track.append(input_track.state)
            output_track.metadata.update(input_track.metadata)
            output_track.sub_tracks.add(input_track)

            new_track_library[input_track.id] = output_track

        self.track_library = new_track_library
        return self.tracks
