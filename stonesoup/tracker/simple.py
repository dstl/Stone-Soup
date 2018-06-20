# -*- coding: utf-8 -*-
from .base import Tracker
from ..base import Property
from ..dataassociator import DataAssociator
from ..deletor import Deletor
from ..reader import DetectionReader
from ..initiator import Initiator
from ..updater import Updater


class SingleTargetTracker(Tracker):
    """A simple single target tracker.

    Track a single object using Stone Soup components. The tracker works by
    first calling the :attr:`data_associator` with the active track, and then
    either updating the track state with the result of the :attr:`updater` if
    a detection is associated, or with the prediction if no detection is
    associated to the track. The track is then checked for deletion by the
    :attr:`deletor`, and if deleted the :attr:`initiator` is called to generate
    a new track. Similarly if no track is present (i.e. tracker is initialised
    or deleted in previous iteration), only the :attr:`initiator` is called.

    Parameters
    ----------

    Attributes
    ----------
    track : :class:`~.Track`
        Current track being maintained. Also accessible as the sole item in
        :attr:`tracks`
    """
    initiator = Property(
        Initiator,
        doc="Initiator used to initialise the track.")
    deletor = Property(
        Deletor,
        doc="Deletor used to delete the track")
    detector = Property(
        DetectionReader,
        doc="Detector used to generate detection objects.")
    data_associator = Property(
        DataAssociator,
        doc="Association algorithm to pair predictions to detections")
    updater = Property(
        Updater,
        doc="Updater used to update the track object to the new state.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track = None

    @property
    def tracks(self):
        return {self.track}

    def tracks_gen(self):
        self.track = None

        for time, detections in self.detector.detections_gen():

            if self.track is not None:
                associations = self.data_associator.associate(
                        self.tracks, detections, time)
                if associations[self.track].detection is not None:
                    state_post = self.updater.update(
                        associations[self.track].prediction,
                        associations[self.track].detection,
                        associations[self.track].innovation)
                    self.track.states.append(state_post)
                else:
                    self.track.states.append(
                        associations[self.track].prediction)

            if self.track is None or self.deletor.delete_tracks(self.tracks):
                new_tracks = self.initiator.initiate(detections)
                if new_tracks:
                    track = next(iter(new_tracks))
                    self.track = track
                else:
                    self.track = None

            yield time, self.tracks


class MultiTargetTracker(Tracker):
    """A simple multi target tracker.

    Track multiple objects using Stone Soup components. The tracker works by
    first calling the :attr:`data_associator` with the active tracks, and then
    either updating the track state with the result of the :attr:`updater` if
    a detection is associated, or with the prediction if no detection is
    associated to the track. Tracks are then checked for deletion by the
    :attr:`deletor`, and remaining unassociated detections are passed to the
    :attr:`initiator` to generate new tracks.

    Parameters
    ----------
    """
    initiator = Property(
        Initiator,
        doc="Initiator used to initialise the track.")
    deletor = Property(
        Deletor,
        doc="Initiator used to initialise the track.")
    detector = Property(
        DetectionReader,
        doc="Detector used to generate detection objects.")
    data_associator = Property(
        DataAssociator,
        doc="Association algorithm to pair predictions to detections")
    updater = Property(
        Updater,
        doc="Updater used to update the track object to the new state.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()

    @property
    def tracks(self):
        return self._tracks.copy()

    def tracks_gen(self):
        self._tracks = set()

        for time, detections in self.detector.detections_gen():

            associations = self.data_associator.associate(
                self._tracks, detections, time)
            associated_detections = set()
            for track, hypothesis in associations.items():
                if hypothesis.detection is not None:
                    state_post = self.updater.update(
                        hypothesis.prediction,
                        hypothesis.detection,
                        hypothesis.innovation)
                    track.states.append(state_post)
                    associated_detections.add(hypothesis.detection)
                else:
                    track.states.append(hypothesis.prediction)

            self._tracks -= self.deletor.delete_tracks(self._tracks)
            self._tracks |= self.initiator.initiate(
                detections - associated_detections)

            yield time, self.tracks
