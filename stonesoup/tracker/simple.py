# -*- coding: utf-8 -*-
from .base import Tracker
from ..base import Property
from ..dataassociator import DataAssociator
from ..deletor import Deletor
from ..reader import DetectionReader
from ..initiator import Initiator
from ..updater import Updater


class SingleTargetTracker(Tracker):
    """A simple tracker.

    Track an object using StoneSoup components.
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
