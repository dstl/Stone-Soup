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
        self.tracks = set()
        super().__init__(*args, **kwargs)

    def tracks_gen(self):
        track = None

        for time, detections in self.detector.detections_gen():

            if track is not None:
                associations = self.data_associator.associate(
                        {track}, detections, time)
                if associations[track].detection is not None:
                    state_post = self.updater.update(
                        associations[track].prediction,
                        associations[track].detection,
                        associations[track].innovation)
                    track.states.append(state_post)
                else:
                    track.states.append(associations[track].prediction)

            if track is None or self.deletor.delete_tracks({track}):
                new_tracks = self.initiator.initiate(detections)
                if new_tracks:
                    track = next(iter(new_tracks))
                    self.tracks.add(track)
                else:
                    track = None

            yield time, {track}
