# -*- coding: utf-8 -*-
import copy

from .base import Tracker
from ..base import Property
from ..types import State, Track
from ..dataassociator import DataAssociator
from ..reader import DetectionReader
from ..updater import Updater


class SingleTargetTracker(Tracker):
    """A simple tracker.

    Track an object using StoneSoup components.
    """
    detector = Property(
        DetectionReader,
        doc="Detector used to generate detection objects.")
    data_associator = Property(
        DataAssociator,
        doc="Association algorithm to pair predictions to detections")
    updater = Property(
        Updater,
        doc="Updater used to update the track object to the new state.")
    initial_state = Property(
        State,
        doc="First state")

    def __init__(self, *args, **kwargs):
        self.tracks = set()
        super().__init__(*args, **kwargs)

    def tracks_gen(self):
        track = None

        for time, detections in self.detector.detections_gen():

            # TODO: Initialiser
            if track is None:
                state = copy.deepcopy(self.initial_state)
                if state.timestamp is None:
                    state.timestamp = time
                track = Track()
                track.states.append(state)
                self.tracks.add(track)

            associations = self.data_associator.associate({track}, detections,
                                                          time)

            if associations[track].detection is not None:
                state_post, _ = self.updater.update(
                    associations[track].prediction,
                    associations[track].innovation,
                    associations[track].detection)
                track.states.append(state_post)
            else:
                track.states.append(associations[track].prediction)

            yield {track}
