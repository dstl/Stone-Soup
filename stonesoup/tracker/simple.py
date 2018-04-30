# -*- coding: utf-8 -*-
import copy

from .base import Tracker
from ..base import Property
from ..types import State, Track
from ..detector import Detector
from ..predictor import Predictor
from ..updater import Updater


class SingleTargetTracker(Tracker):
    """A simple tracker.

    Track an object using StoneSoup components.
    """
    detector = Property(
        Detector,
        doc="Detector used to generate detection objects.")
    predictor = Property(
        Predictor,
        doc="Predictor used to predict new state of the track.")
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

            if track is None:
                state = copy.deepcopy(self.initial_state)
                if state.timestamp is None:
                    state.timestamp = time
                track = Track()
                self.tracks.add(track)
            else:
                state = track.state

            if detections:
                # TODO: Data associator
                detection = detections.pop()
                predicted_state = self.predictor.predict(state,
                                                         detection.timestamp)
                track.states.append(self.updater.update(predicted_state,
                                                        detection))
            else:
                predicted_state = self.predictor.predict(state, time)
                track.states.append(predicted_state)

            yield {track}
