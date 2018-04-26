# -*- coding: utf-8 -*-

from .base import Tracker
from ..base import Property
from ..types import State, Track
from ..detector import Detector
from ..predictor import Predictor
from ..updater import Updater


class SingleTragetTracker(Tracker):
    """A simple tracker.

    Track an object using StoneSoup components.
    """
    detector = Property(
        Detector, doc="Detector used to generate detection objects.")
    predictor = Property(
        Predictor, doc="Predictor used to predict new state of the track.")
    updater = Property(
        Updater, doc="Updater used to update the track object to the new state.")
    initial_state = Property(
        State, doc="First state")

    def get_tracks(self):
        track = None

        for detections in self.detector.get_detections():

            if track is None:
                state = self.initial_state
                track = Track()
            else:
                state = track.state

            predicted_state = self.predictor.predict(state)
            detection = detections.pop()
            track.states.append(self.updater.update(predicted_state, detection))

            yield set([track])
