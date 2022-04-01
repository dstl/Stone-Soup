# -*- coding: utf-8 -*-

from .base import Initiator
from ..base import Property
from ..types.hypothesis import SingleHypothesis
from ..types.state import CategoricalState
from ..types.track import Track
from ..updater.categorical import HMMUpdater


class SimpleCategoricalMeasurementInitiator(Initiator):
    """Initiator that creates tracks in a categorical state space.

    Uses state updates from an :class:`HMMUpdater` to initialise new tracks.
    Initialises a new track on every detection received.
    """

    prior_state: CategoricalState = Property(doc="Prior state information")
    updater: HMMUpdater = Property(doc="Hidden Markov model updater")

    def initiate(self, detections, *args, **kwargs):
        """Create a new track for each detection. Updating the :attr:`prior-state` with a
        detection to start-off a new track.
        """

        tracks = set()

        for detection in detections:
            hypothesis = SingleHypothesis(prediction=self.prior_state, measurement=detection)

            update = self.updater.update(hypothesis)

            tracks.add(Track([update]))
        return tracks
