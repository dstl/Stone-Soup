# -*- coding: utf-8 -*-
from ..base import Property
from .state import State, StateMutableSequence


class GroundTruthState(State):
    """Ground Truth State type"""


class GroundTruthPath(StateMutableSequence):
    """Ground Truth Path type

    A :class:`~.StateMutableSequence` representing a track.
    """

    states = Property(
        [GroundTruthState],
        default=None,
        doc="List of groundtruth states to initialise path with. Default "
            "`None` which initialises with an empty list.")
