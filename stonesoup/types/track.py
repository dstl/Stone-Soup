# -*- coding: utf-8 -*-
from ..base import Property
from ..types import State, StateMutableSequence


class Track(StateMutableSequence):
    """Track type

    A :class:`~.StateMutableSequence` representing a track.
    """

    states = Property(
        [State],
        default=None,
        doc="The initial states of the track. Default `None` which initialises"
            "with empty list.")
