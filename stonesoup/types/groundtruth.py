# -*- coding: utf-8 -*-
import uuid
from typing import MutableSequence, MutableMapping

from ..base import Property
from .state import State, StateMutableSequence


class GroundTruthState(State):
    """Ground Truth State type"""
    metadata: MutableMapping = Property(
        default=None, doc='Dictionary of metadata items for Detections.')

    def __init__(self, state_vector, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)
        if self.metadata is None:
            self.metadata = {}


class GroundTruthPath(StateMutableSequence):
    """Ground Truth Path type

    A :class:`~.StateMutableSequence` representing a track.
    """

    states: MutableSequence[GroundTruthState] = Property(
        default=None,
        doc="List of groundtruth states to initialise path with. Default "
            "`None` which initialises with an empty list.")
    id: str = Property(
        default=None,
        doc="The unique path ID. Default `None` where random UUID is "
            "generated.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())
