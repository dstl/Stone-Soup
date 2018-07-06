# -*- coding: utf-8 -*-
from collections.abc import MutableSequence

from ..base import Property
from .base import Type
from .state import State


class GroundTruthState(State):
    """Ground Truth State type"""


class GroundTruthPath(Type, MutableSequence):
    """Ground Truth Path type"""

    states = Property(
        [GroundTruthState],
        default=None,
        doc="List of groundtruth states to initialise path with. Default "
            "`None` which initialises with an empty list.")

    def __init__(self, states=None, *args, **kwargs):
        if states is None:
            states = []
        super().__init__(states, *args, **kwargs)

    def __len__(self):
        return len(self.states)

    def __setitem__(self, index, value):
        return self.states.__setitem__(index, value)

    def insert(self, index, value):
        return self.states.insert(index, value)

    def __delitem__(self, index):
        return self.states.__delitem__(index)

    def __getitem__(self, index):
        return self.states.__getitem__(index)
