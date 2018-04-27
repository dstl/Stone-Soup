# -*- coding: utf-8 -*-
from collections.abc import MutableSequence

from .base import Type
from .state import State


class GroundTruthState(State):
    """Ground Truth State type"""


class GroundTruthPath(Type, MutableSequence):
    """Ground Truth Path type"""

    def __init__(self, *args, **kwargs):
        self._states = list(*args, **kwargs)

    def __len__(self):
        return len(self._states)

    def __setitem__(self, index, value):
        return self._states.__setitem__(index, value)

    def insert(self, index, value):
        return self._states.insert(index, value)

    def __delitem__(self, index):
        return self._states.__delitem__(index)

    def __getitem__(self, index):
        return self._states.__getitem__(index)
