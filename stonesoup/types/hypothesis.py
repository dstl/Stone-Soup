# -*- coding: utf-8 -*-
from ..base import Property
from .state import State


class Hypothesis(State):
    """Hypothesis base type

    Parameters
    ==========
    gate : bool
        True if hypothesis
    """
    gate = Property(bool, default=False)
