# -*- coding: utf-8 -*-
from ..base import Property
from .base import  StateVector


class Hypothesis(StateVector):
    """Hypothesis base type

    Parameters
    ==========
    gate : bool
        True if hypothesis
    """
    gate = Property(bool, default=False)
