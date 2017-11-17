# -*- coding: utf-8 -*-
"""A collection of simple :class:`Feeder` classes.
"""
from collections import deque

from .base import Feeder


class FIFOFeeder(deque, Feeder):
    """First In, First Out Feeder

    The most basic :class:`Feeder` which simply passes data as received.
    Inherits from :class:`collections.deque`"""
