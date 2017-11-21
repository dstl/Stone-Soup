# -*- coding: utf-8 -*-
"""A collection of simple :class:`Feeder` classes.
"""
from collections import deque

from .base import Empty, Feeder


class FIFOFeeder(Feeder):
    """First In, First Out Feeder

    The most basic :class:`Feeder` which simply passes data as received.
    Utilises :class:`collections.deque` internally."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue = deque()

    def put(self, detection):
        """Append detection on internal :class:`collections.deque`.

        Parameters
        ----------
        detection : Detection
            Detection to be added to internal :class:`collections.deque`."""
        self._queue.append(detection)

    def get(self):
        """Pop (left) detection from internal :class:`collections.deque`.

        Returns
        -------
        Detection
            Detection which was first to be added to the Feeder.

        Raises
        ------
        Empty
            If called when empty.
        """
        try:
            return self._queue.popleft()
        except IndexError as err:
            raise Empty(err)
