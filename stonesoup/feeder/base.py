# -*- coding: utf-8 -*-
"""Base classes for Stone Soup feeder"""
from abc import abstractmethod

from ..base import BaseMeta


class Feeder(metaclass=BaseMeta):
    """Feeder base class

    Feeder consumes and outputs :class:`.Detection` data and can be used to
    modify the sequence, duplicate or drop data."""

    @abstractmethod
    def append(self, detection):
        """Append a new :class:`.Detection` to the Feeder."""
        raise NotImplementedError

    @abstractmethod
    def popleft(self):
        """Pop a :class:`.Detection` from the Feeder.

        Raises
        ------
        IndexError
            If called when the Feeder is empty."""
        raise NotImplementedError
