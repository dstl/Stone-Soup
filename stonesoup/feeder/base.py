# -*- coding: utf-8 -*-
"""Base classes for Stone Soup feeder"""
from abc import abstractmethod

from ..base import Base


class Empty(Exception):
    """Exception raised by :class:`Feeder` when empty."""
    pass


class Feeder(Base):
    """Feeder base class

    Feeder consumes and outputs :class:`.Detection` data and can be used to
    modify the sequence, duplicate or drop data."""

    @abstractmethod
    def put(self, detection):
        """Put a new :class:`.Detection` in the Feeder.

        Parameters
        ----------
        detection : Detection
            A :class:`.Detection` to be added to the feeder.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self):
        """Get a :class:`.Detection` from the Feeder.

        Raises
        ------
        Empty
            If called when the Feeder is empty."""
        raise NotImplementedError
