# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import BaseMeta


class Feeder(metaclass=BaseMeta):
    """Feeder base class

    Feeder consumes and outputs :class:`.Detection` data and can be used to
    modify the sequence, duplicate or drop data."""

    @abstractmethod
    def append(self, detection):
        """Append a new :class:`.Detection` to the Feeder"""
        raise NotImplementedError

    @abstractmethod
    def pop(self):
        """Pop a :class:`.Detection` from the Feeder"""
        raise NotImplementedError
