# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base


class Detector(Base):
    """Detector base class

    A Detector processes :class:`.SensorData` to generate :class:`.Detection`
    data."""

    @abstractmethod
    def detections_gen(self):
        """Returns a generator of detections for each time step."""
        raise NotImplemented
