# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Property
from ..reader import DetectionReader, SensorDataReader


class Detector(DetectionReader):
    """Detector base class

    A Detector processes :class:`~.SensorData` to generate :class:`~.Detection`
    data.
    """

    sensor = Property(SensorDataReader, doc="Source of sensor data")

    @abstractmethod
    def detections_gen(self):
        raise NotImplementedError
