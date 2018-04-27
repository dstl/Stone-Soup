# -*- coding: utf-8 -*-
"""Base classes for different Readers."""
from abc import abstractmethod

from ..base import Base


class Reader(Base):
    """Reader base class"""


class DetectionReader(Reader):
    """Detection Reader base class"""
    @abstractmethod
    def detections_gen(self):
        raise NotImplemented


class GroundTruthReader(Reader):
    """Ground Truth Reader base class"""
    @abstractmethod
    def groundtruth_paths_gen(self):
        raise NotImplemented


class SensorDataReader(Reader):
    """Sensor Data Reader base class"""
    @abstractmethod
    def sensordata_gen(self):
        raise NotImplemented
