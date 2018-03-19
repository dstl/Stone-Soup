# -*- coding: utf-8 -*-
from abc import abstractmethod
from ..base import Base
from ..reader.base import DetectionReader, GroundTruthReader, SensorDataReader


class Simulator(Base):
    """Simulator base class"""


class DetectionSimulator(Simulator, DetectionReader):
    """Detection Simulator base class"""
    def get_detections(self):
        """Returns a generator of detections for each time step."""
        raise NotImplemented


class GroundTruthSimulator(Simulator, GroundTruthReader):
    """Ground truth simulator"""
    @abstractmethod
    def get_tracks(self):
        """Returns a generator of tracks for each time step."""
        raise NotImplemented


class SensorSimulator(Simulator, SensorDataReader):
    """Sensor Simulator base class"""
    @abstractmethod
    def get_sensordata(self):
        """Returns a generator of sensor data for each time step."""
        raise NotImplemented
