# -*- coding: utf-8 -*-
from abc import abstractmethod
from ..base import Base
from ..reader.base import DetectionReader, GroundTruthReader, SensorDataReader
from ..detector import Detector


class Simulator(Base):
    """Simulator base class"""


class DetectionSimulator(Simulator, DetectionReader, Detector):
    """Detection Simulator base class"""
    def detections_gen(self):
        """Returns a generator of detections for each time step."""
        raise NotImplemented


class GroundTruthSimulator(Simulator, GroundTruthReader):
    """Ground truth simulator"""
    @abstractmethod
    def groundtruth_paths_gen(self):
        """Returns a generator of tracks for each time step."""
        raise NotImplemented


class SensorSimulator(Simulator, SensorDataReader):
    """Sensor Simulator base class"""
    @abstractmethod
    def sensordata_gen(self):
        """Returns a generator of sensor data for each time step."""
        raise NotImplemented
