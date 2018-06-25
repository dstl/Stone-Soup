# -*- coding: utf-8 -*-
from abc import abstractmethod
from ..base import Base
from ..reader.base import DetectionReader, GroundTruthReader, SensorDataReader


class Simulator(Base):
    """Simulator base class"""


class DetectionSimulator(Simulator, DetectionReader):
    """Detection Simulator base class"""

    @abstractmethod
    def detections_gen(self):
        raise NotImplementedError


class GroundTruthSimulator(Simulator, GroundTruthReader):
    """Ground truth simulator"""

    @abstractmethod
    def groundtruth_paths_gen(self):
        raise NotImplementedError


class SensorSimulator(Simulator, SensorDataReader):
    """Sensor Simulator base class"""

    @abstractmethod
    def sensordata_gen(self):
        raise NotImplementedError
