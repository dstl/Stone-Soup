# -*- coding: utf-8 -*-
from ..base import Base
from ..reader.base import DetectionReader, GroundTruthReader, SensorDataReader


class Simulator(Base):
    """Simulator base class"""


class DetectionSimulator(Simulator, DetectionReader):
    """Detection Simulator base class"""


class GroundTruthSimulator(Simulator, GroundTruthReader):
    """Ground truth simulator"""


class SensorSimulator(Simulator, SensorDataReader):
    """Sensor Simulator base class"""
