# -*- coding: utf-8 -*-
"""Base classes for different Readers."""
from ..base import Base


class Reader(Base):
    """Reader base class"""


class DetectionReader(Reader):
    """Detection Reader base class"""


class GroundTruthReader(Reader):
    """Ground Truth Reader base class"""


class SensorDataReader(Reader):
    """Sensor Data Reader base class"""
