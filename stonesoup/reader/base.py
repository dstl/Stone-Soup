# -*- coding: utf-8 -*-
from ..base import BaseMeta


class Reader(metaclass=BaseMeta):
    """Reader base class"""


class DetectionReader(Reader):
    """Detection Reader base class"""


class GroundTruthReader(Reader):
    """Ground Truth Reader base class"""


class SensorDataReader(Reader):
    """Sensor Data Reader base class"""
