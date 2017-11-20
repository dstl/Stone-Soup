# -*- coding: utf-8 -*-
from ..base import BaseMeta


class Simulator(metaclass=BaseMeta):
    """Simulator base class"""


class DetectionSimulator(Simulator):
    """Detection Simulator base class"""


class SensorSimulator(Simulator):
    """Sensor Simulator base class"""
