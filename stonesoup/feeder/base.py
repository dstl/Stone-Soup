# -*- coding: utf-8 -*-
"""Base classes for Stone Soup feeder"""
from ..base import Property
from ..reader import DetectionReader


class Feeder(DetectionReader):
    """Feeder base class

    Feeder consumes and outputs :class:`.Detection` data and can be used to
    modify the sequence, duplicate or drop data.
    """

    detector = Property(DetectionReader, doc="Source of detections")
