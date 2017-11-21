# -*- coding: utf-8 -*-
"""Reader classes are used for getting data into the framework."""
from .base import DetectionReader, GroundTruthReader, SensorDataReader

from . import file  # noqa: F401

__all__ = ['DetectionReader', 'GroundTruthReader', 'SensorDataReader']
