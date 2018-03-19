# -*- coding: utf-8 -*-
"""Simulators for data input into Stone Soup.

Stone Soup can make use of simulators to generate ground truth tracks, sensor
data and detections. These are similar to :class:`reader.Reader`, but data is
generated, rather than read from a file, etc. They should come with various
configuration options to allow customisation of the simulation.
"""

from .base import (
    Simulator, DetectionSimulator, GroundTruthSimulator, SensorSimulator)

__all__ = ['Simulator']
__all__.extend(subclass_.__name__ for subclass_ in Simulator.subclasses)
