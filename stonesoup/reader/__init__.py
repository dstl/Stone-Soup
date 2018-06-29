# -*- coding: utf-8 -*-
"""Reader classes are used for getting data into the framework."""
from .base import Reader, DetectionReader, GroundTruthReader, SensorDataReader

from .generic import *  # noqa:F401,F403
from .file import *  # noqa:F401,F403
from .yaml import *  # noqa:F401,F403

__all__ = ['DetectionReader', 'GroundTruthReader', 'SensorDataReader']
__all__.extend(subclass_.__name__ for subclass_ in Reader.subclasses)
