# -*- coding: utf-8 -*-
from .base import Detector

__all__ = ['Detector']
__all__.extend(subclass_.__name__ for subclass_ in Detector.subclasses)
