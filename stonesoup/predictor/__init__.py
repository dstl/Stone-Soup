# -*- coding: utf-8 -*-
from .base import Predictor
from .kalman import *  # noqa:F401,F403

__all__ = ['Predictor']
__all__.extend(subclass_.__name__ for subclass_ in Predictor.subclasses)
