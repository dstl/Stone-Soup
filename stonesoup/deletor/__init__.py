# -*- coding: utf-8 -*-
from .base import Deletor
from .simple import *  # noqa:F401,F403

__all__ = ['Deletor']
__all__.extend(subclass_.__name__ for subclass_ in Deletor.subclasses)
