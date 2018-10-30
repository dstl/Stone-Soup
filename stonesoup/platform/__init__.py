# -*- coding: utf-8 -*-
from .base import Platform
from .simple import *  # noqa:F401,F403

__all__ = ['Platform']
__all__.extend(subclass_.__name__ for subclass_ in Platform.subclasses)
