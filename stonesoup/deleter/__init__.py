# -*- coding: utf-8 -*-
from .base import Deleter
from .error import *  # noqa:F401,F403

__all__ = ['Deleter']
__all__.extend(subclass_.__name__ for subclass_ in Deleter.subclasses)
