# -*- coding: utf-8 -*-
from .base import Hypothesis

__all__ = ['Hypothesis']
__all__.extend(subclass_.__name__ for subclass_ in Hypothesis.subclasses)
