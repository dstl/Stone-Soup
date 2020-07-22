# -*- coding: utf-8 -*-
"""
Objects that include calls to code in different languages should inherit from
the relevant wrapper along with the standard inheritance. The wrapper objects
include functions for managing the bridge between languages and converting
between data types.
"""
from .base import Wrapper

__all__ = ['Wrapper']
