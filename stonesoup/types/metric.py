# -*- coding: utf-8 -*-
from .base import Type
from .base import Property


class Metric(Type):
    """Metric type"""

    title = Property(str, doc = 'Name of the metric')
    # value = Property(any, doc = 'Value of the metric') # I don't know how to code this