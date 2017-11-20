# -*- coding: utf-8 -*-
from ..base import BaseMeta


class Writer(metaclass=BaseMeta):
    """Writer base class"""


class MetricsWriter(Writer):
    """Metrics Writer base class"""


class TrackWriter(Writer):
    """Track Writer base class"""
