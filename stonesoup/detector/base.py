# -*- coding: utf-8 -*-
from ..base import BaseMeta


class Detector(metaclass=BaseMeta):
    """Detector base class

    A Detector processes :class:`.SensorData` to generate :class:`.Detection`
    data."""
