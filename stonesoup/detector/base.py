# -*- coding: utf-8 -*-
from ..base import Base


class Detector(Base):
    """Detector base class

    A Detector processes :class:`.SensorData` to generate :class:`.Detection`
    data."""
