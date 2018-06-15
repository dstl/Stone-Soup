# -*- coding: utf-8 -*-
from ..base import Base, Property
from ..tracker import Tracker


class MetricGenerator(Base):
    """Metric Generator base class

    Generates :class:`.Metric` which is used to asses the performance of a run.
    Consumes :class:`.Track` data and optionally :py:class:`.GroundTruth` data.
    """

    tracker = Property(
        Tracker, doc="Tracks which metrics will be generated for")
