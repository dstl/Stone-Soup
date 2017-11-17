# -*- coding: utf-8 -*-
class MetricGenerator():
    """Metric Generator base class

    Generates :class:`.Metric` which is used to asses the performance of a run.
    Consumes :class:`.Track` data and optionally :py:class:`.GroundTruth` data.
    """
