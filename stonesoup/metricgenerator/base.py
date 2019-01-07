# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..tracker import Tracker


class MetricGenerator(Base):
    """Metric Generator base class

    Generates :class:`.Metric` which is used to asses the performance of a run.
    Consumes :class:`.Track` data and optionally :py:class:`.GroundTruth` data.
    """

    @abstractmethod
    def compute_metric(self, manager, **kwargs):
        """Compute metric

        Parameters
        ----------
        prior : :class:`~.MetricManager`
            MetricManager containing the data

        Returns
        -------
        : :class:`~.Metric`
            Metric produced
        """
        raise NotImplementedError

class MetricManager(Base):
    """Metric Manager base class
        Holds the data and manages the production of Metrics through MetricGenerator classes
    """

class PlotGenerator(MetricGenerator):
    """
    PlotGenerator base class. For metrics that are plots. Should return a PlotMetric
    """