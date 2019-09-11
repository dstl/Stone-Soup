# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base


class MetricGenerator(Base):
    """Metric Generator base class

    Generates :class:`~.Metric` objects used to asses the performance of a
    tracker using data held in a :class:`~.MetricManager` object
    """

    @abstractmethod
    def compute_metric(self, manager, **kwargs):
        """Compute metric

        Parameters
        ----------
        manager : MetricManager
            containing the data to be used to create the metric(s)

        Returns
        -------
        : list of :class:`~.Metric` objects
            Generated metrics
        """
        raise NotImplementedError


class MetricManager(Base):
    """Metric Manager base class

    Holds the data and manages the production of :class:`~.Metric` objects
    through a :class:`~.MetricGenerator`
    """


class MetricTableGenerator(Base):
    """Metric Table base class

    Takes a set of :class: `~.Metric` objects and outputs a table of the
    values next to a description and target for each metric"""

    @abstractmethod
    def generate_table(self, **kwargs):
        """Generate table

        Parameters
        ----------
        : set of :class: `~.Metric` objects

        Returns
        -------
        matplotlib.Table object"""

        raise NotImplementedError


class PlotGenerator(MetricGenerator):
    """PlotGenerator base class

    PlotGenerators generate metrics that are visualisations. Return
    :class:`~.PlottingMetric` objects.
    """
