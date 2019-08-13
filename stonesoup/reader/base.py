# -*- coding: utf-8 -*-
"""Base classes for different Readers."""
from abc import abstractmethod

from ..base import Base
from stonesoup.buffered_generator import BufferedGenerator


class Reader(Base, BufferedGenerator):
    """Reader base class"""


class DetectionReader(Reader):
    """Detection Reader base class"""

    @abstractmethod
    def detections_gen(self):
        """Returns a generator of detections for each time step.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.Detection`
            Detections generate in the time step
        """
        raise NotImplementedError


class GroundTruthReader(Reader):
    """Ground Truth Reader base class"""

    @abstractmethod
    def groundtruth_paths_gen(self):
        """Returns a generator of ground truth paths for each time step.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.GroundTruthPath`
            Ground truth paths existing in the time step
        """
        raise NotImplementedError


class SensorDataReader(Reader):
    """Sensor Data Reader base class"""

    @abstractmethod
    def sensor_data_gen(self):
        """Returns a generator of sensor data for each time step.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.SensorData`
            Sensor data generated in the time step
        """
        raise NotImplementedError
