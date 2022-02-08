# -*- coding: utf-8 -*-
"""Base classes for different Readers."""
from abc import abstractmethod

from ..base import Base
from ..buffered_generator import BufferedGenerator


class Reader(Base, BufferedGenerator):
    """Reader base class"""


class DetectionReader(Reader):
    """Detection Reader base class"""

    @property
    def detections(self):
        return self.current[1]

    @abstractmethod
    @BufferedGenerator.generator_method
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

    @property
    def groundtruth_paths(self):
        return self.current[1]

    @abstractmethod
    @BufferedGenerator.generator_method
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

    @property
    def sensor_data(self):
        return self.current[1]

    @abstractmethod
    @BufferedGenerator.generator_method
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


class FrameReader(SensorDataReader):
    """FrameReader base class

    A FrameReader produces :class:`~.SensorData` in the form of
    :class:`~ImageFrame` objects.
    """

    @property
    def frame(self):
        return self.sensor_data

    @abstractmethod
    @BufferedGenerator.generator_method
    def frames_gen(self):
        """Returns a generator of frames for each time step.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.ImageFrame`
            Generated frame in the time step
        """
        raise NotImplementedError

    @BufferedGenerator.generator_method
    def sensor_data_gen(self):
        """Returns a generator of frames for each time step.

        Note
        ----
        This is just a wrapper around (and therefore performs identically
        to) :meth:`~frames_gen`.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.ImageFrame`
            Generated frame in the time step
        """
        yield from self.frames_gen()
