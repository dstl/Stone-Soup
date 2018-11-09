# -*- coding: utf-8 -*-
from .base import Base, Property


from .reader import DetectionReader, GroundTruthReader, SensorDataReader
from .tracker import Tracker
from .metricgenerator import MetricGenerator
from .writer import MetricsWriter, TrackWriter


class Run(Base):
    """Run base class

    Generates :class:`.Run` which describes a single configured instance of the
    framework; an end to end sequence of components.
    """

    detectionsource = Property(
        DetectionReader,
        doc="A source of detections, be it simulator, sensor, or file")

    tracker = Property(
        Tracker,
        doc="Generates tracks from detections")

    groundtruthsource = Property(
        GroundTruthReader,
        default=None,
        doc="A source of ground truth data")

    sensordatareader = Property(
        SensorDataReader,
        default=None,
        doc="A source of sensor data")

    metricgenerator = Property(
        MetricGenerator,
        default=None,
        doc="Generate metrics from tracker output")

    trackoutput = Property(
        TrackWriter,
        default=None,
        doc="Writes tracker output")

    metricoutput = Property(
        MetricsWriter,
        default=None,
        doc="Writes metric output")
