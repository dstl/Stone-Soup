# -*- coding: utf-8 -*-
"""Provides an ability to generate and load configuration from YAML.

Stone Soup utilises YAML_ for configuration files. The :doc:`stonesoup.base`
feature of components is exploited in order to store the configuration of the
components used for a run.

.. _YAML: http://yaml.org/"""
from abc import ABC, abstractmethod
from io import StringIO

from .serialise import YAML
from .models.measurement.base import MeasurementModel
from .reader.base import DetectionReader, GroundTruthReader
from .tracker.base import Tracker


class Configuration:
    pass


class ConfigurationFile(ABC):
    """Base configuration class."""

    @abstractmethod
    def dump(self, data, stream, *args, **kwargs):
        """Dump configuration to a stream."""
        raise NotImplementedError

    def dumps(self, data, *args, **kwargs):
        """Return configuration as a string."""
        stream = StringIO()
        self.dump(data, stream, *args, **kwargs)
        return stream.getvalue()

    @abstractmethod
    def load(self, stream):
        """Load configuration from a stream."""
        raise NotImplementedError


class YAMLConfig(YAML):
    def __init__(self):
        super().__init__()
        self.detection_readers = set()
        self.groundtruth_readers = set()
        self.trackers = set()
        self.measurement_models = set()

    def declarative_from_yaml(self, constructor, tag_suffix, node):
        component = super().declarative_from_yaml(
            constructor, tag_suffix, node)
        if isinstance(component, DetectionReader):
            self.detection_readers.add(component)
        elif isinstance(component, GroundTruthReader):
            self.groundtruth_readers.add(component)
        elif isinstance(component, Tracker):
            self.trackers.add(component)
        elif isinstance(component, MeasurementModel):
            self.measurement_models.add(component)
        return component
