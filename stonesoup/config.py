# -*- coding: utf-8 -*-
"""Provides an ability to generate and load configuration from YAML.

Stone Soup utilises YAML_ for configuration files. The :doc:`stonesoup.base`
feature of components is exploited in order to store the configuration of the
components used for a run.

.. _YAML: http://yaml.org/"""
from abc import ABC, abstractmethod
from io import StringIO


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
