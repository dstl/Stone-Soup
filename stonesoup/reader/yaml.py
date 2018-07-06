# -*- coding: utf-8 -*-
from pathlib import Path

from ..base import Property
from ..serialise import YAML
from .base import DetectionReader, GroundTruthReader, SensorDataReader


class YAMLReader(DetectionReader, GroundTruthReader, SensorDataReader):
    """YAML Detection Writer"""
    path = Property(Path, doc="File to read data from")
    _yaml = YAML()

    def __init__(self, path, *args, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)  # Ensure Path
        super().__init__(path, *args, **kwargs)
        self._detections = set()
        self._groundtruth_paths = set()
        self._sensor_data = set()
        self._tracks = set()

    @property
    def detections(self):
        return self._detections

    @property
    def groundtruth_paths(self):
        return self._groundtruth_paths

    @property
    def sensor_data(self):
        return self._sensor_data

    @property
    def tracks(self):
        return self._tracks

    def detections_gen(self):
        for time, _ in self.data_gen():
            yield time, self.detections

    def groundtruth_paths_gen(self):
        for time, _ in self.data_gen():
            yield time, self.groundtruth_paths

    def sensor_data_gen(self):
        for time, _ in self.data_gen():
            yield time, self.sensor_data

    def tracks_gen(self):
        for time, _ in self.data_gen():
            yield time, self.tracks

    def data_gen(self):
        for document in self._yaml.load_all(self.path):
            self._detections = document.get('detections', set())
            self._groundtruth_paths = document.get('groundtruth_paths', set())
            self._sensor_data = document.get('sensor_data', set())
            self._tracks = document.get('tracks', set())
            yield document.pop('time'), document
