# -*- coding: utf-8 -*-
from pathlib import Path

from ..base import Property
from ..serialise import YAML
from ..reader import DetectionReader, GroundTruthReader, SensorDataReader
from ..tracker import Tracker
from .base import Writer


class YAMLWriter(Writer):
    """YAML Writer"""
    path = Property(Path,
                    doc="File to save data to. Str will be converted to Path")
    groundtruth_source = Property(GroundTruthReader, default=None)
    sensor_data_source = Property(SensorDataReader, default=None)
    detections_source = Property(DetectionReader, default=None)
    tracks_source = Property(Tracker, default=None)

    def __init__(self, path, *args, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)  # Ensure Path
        super().__init__(path, *args, **kwargs)
        if not any((self.groundtruth_source, self.sensor_data_source,
                   self.detections_source, self.tracks_source)):
            raise ValueError("At least one source required")

        self._file = self.path.open('w')

        yaml = YAML()
        # Required as will be writing multiple documents to file
        yaml._yaml.explicit_start = True
        yaml._yaml.explicit_end = True
        self._yaml = yaml

    def write(self):
        if self.tracks_source:
            gen = self.tracks_source
        elif self.detections_source:
            gen = self.detections_source
        elif self.sensor_data_source:
            gen = self.sensor_data_source
        elif self.groundtruth_source:
            gen = self.groundtruth_source
        else:  # pragma: no cover
            raise RuntimeError("At least one source required")

        for time, _ in gen:
            data = {'time': time}
            if self.tracks_source:
                data['tracks'] = self.tracks_source.tracks
            if self.detections_source:
                data['detections'] = self.detections_source.detections
            if self.sensor_data_source:
                data['sensor_data'] = self.sensor_data_source.sensor_data
            if self.groundtruth_source:
                data['groundtruth_paths'] = \
                    self.groundtruth_source.groundtruth_paths
            self._yaml.dump(data, self._file)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if getattr(self, '_file', None):
            self._file.close()

    def __del__(self):
        self.__exit__()
