from pathlib import Path

from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..serialise import YAML
from .base import DetectionReader, GroundTruthReader, SensorDataReader
from ..tracker import Tracker
from .file import FileReader


class YAMLReader(FileReader, BufferedGenerator):
    """YAML Reader"""
    path: Path = Property(doc="File to read data from")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._yaml = YAML(typ='safe')

    @BufferedGenerator.generator_method
    def data_gen(self):
        for document in self._yaml.load_all(self.path):
            yield document.pop('time'), document


class YAMLDetectionReader(YAMLReader, DetectionReader):
    """YAML Detection Reader"""

    def data_gen(self):
        yield from super().data_gen()

    @BufferedGenerator.generator_method
    def detections_gen(self):
        for time, document in self.data_gen():
            yield time, document.get('detections', set())


class YAMLGroundTruthReader(YAMLReader, GroundTruthReader):
    """YAML Ground Truth Reader"""

    def data_gen(self):
        yield from super().data_gen()

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        paths = dict()
        for time, document in self.data_gen():
            updated_paths = set()
            for path in document.get('groundtruth_paths', set()):
                if path.id in paths:
                    paths[path.id].states = path.states
                else:
                    paths[path.id] = path
                updated_paths.add(paths[path.id])

            yield time, updated_paths


class YAMLSensorDataReader(YAMLReader, SensorDataReader):
    """YAML Sensor Data Reader"""

    def data_gen(self):
        yield from super().data_gen()

    @BufferedGenerator.generator_method
    def sensor_data_gen(self):
        for time, document in self.data_gen():
            yield time, document.get('sensor_data', set())


class YAMLTrackReader(YAMLReader):
    """YAML Track Reader"""

    def data_gen(self):
        yield from super().data_gen()

    def __iter__(self):
        self.data_iter = iter(self.data_gen())
        self._tracks = dict()
        return self

    @property
    def tracks(self):
        return self._tracks

    def __next__(self):
        time, document = next(self.data_iter)
        updated_tracks = set()
        for track in document.get('tracks', set()):
            if track.id in self.tracks:
                self._tracks[track.id].states = track.states
            else:
                self._tracks[track.id] = track
            updated_tracks.add(self.tracks[track.id])

        return time, updated_tracks
