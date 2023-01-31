from pathlib import Path
from typing import Union, Set

from stonesoup.metricgenerator.base import MetricManager
from stonesoup.simulator import GroundTruthSimulator, DetectionSimulator
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath
from ..base import Property
from ..serialise import YAML
from ..reader import DetectionReader, GroundTruthReader, SensorDataReader
from ..tracker import Tracker
from .base import Writer


class YAMLWriter(Writer):
    """YAML Writer"""
    path: Path = Property(doc="File to save data to. Str will be converted to Path")
    groundtruth_source: GroundTruthReader = Property(default=None)
    sensor_data_source: SensorDataReader = Property(default=None)
    detections_source: DetectionReader = Property(default=None)
    tracks_source: Tracker = Property(default=None)

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
        yaml.explicit_start = True
        yaml.explicit_end = True
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


class YAMLConfigWriter(Writer):
    """Run Manager YAML Configuration Writer

    YAML Writer for generating Run Manager Configuration files from Tracker and Simulator
    components
    """
    path: Path = Property(doc="File to save data to. Str will be converted to Path")
    tracker: Tracker = Property(
        doc="The tracker used in the run manager.")
    groundtruths: Union[Set[GroundTruthPath], GroundTruthSimulator] = Property(
        doc="The ground truth paths required by the run manager.",
        default=None)
    detections: Union[Set[Detection], DetectionSimulator] = Property(
        doc="The detections for use with the run manager",
        default=None)
    metricmanager: MetricManager = Property(
        doc="Metric manager containing metrics to be processed using the run manager.",
        default=None)

    def __init__(self, path, *args, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)  # Ensure Path
        super().__init__(path, *args, **kwargs)
        if not any((self.tracker, self.groundtruths,
                   self.detections, self.metricmanager)):
            raise ValueError("At least one object required to write to YAML file.")
        self.kwargs = kwargs
        self._file = self.path.open('w')

        self.data = dict()
        self._yaml = YAML(typ="SAFE")

    def write(self):
        if self.tracker is not None:
            self.data["tracker"] = self.tracker
        if self.groundtruths is not None:
            self.data["groundtruth"] = self.groundtruths
        if self.detections is not None:
            self.data["detections"] = self.detections
        if self.metricmanager is not None:
            self.data["metric_manager"] = self.metricmanager
        if self.kwargs:
            for k in self.kwargs.keys():
                self.data[k] = self.kwargs[k]
        self._yaml.dump(self.data, self._file)
