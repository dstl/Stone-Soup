# -*- coding: utf-8 -*-
from pathlib import Path

from ..base import Property
from ..serialise import YAML
from .base import DetectionReader


class YAMLDetectionReader(DetectionReader):
    """YAML Detection Writer"""
    path = Property(Path, doc="File to save detections to")
    _yaml = YAML()

    def __init__(self, path, *args, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)  # Ensure Path
        super().__init__(path, *args, **kwargs)
        self._detections = set()

    @property
    def detections(self):
        return self._detections

    def detections_gen(self):
        for document in self._yaml.load_all(self.path):
            self._detections = document['detections']
            yield document['time'], document['detections']
