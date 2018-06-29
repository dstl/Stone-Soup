# -*- coding: utf-8 -*-
from path import Path

from ..base import Property
from ..serialise import YAML
from .base import Writer


class YAMLDetectionWriter(Writer):
    """YAML Detection Writer"""
    path = Property(Path, doc="File to save detections to")

    def __init__(self, path, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self._file = open(path, 'w')

        yaml = YAML()
        # Required as will be writing multiple documents to file
        yaml._yaml.explicit_start = True
        yaml._yaml.explicit_end = True
        self._yaml = yaml

    def write(self, time, detections):
        self._yaml.dump({"time": time, "detections": detections}, self._file)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._file.close()

    def __del__(self):
        self.__exit__()
