# -*- coding: utf-8 -*-
"""Generic readers for Stone Soup.

This is a collection of generic readers for Stone Soup, allowing quick reading
of data that is in common formats.
"""

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from math import modf

import numpy as np
from dateutil.parser import parse

from ..base import Property
from ..types.detection import Detection
from .base import GroundTruthReader, DetectionReader
from .file import TextFileReader
from stonesoup.buffered_generator import BufferedGenerator
from ..types.groundtruth import GroundTruthPath, GroundTruthState


class CSVGroundTruthReader(GroundTruthReader, TextFileReader):
    state_vector_fields = Property(
        [str], doc='List of columns names to be used in state vector')
    time_field = Property(
        str, doc='Name of column to be used as time field')
    time_field_format = Property(
        str, default=None, doc='Optional datetime format')
    timestamp = Property(
        bool, default=False, doc='Treat time field as a timestamp from epoch')
    path_id_field = Property(
        str, doc='Name of column to be used as path ID')
    csv_options = Property(
        dict, default={},
        doc='Keyword arguments for the underlying csv reader')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._groundtruth_paths = set()

    @property
    def groundtruth_paths(self):
        return self._groundtruth_paths.copy()

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        with self.path.open(encoding=self.encoding, newline='') as csv_file:
            groundtruth_dict = defaultdict(GroundTruthPath)

            reader = csv.DictReader(csv_file, **self.csv_options)
            for row in reader:
                if self.time_field_format is not None:
                    time_field_value = datetime.strptime(
                        row[self.time_field], self.time_field_format)
                elif self.timestamp is True:
                    fractional, timestamp = modf(float(row[self.time_field]))
                    time_field_value = datetime.utcfromtimestamp(
                        int(timestamp))
                    time_field_value += timedelta(microseconds=fractional*1E6)
                else:
                    time_field_value = parse(row[self.time_field])

                state = GroundTruthState(np.array(
                    [[row[col_name]] for col_name in self.state_vector_fields],
                    dtype=np.float32), time_field_value)

                groundtruth_dict[row[self.path_id_field]].append(state)
                self._groundtruth_paths = set(groundtruth_dict.values())

                yield time_field_value, self._groundtruth_paths


class CSVDetectionReader(DetectionReader, TextFileReader):
    """A simple detection reader for csv files of detections.

    CSV file must have headers, as these are used to determine which fields
    to use to generate the detection.

    Parameters
    ----------
    """

    state_vector_fields = Property(
        [str], doc='List of columns names to be used in state vector')
    time_field = Property(
        str, doc='Name of column to be used as time field')
    time_field_format = Property(
        str, default=None, doc='Optional datetime format')
    timestamp = Property(
        bool, default=False, doc='Treat time field as a timestamp from epoch')
    metadata_fields = Property(
        [str], default=None, doc='List of columns to be saved as metadata, '
                                 'default all')
    csv_options = Property(
        dict, default={},
        doc='Keyword arguments for the underlying csv reader')

    @BufferedGenerator.generator_method
    def detections_gen(self):
        with self.path.open(encoding=self.encoding, newline='') as csv_file:
            reader = csv.DictReader(csv_file, **self.csv_options)
            for row in reader:
                if self.time_field_format is not None:
                    time_field_value = datetime.strptime(
                        row[self.time_field], self.time_field_format)
                elif self.timestamp is True:
                    fractional, timestamp = modf(float(row[self.time_field]))
                    time_field_value = datetime.utcfromtimestamp(
                        int(timestamp))
                    time_field_value += timedelta(microseconds=fractional*1E6)
                else:
                    time_field_value = parse(row[self.time_field])

                if self.metadata_fields is None:
                    local_metadata = dict(row)
                    copy_local_metadata = dict(local_metadata)
                    for (key, value) in copy_local_metadata.items():
                        if (key == self.time_field) or \
                                (key in self.state_vector_fields):
                            del local_metadata[key]
                else:
                    local_metadata = {field: row[field]
                                      for field in self.metadata_fields
                                      if field in row}

                detect = Detection(np.array(
                    [[row[col_name]] for col_name in self.state_vector_fields],
                    dtype=np.float32), time_field_value,
                    metadata=local_metadata)
                yield time_field_value, {detect}
