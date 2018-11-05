# -*- coding: utf-8 -*-
"""Generic readers for Stone Soup.

This is a collection of generic readers for Stone Soup, allowing quick reading
of data that is in common formats.
"""

import csv
from datetime import datetime

import numpy as np
from dateutil.parser import parse

from ..base import Property
from ..types import Detection
from .base import DetectionReader
from .file import TextFileReader


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = set()

    @property
    def detections(self):
        return self._detections.copy()

    def detections_gen(self):
        with self.path.open(encoding=self.encoding, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if self.time_field_format is not None:
                    time_field_value = datetime.strptime(
                        row[self.time_field], self.time_field_format)
                elif self.timestamp is True:
                    time_field_value = datetime.utcfromtimestamp(
                        int(row[self.time_field]))
                else:
                    time_field_value = parse(row[self.time_field])

                if self.metadata_fields is None:
                    local_metadata = dict(row)
                    copy_local_metadata = dict(local_metadata)
                    for (key, value) in copy_local_metadata.items():
                        if (key == self.time_field) or\
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
                self._detections = {detect}
                yield time_field_value, self.detections
