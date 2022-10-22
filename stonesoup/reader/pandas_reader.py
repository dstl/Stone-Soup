#!/usr/bin/env python
import numpy as np

from datetime import datetime, timedelta
from dateutil.parser import parse
from math import modf
from typing import Sequence, Collection

try:
    import pandas as pd
except ImportError as error:
    raise ImportError(
        "Pandas Readers require dependency 'pandas' being installed. ") from error

from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..reader.base import GroundTruthReader, DetectionReader, Reader
from ..types.detection import Detection
from ..types.groundtruth import GroundTruthPath, GroundTruthState


class _DataFrameReader(Reader):
    state_vector_fields: Sequence[str] = Property(
        doc='List of columns names to be used in state vector')
    time_field: str = Property(
        doc='Name of column to be used as time field')
    time_field_format: str = Property(
        default=None, doc='Optional datetime format')
    timestamp: bool = Property(
        default=False, doc='Treat time field as a timestamp from epoch')
    metadata_fields: Collection[str] = Property(
        default=None, doc='List of columns to be saved as metadata, default all')

    def _get_metadata(self, row):
        if self.metadata_fields is None:
            local_metadata = dict(row)
            for key in list(local_metadata):
                if key == self.time_field or key in self.state_vector_fields:
                    del local_metadata[key]
        else:
            local_metadata = {field: row[field]
                              for field in self.metadata_fields
                              if field in row}
        return local_metadata

    def _get_time(self, row):
        if self.time_field_format is not None:
            time_field_value = datetime.strptime(row[self.time_field], self.time_field_format)
        elif self.timestamp:
            fractional, timestamp = modf(float(row[self.time_field]))
            time_field_value = datetime.utcfromtimestamp(int(timestamp))
            time_field_value += timedelta(microseconds=fractional * 1E6)
        else:
            time_field_value = row[self.time_field]

            if not isinstance(time_field_value, datetime):
                time_field_value = parse(time_field_value, ignoretz=True)

        return time_field_value


class DataFrameGroundTruthReader(GroundTruthReader, _DataFrameReader):
    """A custom reader for pandas DataFrames containing truth data.

    The DataFrame must have colums containing all fields needed to generate the
    ground truth state. Those states with the same ID will be put into
    a :class:`~.GroundTruthPath` in sequence, and all paths that are updated at the same time
    are yielded together, and such assumes file is in time order.

    Parameters
    ----------
    """
    dataframe: pd.DataFrame = Property(doc="DataFrame containing the ground truth data.")
    path_id_field: str = Property(doc='Name of column to be used as path ID')

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        """ Generator method for providing each row of ground truth data. """
        groundtruth_dict = {}
        updated_paths = set()
        previous_time = None
        for row in self.dataframe.to_dict(orient="records"):

            time = self._get_time(row)
            if previous_time is not None and previous_time != time:
                yield previous_time, updated_paths
                updated_paths = set()
            previous_time = time

            state = GroundTruthState(np.array([[row[col_name]] for col_name
                                              in self.state_vector_fields],
                                              dtype=np.float_), timestamp=time,
                                     metadata=self._get_metadata(row))

            id_ = row[self.path_id_field]
            if id_ not in groundtruth_dict:
                groundtruth_dict[id_] = GroundTruthPath(id=id_)
            groundtruth_path = groundtruth_dict[id_]
            groundtruth_path.append(state)
            updated_paths.add(groundtruth_path)

            # Yield remaining
        yield previous_time, updated_paths


class DataFrameDetectionReader(DetectionReader, _DataFrameReader):
    """A custom detection reader for DataFrames containing detections.

    DataFrame must have headers with the appropriate fields needed to generate
    the detection. Detections at the same time are yielded together, and such assume file is in
    time order.

    Parameters
    ----------
    """
    dataframe: pd.DataFrame = Property(doc="DataFrame containing the ground truth data.")

    @BufferedGenerator.generator_method
    def detections_gen(self):
        detections = set()
        previous_time = None
        for row in self.dataframe.to_dict(orient="records"):

            time = self._get_time(row)
            if previous_time is not None and previous_time != time:
                yield previous_time, detections
                detections = set()
            previous_time = time

            detections.add(Detection(
                np.array([[row[col_name]] for col_name in self.state_vector_fields],
                         dtype=np.float_),
                timestamp=time,
                metadata=self._get_metadata(row)))

        # Yield remaining
        yield previous_time, detections
