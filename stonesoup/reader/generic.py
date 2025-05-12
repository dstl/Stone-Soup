"""Generic readers for Stone Soup.

This is a collection of generic readers for Stone Soup, allowing quick reading
of data that is in common formats.
"""

import csv
from datetime import datetime, timedelta, timezone
from collections.abc import Collection, Mapping, Sequence, Iterable, Iterator

from math import modf

import numpy as np
from dateutil.parser import parse

from .base import GroundTruthReader, DetectionReader, Reader
from .file import TextFileReader
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.detection import Detection
from ..types.groundtruth import GroundTruthPath, GroundTruthState


class _DictReader(Reader):
    """Abstract reader for reading dictionaries and outputing :class:`~State`

    This class provides an abstract base for reading dictionaries and converting them into
    state vectors and metadata. It handles the extraction of time fields and metadata fields
    from the input dictionaries.
    """

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

    @property
    def _default_metadata_fields_to_ignore(self) -> set[str]:
        return {self.time_field} | set(self.state_vector_fields)

    def _get_metadata(self, row: dict) -> dict:
        if self.metadata_fields is None:
            local_metadata = {
                key: value
                for key, value in row.items()
                if key not in self._default_metadata_fields_to_ignore
            }
        else:
            local_metadata = {
                key: value
                for key, value in row.items()
                if key in self.metadata_fields
            }
        return local_metadata

    def _get_time(self, row: dict) -> datetime:
        if self.time_field_format is not None:
            time_field_value = datetime.strptime(row[self.time_field], self.time_field_format)
        elif self.timestamp is True:
            fractional, timestamp = modf(float(row[self.time_field]))
            time_field_value = datetime.fromtimestamp(
                int(timestamp), timezone.utc).replace(tzinfo=None)
            time_field_value += timedelta(microseconds=fractional * 1E6)
        else:
            time_field_value = row[self.time_field]

            if not isinstance(time_field_value, datetime):
                time_field_value = parse(time_field_value, ignoretz=True)

        return time_field_value


class _DictionaryReader(_DictReader):
    dictionaries: Iterator[dict] = Property(
        doc='A source of :class:`dict` data that contains state information.')

    @property
    def dict_reader(self) -> Iterator[dict]:
        yield from self.dictionaries


class _CSVReader(_DictReader, TextFileReader):
    csv_options: Mapping = Property(
        default={}, doc='Keyword arguments for the underlying csv reader')

    @property
    def dict_reader(self) -> Iterator[dict]:
        with self.path.open(encoding=self.encoding, newline='') as csv_file:
            yield from csv.DictReader(csv_file, **self.csv_options)


class _DictGroundTruthReader(GroundTruthReader, _DictReader):
    """An abstract reader for dictionaries containing truth data."""

    path_id_field: str = Property(doc='Name of column to be used as path ID')

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self) -> Iterator[tuple[datetime, set[GroundTruthPath]]]:

        """
        Generator method that yields :class:`~.GroundTruthPath`s.

        This method reads rows from the dictionary reader, processes them into ground truth states,
        and groups them into ground truth paths based on the path ID field. It yields the paths
        at each unique timestamp.

        Yields
        ------
        tuple[datetime, set[GroundTruthPath]]
            A tuple containing the timestamp and a set of updated ground truth paths.
        """

        groundtruth_dict = {}
        updated_paths = set()
        previous_time = None
        for row in self.dict_reader:

            time = self._get_time(row)
            if previous_time is not None and previous_time != time:
                yield previous_time, updated_paths
                updated_paths = set()
            previous_time = time

            state = GroundTruthState(
                np.array([[row[col_name]] for col_name in self.state_vector_fields],
                            dtype=np.float64),
                timestamp=time,
                metadata=self._get_metadata(row))

            id_ = row[self.path_id_field]
            if id_ not in groundtruth_dict:
                groundtruth_dict[id_] = GroundTruthPath(id=id_)
            groundtruth_path = groundtruth_dict[id_]
            groundtruth_path.append(state)
            updated_paths.add(groundtruth_path)

        # Yield remaining
        yield previous_time, updated_paths

    @property
    def _default_metadata_fields_to_ignore(self) -> set[str]:
        return {self.path_id_field} | super()._default_metadata_fields_to_ignore


class DictionaryGroundTruthReader(_DictGroundTruthReader, _DictionaryReader):
    """TODO"""


class CSVGroundTruthReader(_DictGroundTruthReader, _CSVReader):
    """A simple reader for csv files of truth data.

    CSV file must have headers, as these are used to determine which fields
    to use to generate the ground truth state. Those states with the same ID will be put into
    a :class:`~.GroundTruthPath` in sequence, and all paths that are updated at the same time
    are yielded together, and such assumes file is in time order.

    Parameters
    ----------
    """


class _DictDetectionReader(DetectionReader, _DictReader):
    """An abstract reader for dictionaries containing detections."""

    @BufferedGenerator.generator_method
    def detections_gen(self) -> Iterator[tuple[datetime, set[Detection]]]:
        """Generator method that yields detections.

        This method reads rows (:class:`dict`) from the input source, processes them into
        detections, and groups them by unique timestamps. It yields the detections at each unique
        timestamp.
        """
        detections = set()
        previous_time = None
        for row in self.dict_reader:

            time = self._get_time(row)
            if previous_time is not None and previous_time != time:
                yield previous_time, detections
                detections = set()
            previous_time = time

            detections.add(Detection(
                np.array([[row[col_name]] for col_name in self.state_vector_fields],
                            dtype=np.float64),
                timestamp=time,
                metadata=self._get_metadata(row)))

        # Yield remaining
        yield previous_time, detections


class DictionaryDetectionReader(_DictDetectionReader, _DictionaryReader):
    """TODO"""


class CSVDetectionReader(_DictDetectionReader, _CSVReader):
    """A simple detection reader for csv files of detections.

    CSV file must have headers, as these are used to determine which fields to use to generate
    the detection. Detections at the same time are yielded together, and such assume file is in
    time order.

    Parameters
    ----------
    """
