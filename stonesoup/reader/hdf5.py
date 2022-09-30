"""Generic HDF5 readers for Stone Soup.

This is a collection of generic readers for Stone Soup, allowing quick reading
of data that is in `HDF5 <https://hdfgroup.org/>`_ format, using the `h5py
<https://docs.h5py.org/>`_ library.
"""

from datetime import datetime, timedelta
from typing import Collection, Sequence

try:
    import h5py
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "HDF5 Readers require the dependency 'h5py' to be installed."
    ) from error
import numpy as np
from dateutil.parser import parse

from .base import GroundTruthReader, DetectionReader
from .file import BinaryFileReader
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.detection import Detection
from ..types.groundtruth import GroundTruthPath, GroundTruthState


class _HDF5Reader(BinaryFileReader):
    state_vector_fields: Sequence[str] = Property(
        doc="Paths of datasets to be used in state vector"
    )
    time_field: str = Property(doc="Path of dataset to be used as time field")
    time_field_format: str = Property(default=None, doc="Optional datetime format")
    timestamp: bool = Property(
        default=False, doc="Treat time field as a timestamp from epoch"
    )
    time_res_second: int = Property(
        default=1, doc="Desired maximum resolution of time values in seconds",
    )
    time_res_micro: int = Property(
        default=1e6,
        doc="Desired maximum sub-second resolution of time values in microseconds",
    )
    metadata_fields: Collection[str] = Property(
        default=None, doc="Paths of datasets to be saved as metadata, default all"
    )

    def _discover_metadata_fields(self, hdf5_file):
        """Recurse through all objects in a file and treat any dataset with
        the same number of records as a valid metadata field path, excluding
        datasets that are already specified for state values.

        Parameters
        ----------
        hdf5_file : :class:`h5py.File`
            The HDF5 file to walk through
        """

        self.metadata_fields = []
        record_count = len(hdf5_file[self.time_field])
        obj_paths = []
        hdf5_file.visit(obj_paths.append)

        for obj_path in obj_paths:
            obj = hdf5_file[obj_path]
            if isinstance(obj, h5py.Dataset):
                if (
                    obj_path not in self.state_vector_fields
                    and obj_path != self.time_field
                    and len(obj) == record_count
                ):
                    self.metadata_fields.append(obj_path)

    def _get_metadata(self, hdf5_file, row):
        """Construct a dictionary of metadata values for a single record.

        Parameters
        ----------
        hdf5_file : :class:`h5py.File`
            The HDF5 file to read from
        row : int
            The row index of the record

        Returns
        -------
        : dict
            The metadata values for the record
        """
        if self.metadata_fields is None:
            self._discover_metadata_fields(hdf5_file)

        local_metadata = {
            **{
                field: hdf5_file[field][row]
                for field in self.metadata_fields
                if field in hdf5_file
                and h5py.check_string_dtype(hdf5_file[field].dtype) is None
            },  # Merge string and non-string fields into the same dict
            **{
                field: hdf5_file[field].asstr()[row]
                for field in self.metadata_fields
                if field in hdf5_file
                and h5py.check_string_dtype(hdf5_file[field].dtype) is not None
            },
        }

        return local_metadata

    def _get_time(self, raw_time_val):
        """Interpret a time value as a datetime object.

        Parameters
        ----------
        raw_time_val : str or float or int
            A formatted time string, or a POSIX timestamp to convert

        Returns
        -------
        : :class:`datetime.datetime`
            The parsed time value
        """
        if self.time_field_format is not None:
            time_field_value = datetime.strptime(raw_time_val, self.time_field_format)
        elif self.timestamp is True:
            time_field_value = datetime.utcfromtimestamp(raw_time_val)
        else:
            time_field_value = parse(raw_time_val, ignoretz=True)

        # Reduce timing resolution, as applicable
        time_field_value = time_field_value - timedelta(
            seconds=time_field_value.second % self.time_res_second,
            microseconds=time_field_value.microsecond % self.time_res_micro,
        )
        return time_field_value


class HDF5GroundTruthReader(GroundTruthReader, _HDF5Reader):
    """A simple reader for HDF5 files of truth data.

    HDF5 files are hierarchically structured data files with embedded metadata. This
    reader will extract values that are placed anywhere within the hierarchy, but it
    assumes all datasets are 1D arrays of base types representing 'columns' of data.
    All fields must be the same length, and a 'row' of data is constructed from the
    values at the same positional index in each column. Those states with the same ID
    will be put into a :class:`~.GroundTruthPath` in sequence, and all paths that are
    updated at the same time are yielded together, and such assumes file is in time
    order.

    Parameters
    ----------
    """

    path_id_field: str = Property(doc="Path of dataset to be used as path ID")

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        with h5py.File(self.path, "r") as hdf5_file:
            groundtruth_dict = {}
            updated_paths = set()
            previous_time = None

            time_values = hdf5_file[self.time_field]
            if not self.timestamp:
                time_values = time_values.asstr()

            for i, raw_time_val in enumerate(time_values):

                time = self._get_time(raw_time_val)
                if previous_time is not None and previous_time != time:
                    yield previous_time, updated_paths
                    updated_paths = set()
                previous_time = time

                state = GroundTruthState(
                    np.array(
                        [
                            [hdf5_file[field_path][i]]
                            for field_path in self.state_vector_fields
                        ],
                        dtype=np.float64,
                    ),
                    timestamp=time,
                    metadata=self._get_metadata(hdf5_file, i),
                )

                id_ = hdf5_file[self.path_id_field][i]
                if id_ not in groundtruth_dict:
                    groundtruth_dict[id_] = GroundTruthPath(id=id_)
                groundtruth_path = groundtruth_dict[id_]
                groundtruth_path.append(state)
                updated_paths.add(groundtruth_path)

            # Yield remaining
            yield previous_time, updated_paths


class HDF5DetectionReader(DetectionReader, _HDF5Reader):
    """A simple detection reader for HDF5 files of detections.

    HDF5 files are hierarchically structured data files with embedded metadata. This
    reader will extract values that are placed anywhere within the hierarchy, but it
    assumes all datasets are 1D arrays of base types representing 'columns' of data.
    All fields must be the same length, and a 'row' of data is constructed from the
    values at the same positional index in each column. Detections at the same time
    are yielded together, and such assume file is in time order.

    Parameters
    ----------
    """

    @BufferedGenerator.generator_method
    def detections_gen(self):
        with h5py.File(self.path, "r") as hdf5_file:
            detections = set()
            previous_time = None

            time_values = hdf5_file[self.time_field]
            if not self.timestamp:
                time_values = time_values.asstr()

            for i, raw_time_val in enumerate(time_values):

                time = self._get_time(raw_time_val)
                if previous_time is not None and previous_time != time:
                    yield previous_time, detections
                    detections = set()
                previous_time = time

                detections.add(
                    Detection(
                        np.array(
                            [
                                [hdf5_file[field_path][i]]
                                for field_path in self.state_vector_fields
                            ],
                            dtype=np.float64,
                        ),
                        timestamp=time,
                        metadata=self._get_metadata(hdf5_file, i),
                    )
                )

            # Yield remaining
            yield previous_time, detections
