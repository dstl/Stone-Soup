
from collections.abc import Iterator

try:
    import pandas as pd
except ImportError as error:
    error_msg = "Pandas Readers require dependency 'pandas' being installed. "
    raise ImportError(error_msg) from error

from ..base import Property
from .generic import _DictDetectionReader, _DictGroundTruthReader, _DictReader, _DictTrackReader


class _DataFrameReader(_DictReader):
    dataframe: pd.DataFrame = Property(doc="DataFrame containing the state data.")

    @property
    def dict_reader(self) -> Iterator[dict]:
        yield from self.dataframe.to_dict(orient="records")


class DataFrameGroundTruthReader(_DictGroundTruthReader, _DataFrameReader):
    """A custom reader for pandas DataFrames containing truth data.

    The DataFrame must have columns containing all fields needed to generate the
    ground truth state. Those states with the same ID will be put into
    a :class:`~.GroundTruthPath` in sequence. All paths that are updated at the same time
    are yielded together. Assume DataFrame is in time order.

    Parameters
    ----------
    """


class DataFrameDetectionReader(_DictDetectionReader, _DataFrameReader):
    """A custom detection reader for DataFrames containing detections.

    The DataFrame must have columns containing all fields needed to generate the detection.
    Detections at the same time are yielded together. Assume DataFrame is in time order.

    Parameters
    ----------
    """


class DataFrameTrackReader(_DictTrackReader, _DataFrameReader):
    """A :class:`~.TrackReader` class for reading in :class:`~.Track` from
    a pandas DataFrame.

    The DataFrame must have columns containing all fields needed to generate the
    track states. Those states with the same ID will be put into
    a :class:`~.Track` in sequence. All paths that are updated at the same time
    are yielded together. Assume DataFrame is in time order.

    Parameters
    ----------
    """
