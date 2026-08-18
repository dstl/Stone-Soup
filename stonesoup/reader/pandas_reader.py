
from collections.abc import Iterable, Iterator

try:
    import pandas as pd
except ImportError as error:
    error_msg = "Pandas Readers require dependency 'pandas' being installed. "
    raise ImportError(error_msg) from error

from ..base import Property
from .generic import _DictDetectionReader, _DictGroundTruthReader, _DictReader, _DictTrackReader


class _DataFrameReader(_DictReader):
    dataframe: pd.DataFrame | Iterable[pd.DataFrame] = Property(
        doc="DataFrame containing the state data, or an iterable yielding DataFrames in time "
            "order."
    )

    @property
    def dict_reader(self) -> Iterator[dict]:
        if isinstance(self.dataframe, pd.DataFrame):
            dataframes = (self.dataframe,)
        else:
            dataframes = self.dataframe

        for dataframe in dataframes:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(
                    "dataframe must be a pandas DataFrame or an iterable yielding DataFrames")
            yield from dataframe.to_dict(orient="records")


class DataFrameGroundTruthReader(_DictGroundTruthReader, _DataFrameReader):
    """A custom reader for pandas DataFrames containing truth data.

    The input may be a single DataFrame or an iterable yielding DataFrames. Each DataFrame must
    have columns containing all fields needed to generate the ground truth state. When an iterable
    is supplied, its DataFrames must collectively be in time order. States with the same ID are
    put into a :class:`~.GroundTruthPath` in sequence, and all paths updated at the same time are
    yielded together.

    Parameters
    ----------
    """


class DataFrameDetectionReader(_DictDetectionReader, _DataFrameReader):
    """A custom detection reader for DataFrames containing detections.

    The input may be a single DataFrame or an iterable yielding DataFrames. Each DataFrame must
    have columns containing all fields needed to generate the detections. When an iterable is
    supplied, its DataFrames must collectively be in time order. Detections at the same time are
    yielded together.

    Parameters
    ----------
    """


class DataFrameTrackReader(_DictTrackReader, _DataFrameReader):
    """A :class:`~.TrackReader` class for reading in :class:`~.Track` from
    pandas DataFrames.

    The input may be a single DataFrame or an iterable yielding DataFrames. Each DataFrame must
    have columns containing all fields needed to generate the track states. When an iterable is
    supplied, its DataFrames must collectively be in time order. States with the same ID are put
    into a :class:`~.Track` in sequence, and all tracks updated at the same time are yielded
    together.

    Parameters
    ----------
    """
