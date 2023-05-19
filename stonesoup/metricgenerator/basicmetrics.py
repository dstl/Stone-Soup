from .base import MetricGenerator
from ..types.metric import TimeRangeMetric
from ..types.time import TimeRange

from ..base import Property


class BasicMetrics(MetricGenerator):
    """Calculates simple metrics like number of tracks, truth and
    ratio of track-to-truth"""
    generator_name: str = Property(doc="Unique identifier to use when accessing generated metrics from MultiManager")
    tracks_key: str = Property(doc='Key to access set of tracks added to MultiManager')
    truths_key: str = Property(doc="Key to access set of ground truths added to MultiManager. Or key to access a second"
                                   " set of tracks for track-to-track metric generation")

    def compute_metric(self, manager, *args, **kwargs):
        """Compute the metric using the data in the metric manager

        Parameters
        ----------
        manager : MetricManager
            containing the data to be used to create the metric(s)

        Returns
        ----------
        : list of :class:`~.Metric`
            Contains the metric information
        """
        tracks = self._get_data(manager, self.tracks_key)
        track_or_truth = self._get_data(manager, self.truths_key)

        metrics = []

        # Make a list of all the unique timestamps used
        timestamps = {state.timestamp for state in tracks}
        timestamps |= {state.timestamp
                       for path in track_or_truth
                       for state in path}

        # Number of tracks
        metrics.append(TimeRangeMetric(
            title='Number of targets',
            value=len(track_or_truth),
            time_range=TimeRange(
                start_timestamp=min(timestamps),
                end_timestamp=max(timestamps)),
            generator=self))

        metrics.append(TimeRangeMetric(
            title='Number of tracks',
            value=len(tracks),
            time_range=TimeRange(
                start_timestamp=min(timestamps),
                end_timestamp=max(timestamps)),
            generator=self))

        metrics.append(TimeRangeMetric(
            title='Track-to-target ratio',
            value=len(tracks) / len(track_or_truth),
            time_range=TimeRange(
                start_timestamp=min(timestamps),
                end_timestamp=max(timestamps)),
            generator=self))

        return metrics
