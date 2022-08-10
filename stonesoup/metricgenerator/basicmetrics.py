from .base import MetricGenerator
from ..types.metric import TimeRangeMetric
from ..types.time import TimeRange


class BasicMetrics(MetricGenerator):
    """Calculates simple metrics like number of tracks, truth and
    ratio of track-to-truth"""

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

        metrics = []

        # Make a list of all the unique timestamps used
        timestamps = {state.timestamp for state in manager.tracks}
        timestamps |= {state.timestamp
                       for path in manager.groundtruth_paths
                       for state in path}

        # Number of tracks
        metrics.append(TimeRangeMetric(
            title='Number of targets',
            value=len(manager.groundtruth_paths),
            time_range=TimeRange(
                start_timestamp=min(timestamps),
                end_timestamp=max(timestamps)),
            generator=self))

        metrics.append(TimeRangeMetric(
            title='Number of tracks',
            value=len(manager.tracks),
            time_range=TimeRange(
                start_timestamp=min(timestamps),
                end_timestamp=max(timestamps)),
            generator=self))

        metrics.append(TimeRangeMetric(
            title='Track-to-target ratio',
            value=len(manager.tracks) / len(manager.groundtruth_paths),
            time_range=TimeRange(
                start_timestamp=min(timestamps),
                end_timestamp=max(timestamps)),
            generator=self))

        return metrics
