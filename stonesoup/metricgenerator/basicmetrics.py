from .base import MetricGenerator
from ..types.metric import TimeRangeMetric
from ..types.time import TimeRange

from ..base import Property


class BasicMetrics(MetricGenerator):
    """Calculates simple metrics like number of tracks, truth and
    ratio of track-to-truth"""
    tracks_key: str = Property(doc="Key to access desired set of tracks added to MultiManager")
    truths_key: str = Property(doc="Key to access desired set of groundtruths added to MultiManager")
    generator_name: str = Property(doc="Name given to generator to use when accessing generated metrics from "
                                       "MultiManager")

    def get_ground_truths(self, manager):
        return manager.states_sets[self.truths_key]

    def get_tracks(self, manager):
        return manager.states_sets[self.tracks_key]

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
        groundtruth_paths = self.get_ground_truths(manager)
        tracks = self.get_tracks(manager)
        metrics = []

        # Make a list of all the unique timestamps used
        timestamps = {state.timestamp for state in tracks}
        timestamps |= {state.timestamp
                       for path in groundtruth_paths
                       for state in path}

        # Number of tracks
        metrics.append(TimeRangeMetric(
            title='Number of targets',
            value=len(groundtruth_paths),
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
            value=len(tracks) / len(groundtruth_paths),
            time_range=TimeRange(
                start_timestamp=min(timestamps),
                end_timestamp=max(timestamps)),
            generator=self))

        return metrics
