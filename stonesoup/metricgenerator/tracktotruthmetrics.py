# -*- coding: utf-8 -*-
import datetime
from operator import attrgetter

import numpy as np

from .base import MetricGenerator
from ..types.metric import SingleTimeMetric, TimeRangeMetric
from ..types.time import TimeRange


class SIAPMetrics(MetricGenerator):
    """Computes the SIAP metrics
    """

    def compute_metric(self, manager, *args, **kwargs):
        """Compute metric

        Parameters
        ----------
        manager : MetricManager
            containing the data to be used to create the metric(s)

        Returns
        -------
        : list of :class:`~.Metric` objects
            Generated metrics
        """
        metrics = [self.C_time_range(manager),
                   self.A_time_range(manager),
                   self.S_time_range(manager),
                   self.LT(manager),
                   self.LS(manager)]
        return metrics

    def C_single_time(self, manager, timestamp):
        """SIAP metric C at a time

        Number of truth tracked at timestamp / number of truth at timestamp

        Parameters
        ----------
        manager : MetricManager
            containing the data to be used to create the metric(s)
        timestamp: datetime.datetime
            timestamp at which to compute the metric

        Returns
        -------
        SingleTimeMetric
            Contains the metric information
        """

        live_truth = [t for t in manager.groundtruth_paths if
                      timestamp in [s.timestamp for s in t.states]]
        n_associations = 0
        for ltruth in live_truth:
            n_associations += sum(
                1
                for assoc in manager.association_set.associations
                if ltruth in assoc.objects and timestamp in assoc.time_range)

        return SingleTimeMetric(title="SIAP C at timestamp",
                                value=n_associations / len(live_truth),
                                timestamp=timestamp,
                                generator=self)

    def C_time_range(self, manager):
        """SIAP metric C over time

        Sum(number of truth tracked at timestamp)/sum(number of truth
         at timestamp). Both over all timestamps

        Parameters
        ----------
        manager : MetricManager
            Containing the data to be used to create the metric(s)

        Returns
        -------
        TimeRangeMetric
            Contains the metric information

        """

        timestamps = manager.list_timestamps()
        C = self._jt_sum(manager, timestamps) / self._j_sum(
            manager, timestamps)
        return TimeRangeMetric(
            title="SIAP C",
            value=C,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def A_time_range(self, manager):
        """SIAP metric A over time

        Number of assigned tracks / number of truth objects

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric

        Returns
        -------
        TimeRangeMetric
            Contains the metric information
        """

        timestamps = manager.list_timestamps()
        A = self._na_sum(manager, timestamps) / self._jt_sum(
            manager, timestamps)
        return TimeRangeMetric(
            title="SIAP A",
            value=A,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def S_time_range(self, manager):
        """SIAP metric S over time

        Average percentage of tracks that are spurious (unassigned)

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric

        Returns
        -------
        TimeRangeMetric
            Contains the metric information
        """
        timestamps = manager.list_timestamps()
        numerator = sum(
            self._n_t(manager, timestamp) - self._na_t(manager, timestamp)
            for timestamp in timestamps)

        S = numerator / self._n_sum(manager, timestamps)
        return TimeRangeMetric(
            title="SIAP S",
            value=S,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def LT(self, manager):
        """SIAP metric LT over time

        1/ Average number of excess trackes

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric

        Returns
        -------
        TimeRangeMetric
            Contains the metric information
        """

        r = self._r(manager)
        if r == 0:
            value = np.inf
        else:
            value = 1 / r

        timestamps = manager.list_timestamps()
        return TimeRangeMetric(
            title="SIAP LT",
            value=value,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def LS(self, manager):
        """SIAP metric LS over time

        Percentage of truth successfully tracked

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric

        Returns
        -------
        TimeRangeMetric
            Contains the metric information
        """
        numerator = sum(self._tl_j(manager, truth).total_seconds()
                        for truth in manager.groundtruth_paths)
        denominator = sum(self._t_j(truth).total_seconds()
                          for truth in manager.groundtruth_paths)

        timestamps = manager.list_timestamps()
        return TimeRangeMetric(
            title="SIAP LS",
            value=numerator / denominator,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def _j_t(self, manager, timestamp):
        """Number of truth objects at timestamp

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric
        timestamp: datetime.datetime
            Timestamp at which to compute the metric

        Returns
        -------
        int
            Number of truth objects in manager with a state at timestamp
        """
        return sum(
            1
            for path in manager.groundtruth_paths
            if timestamp in (state.timestamp for state in path))

    def _j_sum(self, manager, timestamps):
        """Sum number of truth objects over all timestamps

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric
        timestamps: iterable of :class:`datetime.datetime`
            Timestamps over which to compute the metric

        Returns
        -------
        int
            Sum number of truth objects over all timestamps
        """

        return sum(self._j_t(manager, timestamp) for timestamp in timestamps)

    def _jt_t(self, manager, timestamp):
        """Number of truth objects being tracked at time timestamp

        Parameters
        ----------
        manager: MetricManager
            Contains the data to be used to create the metric
        timestamp: datetime.datetime
            Timestamp over which to compute the metric

        Returns
        -------
        int
            Number of truth objects with states at timestamp
        """

        return sum(
            1
            for assoc in manager.association_set.associations
            if timestamp in assoc.time_range)

    def _jt_sum(self, manager, timestamps):
        """Sum number of truth objects being tracked at a given timestamp
        over list of timestamps

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric
        timestamps: iterable of :class:`datetime.datetime`
            Timestamps over which to compute the metric

        Returns
        -------
        int
            total number of truth objects

        """

        return sum(self._jt_t(manager, timestamp)
                   for timestamp in timestamps)

    def _na_t(self, manager, timestamp):
        """Number of associated tracks at a timestamp

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric
        timestamp: datetime.datetime
            Timestamp over which to compute the metric

        Returns
        -------
        int
            Number of associated tracks
        """

        assocs = manager.association_set.associations_at_timestamp(timestamp)
        n_associated_tracks = 0
        for track in manager.tracks:
            for assoc in assocs:
                if track in assoc.objects:
                    n_associated_tracks += 1
                    break
        return n_associated_tracks

    def _na_sum(self, manager, timestamps):
        """Sum of number of associated tracks at a timestamp over list of
        timestamps

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric
        timestamps: iterable of :class:`datetime.datetime`
            Timestamps over which to compute the metric

        Returns
        -------
        int
            Sum of the number of associated tracks
        """

        return sum(self._na_t(manager, timestamp) for timestamp in timestamps)

    def _n_t(self, manager, timestamp):
        """Number of tracks at timestamp

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric
        timestamp: datetime.datetime
            Timestamp over which to compute the metric

        Returns
        -------
        int
            Number of tracks
        """

        return sum(
            1
            for track in manager.tracks
            if timestamp in (state.timestamp for state in track.states))

    def _n_sum(self, manager, timestamps):
        """Sum of number of tracks over timestamps

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric
        timestamps: iterable of :class:`datetime.datetime`
            Timestamps over which to compute the metric

        Returns
        -------
        int
            Sum number of tracks
        """

        return sum(self._n_t(manager, timestamp) for timestamp in timestamps)

    def _tt_j(self, manager, truth):
        """Total time that object is tracked for

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric
        truth: GroundTruthPath
            Truth object to compute the metric for

        Returns
        -------
        datetime.timedelta
            The duration that an object is tracked for
        """

        assocs = manager.association_set.associations_including_objects(
            [truth])
        timestamps = sorted(s.timestamp for s in truth)
        total_time = datetime.timedelta(0)
        for i_timestamp, timestamp in enumerate(timestamps[:-1]):
            for assoc in assocs:
                # If both timestamps are in one association then add the
                # difference to the total difference and stop looking
                if timestamp in assoc.time_range \
                        and timestamps[i_timestamp + 1] in assoc.time_range:
                    total_time += (timestamps[i_timestamp + 1] - timestamp)
                    break

        return total_time

    def _nu_j(self, manager, truth):
        """Minimum number of tracks needed to track truth over the timestamps
        that truth is tracked for

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric
        truth: GroundTruthPath
            Truth object to compute the metric for

        Returns
        -------
        int
            Minimum number of tracks needed to track truth over the timestamps
            that truth is tracked for

        """

        # Starting at the beginning of the truth find the track associated at
        # that timestamp with the longest length, increase the track count by
        # one and move time to the end of that track. Repeat until the end of
        # the truth is reached. If no tracks present at a point then move on to
        # the next timestamp in the truth.

        truth_timestamps = sorted(i.timestamp for i in truth.states)
        assocs = manager.association_set.associations_including_objects(
            [truth])
        n_truth_needed = 0
        current_time = truth_timestamps[0]

        while current_time < truth_timestamps[-1]:
            assocs_at_time = sorted(
                (assoc
                 for assoc in assocs
                 if current_time in assoc.time_range),
                key=attrgetter('time_range.end_timestamp'),
                reverse=True)
            if not assocs_at_time:
                i_timestamp = truth_timestamps.index(current_time)
                current_time = truth_timestamps[i_timestamp + 1]
            else:
                current_time = assocs_at_time[0].time_range.end_timestamp
                n_truth_needed += 1

                # If not yet at the end of the truth timestamps then move on
                # to the next one
                if current_time < truth_timestamps[-1]:
                    i_timestamp = truth_timestamps.index(current_time)
                    current_time = truth_timestamps[i_timestamp + 1]

        return n_truth_needed

    def _tl_j(self, manager, truth):
        """Total time of the longest track on the truth

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric
        truth: GroundTruthPath
            Truth object to compute the metric for

        Returns
        -------
        datetime.timedelta
            The length of the longest track
        """

        assocs = manager.association_set.associations_including_objects(
            [truth])
        if not assocs:
            return datetime.timedelta(0)
        else:
            return max(assoc.time_range.duration for assoc in assocs)

    def _r(self, manager):
        """Average number of excess tracks assigned

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric

        Returns
        -------
        int
            Average number of excess tracks assigned
        """

        numerator = sum(self._nu_j(manager, truth) - 1
                        for truth in manager.groundtruth_paths)
        denominator = sum(self._tt_j(manager, truth).total_seconds()
                          for truth in manager.groundtruth_paths)
        return numerator / denominator

    def _t_j(self, truth):
        """Total time truth exists for

        Parameters
        ----------
        truth: GroundTruthPath
            Truth object to compute the metric for

        Returns
        -------
        datetime.timedelta
            The time the truth object exists for
        """

        timestamps = [s.timestamp for s in truth.states]
        return max(timestamps) - min(timestamps)
