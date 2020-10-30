# -*- coding: utf-8 -*-
import datetime
import warnings
from operator import attrgetter

import numpy as np

from .base import MetricGenerator
from ..base import Property
from ..measures import EuclideanWeighted
from ..types.metric import SingleTimeMetric, TimeRangeMetric
from ..types.time import TimeRange


class SIAPMetrics(MetricGenerator):
    """SIAP Metrics

    Computes the Single Integrated Air Picture (SIAP) metrics as defined by the Systems Engineering
    Task Force. The implementation provided here is derived from [1] and focuses on providing the
    SIAP attribute measures.

    The SIAP metrics provided require provision of ground truth information.

    In the original paper the calculations are dependent upon :math:`m` which corresponds to the
    identifying number of the sense capability which is being assessed. This is not used in this
    implementation, with the assumption being that the fused sensor set is being assessed.

    Reference
        [1] Single Integrated Air Picture (SIAP) Metrics Implementation,
        Votruba et al, 29-10-2001
    """

    position_weighting: np.ndarray = Property(default=None,
                                              doc="Weighting(s) to be used by euclidean measure in"
                                                  "position kinematic accuracy calculations. If"
                                                  "None, weights are all 1")
    velocity_weighting: np.ndarray = Property(default=None,
                                              doc="Weighting(s) to be used by euclidean measure in"
                                                  "velocity kinematic accuracy calculations. If"
                                                  "None, weights are all 1")
    position_mapping: np.ndarray = Property(default=None,
                                            doc="Mapping array which specifies which elements"
                                                "within state space state vectors correspond to"
                                                "position")
    velocity_mapping: np.ndarray = Property(default=None,
                                            doc="Mapping array which specifies which elements"
                                                "within state space state vectors correspond to"
                                                "velocity")

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
        C = self.C_time_range(manager)
        A = self.A_time_range(manager)
        S = self.S_time_range(manager)
        LT = self.LT(manager)
        LS = self.LS(manager)
        nt = self.num_tracks(manager)
        nj = self.num_truths(manager)

        metrics = [C, A, S, LT, LS, nt, nj]

        timestamped_metrics = {'T C': [], 'T A': [], 'T S': []}
        timestamps = manager.list_timestamps()
        for timestamp in timestamps:
            timestamped_metrics['T C'].append(self.C_single_time(manager, timestamp))
            timestamped_metrics['T A'].append(self.A_single_time(manager, timestamp))
            timestamped_metrics['T S'].append(self.S_single_time(manager, timestamp))
        t_metrics = [TimeRangeMetric(title=key,
                                     value=value,
                                     time_range=TimeRange(min(timestamps), max(timestamps)),
                                     generator=self)
                     for key, value in timestamped_metrics.items()]

        if self.position_mapping is not None:
            PA = self.PA(manager)
            metrics.append(PA)
            t_PA = []
            for timestamp in timestamps:
                t_PA.append(self.PA_single_time(manager, timestamp))
            metrics.append(TimeRangeMetric(title='T PA',
                                           value=t_PA,
                                           time_range=TimeRange(min(timestamps), max(timestamps)),
                                           generator=self))

        if self.velocity_mapping is not None:
            VA = self.VA(manager)
            metrics.append(VA)
            t_VA = []
            for timestamp in timestamps:
                t_VA.append(self.VA_single_time(manager, timestamp))
            metrics.append(TimeRangeMetric(title='T VA',
                                           value=t_VA,
                                           time_range=TimeRange(min(timestamps), max(timestamps)),
                                           generator=self))

        metrics.extend(t_metrics)
        return metrics

    @staticmethod
    def _warn_no_truth(manager):
        if len(manager.groundtruth_paths) == 0:
            warnings.warn("No truth to generate SIAP Metric", stacklevel=2)

    @staticmethod
    def _warn_no_tracks(manager):
        if len(manager.tracks) == 0:
            warnings.warn("No tracks to generate SIAP Metric", stacklevel=2)

    def C_single_time(self, manager, timestamp):
        r"""SIAP metric C at a specific time

        Returns an assessment of the number of targets currently being tracked compared to the
        number of true targets at a specific timestamp, :math:`{t}`. The output is a percentage,
        range 0:1, with a target score of 1

        .. math::

              C_{t} = \frac{J{T_m}({t})}{J({t})}

        where
            :math:`J{T_m}({t})` is the number of objects being tracked at timestamp :math:`{t}`
        and
            :math:`J({t})` is the number of true objects at timestamp :math:`{t}`.

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

        numerator = self._jt_t(manager, timestamp)
        try:
            C = numerator / self._j_t(manager, timestamp)
        except ZeroDivisionError:
            C = 0

        return SingleTimeMetric(title="SIAP C at timestamp",
                                value=C,
                                timestamp=timestamp,
                                generator=self)

    def C_time_range(self, manager):
        r"""SIAP metric C over time

        Returns an assessment of the number of targets currently being tracked compared to the
        number of true targets over the time range of the dataset. The output is a percentage,
        range :math:`0:1`, with a score of 1

        .. math::

              C = \frac{\sum_{t_{start}}^{t_{end}}J{T_m}({t})}
              {\sum_{t_{start}}^{t_{end}}J({t})}

        where
            :math:`J{T_m}({t})` is the number of objects being tracked at
            timestamp :math:`{t}`
        and
            :math:`J({t})` is the number of true objects at timestamp
            :math:`{t}`.

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
        try:
            C = self._jt_sum(manager, timestamps) / self._j_sum(
                manager, timestamps)
        except ZeroDivisionError:
            self._warn_no_truth(manager)
            C = 0
        return TimeRangeMetric(
            title="SIAP C",
            value=C,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def A_single_time(self, manager, timestamp):
        r"""SIAP metric A at a specific time

        Returns an assessment of the number of tracks which are assigned to true objects against
        the total number of tracks, at a specific timestamp, :math:`{t}`. The output is a
        percentage, range :math:`0:\infty`, with a target score of 1

        .. math::

              A_{t} = \frac{N{A}({t})}{J{T_m}({t})}

        where
            :math:`N{A}({t})` is the number of tracks assigned to true objects at timestamp
            :math:`{t}`
        and
            :math:`J{T_m}({t})` is the number of objects being tracked at timestamp :math:`{t}`.

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
        try:
            A = self._na_t(manager, timestamp) / self._jt_t(manager, timestamp)
        except ZeroDivisionError:
            A = 1
        return SingleTimeMetric(title="SIAP A at timestamp",
                                value=A,
                                timestamp=timestamp,
                                generator=self)

    def A_time_range(self, manager):
        r"""SIAP metric A over time

        Returns a percentage value which assesses the number of tracks which
        are assigned to true objects against the total number of tracks. The
        target score is 1.

        .. math::

              A = \frac{\sum_{t_{start}}^{t_{end}}N{A}({t})}
              {\sum_{t_{start}}^{t_{end}}J{T_m}({t})}

        where
            :math:`N{A}({t})` is the number of tracks assigned to true objects at timestamp
            :math:`{t}`
        and
            :math:`J{T_m}({t})` is the number of objects being tracked at timestamp :math:`{t}`.

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
        try:
            A = self._na_sum(manager, timestamps) / self._jt_sum(manager, timestamps)
        except ZeroDivisionError:
            self._warn_no_truth(manager)
            self._warn_no_tracks(manager)
            A = 1
        return TimeRangeMetric(
            title="SIAP A",
            value=A,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def S_single_time(self, manager, timestamp):
        r"""SIAP metric S at a specific time

        Returns an assessment of the number of tracks that are deemed to be spurious, i.e.
        unassigned to true objects, at a specific timestamp, :math:`{t}`. The output is a
        percentage, range :math:`0:\infty`, with a target score of 0

        .. math::

              S_{t} = \frac{N({t}) - N{A}({t})}{N({t})}

        where
            :math:`N{A}({t})` is the number of tracks assigned to true objects at timestamp
            :math:`{t}`
        and
            :math:`N({t})` is the number of tracks timestamp :math:`{t}`.

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
        numerator = self._n_t(manager, timestamp) - self._na_t(manager, timestamp)
        try:
            S = numerator / self._n_t(manager, timestamp)
        except ZeroDivisionError:
            S = 0
        return SingleTimeMetric(title="SIAP S at timestamp",
                                value=S,
                                timestamp=timestamp,
                                generator=self)

    def S_time_range(self, manager):
        r"""SIAP metric S over time

        The average percentage of tracks that are deemed to be spurious, i.e. unassigned to true
        objects.

        .. math::

              S = \frac{\sum_{t_{start}}^{t_{end}}[N({t}) - N{A}({t})]}
              {\sum_{t_{start}}^{t_{end}}N({t})}

        where
            :math:`N{A}({t})` is the number of tracks assigned to true objects at timestamp
            :math:`{t}`
        and
            :math:`N({t})` is the number of tracks timestamp :math:`{t}`.

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
        numerator = self._n_sum(manager, timestamps) - self._na_sum(manager, timestamps)
        try:
            S = numerator / self._n_sum(manager, timestamps)
        except ZeroDivisionError:
            self._warn_no_tracks(manager)
            S = 0
        return TimeRangeMetric(
            title="SIAP S",
            value=S,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def LT(self, manager):
        r"""SIAP metric LT over time

        Returns :math:`1/{R}` where :math:`{R}` is the average number of excess tracks assigned.
        Target score is :math:`LT = \infty`

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
            self._warn_no_truth(manager)
            self._warn_no_tracks(manager)
        else:
            value = 1 / r

        timestamps = manager.list_timestamps()
        return TimeRangeMetric(
            title="SIAP LT",
            value=value,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def LS(self, manager):
        r"""SIAP metric LS over time

        Returns the percentage of time that true objects have been tracked across the dataset

        .. math::

            LS = \frac{\sum_{j=1}^{J}T{L}_{j}}{\sum_{j=1}^{J}T_{j}}

        where
            :math:`\sum_{j=1}^{J}T{L}_{j}` is the total time of the longest track on object
            :math:`j`.
        and
            :math:`\sum_{j=1}^{J}T_{j}` is the total duration of true object :math:`j`.

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
        try:
            LS = numerator / denominator
        except ZeroDivisionError:
            self._warn_no_truth(manager)
            LS = 0
        return TimeRangeMetric(
            title="SIAP LS",
            value=LS,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def PA_single_time(self, manager, timestamp):
        r"""SIAP metric PA at a specific time

        Returns an assessment of the average assigned track positional accuracy at a specific
        timestamp, :math:`{t}`. The output is a distance measure, range :math:`0:\infty`, with a
        target score of 0

        .. math::

              PA_{t} = \frac{{\sum_{n\in D(t)}PA_{n}(t)}}{NA(t)}

        where
            :math:`D(t)` is the set of tracks held at timestamp :math:`t` :math:`PA_{n}(t)` is the
            Euclidean distance of track n to its associated truth at timestamp :math:`{t}`
        and
            :math:`N{A}({t})` is the number of tracks assigned to true objects at timestamp
            :math:`{t}`.

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
        numerator = self._assoc_distances_sum_t(manager,
                                                timestamp,
                                                self.position_mapping,
                                                self.position_weighting)
        try:
            PA = numerator / self._na_t(manager, timestamp)
        except ZeroDivisionError:
            PA = 0
        return SingleTimeMetric(title="SIAP PA at timestamp",
                                value=PA,
                                timestamp=timestamp,
                                generator=self)

    def PA(self, manager):
        r"""SIAP metric PA over time

        The average positional accuracy of associated tracks.

        .. math::

              PA = \frac{\sum_{t_{start}}^{t_{end}}{\sum_{n\in D(t)}PA_{n}(t)}}
                        {\sum_{t_{start}}^{t_{end}}{NA(t)}}

        where
            :math:`D(t)` is the set of tracks held at timestamp :math:`t` :math:`PA_{n}(t)` is the
            Euclidean distance of track n to its associated truth at timestamp :math:`{t}`
        and
            :math:`N{A}({t})` is the number of tracks assigned to true objects at timestamp
            :math:`{t}`.

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
        numerator = sum(self._assoc_distances_sum_t(manager,
                                                    timestamp,
                                                    self.position_mapping,
                                                    self.position_weighting)
                        for timestamp in timestamps)
        try:
            PA = numerator / self._na_sum(manager, timestamps)
        except ZeroDivisionError:
            PA = 0
        return TimeRangeMetric(
            title="SIAP PA",
            value=PA,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def VA_single_time(self, manager, timestamp):
        r"""SIAP metric VA at a specific time

        Returns an assessment of the average assigned track velocity accuracy at a specific
        timestamp, :math:`{t}`. The output is a distance measure, range :math:`0:\infty`, with a
        target score of 0

        .. math::

              VA_{t} = \frac{{\sum_{n\in D(t)}VA_{n}(t)}}{NA(t)}

        where
            :math:`D(t)` is the set of tracks held at timestamp :math:`t`, :math:`VA_{n}(t)` is the
            Euclidean distance of track n's velocity components to its associated truth's
            corresponding velocities at timestamp :math:`{t}`
        and
            :math:`N{A}({t})` is the number of tracks assigned to true objects at timestamp
            :math:`{t}`.

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
        numerator = self._assoc_distances_sum_t(manager,
                                                timestamp,
                                                self.velocity_mapping,
                                                self.velocity_weighting)
        try:
            VA = numerator / self._na_t(manager, timestamp)
        except ZeroDivisionError:
            VA = 0
        return SingleTimeMetric(title="SIAP VA at timestamp",
                                value=VA,
                                timestamp=timestamp,
                                generator=self)

    def VA(self, manager):
        r"""SIAP metric VA over time

        The average velocity accuracy of associated tracks.

        .. math::

              VA = \frac{\sum_{t_{start}}^{t_{end}}{\sum_{n\in D(t)}VA_{n}(t)}}
                        {\sum_{t_{start}}^{t_{end}}{NA(t)}}

        where
            :math:`D(t)` is the set of tracks held at timestamp :math:`t` :math:`VA_{n}(t)` is the
            Euclidean distance of track n's velocity components to its associated truth's
            corresponding velocities at timestamp :math:`{t}`
        and
            :math:`N{A}({t})` is the number of tracks assigned to true objects at timestamp
            :math:`{t}`.

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
        numerator = sum(self._assoc_distances_sum_t(manager,
                                                    timestamp,
                                                    self.velocity_mapping,
                                                    self.velocity_weighting)
                        for timestamp in timestamps)
        try:
            VA = numerator / self._na_sum(manager, timestamps)
        except ZeroDivisionError:
            VA = 0
        return TimeRangeMetric(
            title="SIAP VA",
            value=VA,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def num_tracks(self, manager):
        """Calculates the number of tracks stored in the metric manager
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
        nt = len(manager.tracks)

        return TimeRangeMetric(
            title="SIAP nt",
            value=nt,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def num_truths(self, manager):
        """Calculates the number of truths stored in the metric manager
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
        nj = len(manager.groundtruth_paths)
        return TimeRangeMetric(
            title="SIAP nj",
            value=nj,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def _assoc_distances_sum_t(self, manager, timestamp, mapping, weighting):
        """Sum of spatial (positon or velocity) distance between each truth and its associations
        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used to create the metric
        timestamp: datetime.datetime
            Timestamp at which to compute the metric
        mapping: np.ndarray
            Indices of the required positon/velocity components of the state space
        weighting: np.ndarray
            The weighting to be used by the Euclidean measure

        Returns
        -------
        int
            Sum of Euclidean distances (of position or velocity) of each truth to its associated
            tracks in manager at timestamp
        """
        measure = EuclideanWeighted(mapping=mapping, weighting=weighting)
        distance_sum = 0
        for assoc in manager.association_set.associations_at_timestamp(timestamp):
            print('num of assoc objects:', len(assoc.objects))
            track, truth = assoc.objects
            distance_sum += measure(track[timestamp], truth[timestamp])
        return distance_sum

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
        try:
            return numerator / denominator
        except ZeroDivisionError:
            # No truth or tracks
            return 0

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
