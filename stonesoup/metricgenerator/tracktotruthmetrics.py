from operator import attrgetter

from .base import MetricGenerator
from ..base import Property
from ..measures import Measure
from ..types.metric import SingleTimeMetric, TimeRangeMetric
from ..types.time import TimeRange
from ..types.track import Track


class SIAPMetrics(MetricGenerator):
    r"""SIAP Metrics

    Computes the Single Integrated Air Picture (SIAP) metrics as defined by the Systems Engineering
    Task Force. The implementation provided here is derived from [1] and focuses on providing the
    SIAP attribute measures.

    The SIAP metrics provided require provision of ground truth information.

    In the original paper the calculations are dependent upon :math:`m` which corresponds to the
    identifying number of the sense capability which is being assessed. This is not used in this
    implementation, with the assumption being that the fused sensor set is being assessed.

    Metrics:
        * Continuity (C): Fraction of true objects being tracked. The output is in the range
          :math:`0:1`, with a target score of 1.
        * Ambiguity (A): Number of tracks assigned to a true object. The output is unbounded with
          a range of :math:`0:\infty`. The target score is 1.
        * Spuriousness (S): Fraction of tracks that are unassigned to a true object. The output is
          in the range :math:`0:1`, with a target score of 0.
        * Positional Accuracy (PA): Positional error of associated tracks to their respective
          truths. The output is a distance measure, range :math:`0:\infty`, with a target score of
          0.
        * Velocity Accuracy (VA): Velocity error of associated tracks to their respective truths.
          The output is a distance measure, range :math:`0:\infty`, with a target score of 0.
        * Rate of track number changes (R): SIAP continuity measure. Rate of number of track
          changes per truth. The output is in the range :math:`0:\infty`, with a target score of 0.
        * Longest track Segment (LS): SIAP continuity measure. Duration of longest associated
          track segment per truth. The output is a float (seconds), with a target score equal to
          the sum of all true object lifetimes.

    Reference
        [1] Single Integrated Air Picture (SIAP) Metrics Implementation, Votruba et al, 29-10-2001
    """

    position_measure: Measure = Property(
        doc="Distance measure used in calculating position accuracy scores.")
    velocity_measure: Measure = Property(
        doc="Distance measure used in calculating velocity accuracy scores.")
    generator_name: str = Property(doc="Unique identifier to use when accessing generated metrics from MultiManager")
    tracks_key: str = Property(doc='Key to access set of tracks added to MultiManager')
    truths_key: str = Property(doc="Key to access set of ground truths added to MultiManager. Or key to access a second"
                                   " set of tracks for track-to-track metric generation")

    def compute_metric(self, manager, **kwargs):
        r"""Compute metrics:

        .. math::
            \begin{alignat*}{3}
                \textrm{Name} &\quad \textrm{At Time} &&\quad \textrm{TimeRange}\\
                C &\quad \frac{JT({t})}{J({t})} &&\quad \frac{\sum_{t_{start}}^{t_{end}}JT({t})}
                {\sum_{t_{start}}^{t_{end}}J({t})}\\
                A &\quad \frac{N{A}({t})}{JT({t})} &&\quad \frac{\sum_{t_{start}}^{t_{end}}
                N{A}({t})}{\sum_{t_{start}}^{t_{end}}JT({t})}\\
                S &\quad \frac{N({t}) - N{A}({t})}{N({t})} &&\quad \frac{\sum_{t_{start}}^{t_{end}}
                [N({t}) - N{A}({t})]}{\sum_{t_{start}}^{t_{end}}N({t})}\\
                PA &\quad \frac{{\sum_{n\in tracks}PA_{n}(t)}}{NA(t)} &&\quad
                \frac{\sum_{t_{start}}^{t_{end}}{\sum_{n\in tracks}PA_{n}(t)}}
                {\sum_{t_{start}}^{t_{end}}{NA(t)}}\\
                VA &\quad \frac{{\sum_{n\in tracks}VA_{n}(t)}}{NA(t)} &&\quad
                \frac{\sum_{t_{start}}^{t_{end}}{\sum_{n\in tracks}VA_{n}(t)}}
                {\sum_{t_{start}}^{t_{end}}{NA(t)}}\\
                R &\quad -- &&\quad \frac{\sum_{j\in truths}NU_j-1}{\sum_{j\in truths}TT_j}\\
                LS &\quad -- &&\quad \frac{\sum_{j\in truths}T{L}_{j}}{\sum_{j\in truths}T_{j}}
            \end{alignat*}

        Parameters
        ----------
        manager : MetricManager
            containing the data to be used to create the metric(s)

        Returns
        -------
        : list of :class:`~.Metric` objects
            Generated metrics
        """

        timestamps = manager.list_timestamps(generator=self)

        completeness_at_times = list()
        ambiguity_at_times = list()
        spuriousness_at_times = list()
        position_accuracy_at_times = list()
        velocity_accuracy_at_times = list()

        tracks = self._get_data(manager, self.tracks_key)
        ground_truths = self._get_data(manager, self.truths_key)

        J_sum = JT_sum = NA_sum = N_sum = PA_sum = VA_sum = 0

        for timestamp in timestamps:
            Jt = self.num_truths_at_time(ground_truths, timestamp)
            J_sum += Jt
            JTt = self.num_associated_truths_at_time(manager, ground_truths, timestamp)
            JT_sum += JTt
            NAt = self.num_associated_tracks_at_time(manager, tracks, timestamp)
            NA_sum += NAt
            Nt = self.num_tracks_at_time(tracks, timestamp)
            N_sum += Nt
            PAt = self.accuracy_at_time(manager, timestamp, self.position_measure)
            PA_sum += PAt
            VAt = self.accuracy_at_time(manager, timestamp, self.velocity_measure)
            VA_sum += VAt

            completeness_at_times.append(
                SingleTimeMetric(title="SIAP Completeness at timestamp",
                                 value=JTt / Jt if Jt != 0 else 0,
                                 timestamp=timestamp,
                                 generator=self)
            )
            ambiguity_at_times.append(
                SingleTimeMetric(title="SIAP Ambiguity at timestamp",
                                 value=NAt / JTt if JTt != 0 else 1,
                                 timestamp=timestamp,
                                 generator=self)
            )
            spuriousness_at_times.append(
                SingleTimeMetric(title="SIAP Spuriousness at timestamp",
                                 value=(Nt - NAt) / Nt if Nt != 0 else 0,
                                 timestamp=timestamp,
                                 generator=self)
            )
            position_accuracy_at_times.append(
                SingleTimeMetric(title="SIAP Position Accuracy at timestamp",
                                 value=PAt / NAt if NAt != 0 else 0,
                                 timestamp=timestamp,
                                 generator=self)
            )
            velocity_accuracy_at_times.append(
                SingleTimeMetric(title="SIAP Velocity Accuracy at timestamp",
                                 value=VAt / NAt if NAt != 0 else 0,
                                 timestamp=timestamp,
                                 generator=self)
            )

        time_range = TimeRange(min(timestamps), max(timestamps))

        completeness = TimeRangeMetric(title="SIAP Completeness",
                                       value=JT_sum / J_sum if J_sum != 0 else 0,
                                       time_range=time_range,
                                       generator=self)
        ambiguity = TimeRangeMetric(title="SIAP Ambiguity",
                                    value=NA_sum / JT_sum if JT_sum != 0 else 1,
                                    time_range=time_range,
                                    generator=self)
        spuriousness = TimeRangeMetric(title="SIAP Spuriousness",
                                       value=(N_sum - NA_sum) / N_sum if N_sum != 0 else 0,
                                       time_range=time_range,
                                       generator=self)
        position_accuracy = TimeRangeMetric(title="SIAP Position Accuracy",
                                            value=PA_sum / NA_sum if NA_sum != 0 else 0,
                                            time_range=time_range,
                                            generator=self)
        velocity_accuracy = TimeRangeMetric(title="SIAP Velocity Accuracy",
                                            value=VA_sum / NA_sum if NA_sum != 0 else 0,
                                            time_range=time_range,
                                            generator=self)
        R = self.rate_of_track_number_changes(manager, ground_truths)
        rate_track_num = TimeRangeMetric(title="SIAP Rate of Track Number Change",
                                         value=R,
                                         time_range=time_range,
                                         generator=self)
        TL_sum = sum(self.longest_track_time_on_truth(manager, truth)
                     for truth in ground_truths)
        T_sum = sum(self.truth_lifetime(truth) for truth in ground_truths)
        longest_track_seg = TimeRangeMetric(title="SIAP Longest Track Segment",
                                            value=TL_sum / T_sum if T_sum != 0 else 0,
                                            time_range=time_range,
                                            generator=self)

        completeness_at_times = TimeRangeMetric(title="SIAP Completeness at times",
                                                value=completeness_at_times,
                                                time_range=time_range,
                                                generator=self)
        ambiguity_at_times = TimeRangeMetric(title="SIAP Ambiguity at times",
                                             value=ambiguity_at_times,
                                             time_range=time_range,
                                             generator=self)
        spuriousness_at_times = TimeRangeMetric(title="SIAP Spuriousness at times",
                                                value=spuriousness_at_times,
                                                time_range=time_range,
                                                generator=self)
        position_accuracy_at_times = TimeRangeMetric(title="SIAP Position Accuracy at times",
                                                     value=position_accuracy_at_times,
                                                     time_range=time_range,
                                                     generator=self)
        velocity_accuracy_at_times = TimeRangeMetric(title="SIAP Velocity Accuracy at times",
                                                     value=velocity_accuracy_at_times,
                                                     time_range=time_range,
                                                     generator=self)

        return [completeness, ambiguity, spuriousness, position_accuracy, velocity_accuracy,
                rate_track_num, longest_track_seg, completeness_at_times, ambiguity_at_times,
                spuriousness_at_times, position_accuracy_at_times, velocity_accuracy_at_times]

    @staticmethod
    def num_truths_at_time(ground_truths, timestamp):
        """:math:`J(t)`. Calculate the number of true objects held by `manager` at `timestamp`.

        Parameters
        ----------
        ground_truths: set or list of :class:`~.GroundTruthPath` or :class:`~.Track` objects
            Containing the ground truth or track data to be used
        timestamp: datetime.datetime
            Timestamp at which to compute the value

        Returns
        -------
        float
            Number of true objects in `groundtruth` set or list at `timestamp`
        """
        return sum(
            1
            for path in ground_truths
            if timestamp in (state.timestamp for state in path))

    @staticmethod
    def num_associated_truths_at_time(manager, ground_truths, timestamp):
        """:math:`JT(t)`. Calculate the number of associated true objects held by `manager` at
        `timestamp`.

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used
        ground_truths: set or list of :class:`~.GroundTruthPath` or :class:`~.Track objects
            Containing the groundtruth or track data to be used
        timestamp: datetime.datetime
            Timestamp at which to compute the value

        Returns
        -------
        float
            Number of associated true objects held by `manager` at `timestamp`
        """
        associations = manager.association_set.associations_at_timestamp(timestamp)
        association_objects = {thing for assoc in associations for thing in assoc.objects}

        return sum(1 for truth in ground_truths if truth in association_objects)

    @staticmethod
    def num_tracks_at_time(tracks, timestamp):
        """:math:`N(t)`. Calculate the number of tracks held by `manager` at `timestamp`.

        Parameters
        ----------
        tracks: set or list of :class:`~.Track` objects
            Containing the track data to be used
        timestamp: datetime.datetime
            Timestamp at which to compute the value

        Returns
        -------
        float
            Number of tracks in `tracks` set or list at `timestamp`
        """
        return sum(
            1
            for track in tracks
            if timestamp in (state.timestamp for state in track.states))

    @staticmethod
    def num_associated_tracks_at_time(manager, tracks, timestamp):
        """:math:`NA(t)`. Calculate the number of associated tracks held by `manager` at
        `timestamp`.

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used
        tracks: set or list of :class:`~.Track`
            Containing the track data to be used
        timestamp: datetime.datetime
            Timestamp at which to compute the value

        Returns
        -------
        float
            Number of associated tracks held by `manager` at `timestamp`.
        """
        associations = manager.association_set.associations_at_timestamp(timestamp)
        association_objects = {thing for assoc in associations for thing in assoc.objects}

        return sum(1 for track in tracks if track in association_objects)

    def accuracy_at_time(self, manager, timestamp, measure):
        """:math:`PA(t)` or :math:`VA(t)` (dependent on `measure`). Calculate the kinematic
        accuracy of track-truth associations held by `manager` at `timestamp`.

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used
        timestamp: datetime.datetime
            Timestamp at which to compute the value
        measure: Measure
            Measure used to calculate 'distance' between truths and tracks

        Returns
        -------
        float
            Kinematic accuracy of track-truth associations held by `manager` at `timestamp`.

        Note
        ----
            This method adds the 'distance' errors for each and every association. An alternative
            would be to consider each true object and track at most once.
        """
        associations = manager.association_set.associations_at_timestamp(timestamp)
        error_sum = 0
        for association in associations:
            truth, track = self.truth_track_from_association(association)
            error_sum += measure(track[timestamp], truth[timestamp])
        return error_sum

    @staticmethod
    def truth_track_from_association(association):
        """Find truth and track from an association.

        Parameters
        ----------
        association: Association
            Association that contains truth and track as its objects

        Returns
        -------
        GroundTruthPath, Track
            True object and track that are the objects of the `association`
        """
        truth, track = association.objects
        # Sets aren't ordered, so need to ensure correct path is truth/track
        if isinstance(truth, Track):
            truth, track = track, truth
        return truth, track

    @staticmethod
    def total_time_tracked(manager, truth):
        """:math:`TT`. Calculate the total time a `truth` is tracked for by tracks contained by
        `manager`.

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used
        truth: GroundTruthPath
            True object

        Returns
        -------
        float
            Number of seconds that `truth` is tracked for
        """
        assocs = manager.association_set.associations_including_objects([truth])

        if len(assocs) == 0:
            return 0

        truth_timestamps = sorted(state.timestamp for state in truth.states)

        total_time = 0
        for current_time, next_time in zip(truth_timestamps[:-1], truth_timestamps[1:]):
            for assoc in assocs:
                # If both timestamps are in one association then add the difference to the total
                # difference and stop looking
                if current_time in assoc.time_range and next_time in assoc.time_range:
                    total_time += (next_time - current_time).total_seconds()
                    break
        return total_time

    @staticmethod
    def min_num_tracks_needed_to_track(manager, truth):
        """:math:`NU_j`. Calculate the minimum number of tracks needed to track `truth` using
        tracks held
        by `manager`.

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used
        truth: GroundTruthPath
            True object

        Returns
        -------
        int
            Minimum number of tracks needed to track `truth`
        """
        assocs = sorted(manager.association_set.associations_including_objects([truth]),
                        key=attrgetter('time_range.end_timestamp'),
                        reverse=True)

        if len(assocs) == 0:
            return 0

        truth_timestamps = sorted(state.timestamp for state in truth.states)
        num_tracks_needed = 0
        timestamp_index = 0

        while timestamp_index < len(truth_timestamps):
            current_time = truth_timestamps[timestamp_index]
            assoc_at_time = next((assoc for assoc in assocs if current_time in assoc.time_range),
                                 None)
            if not assoc_at_time:
                timestamp_index += 1
            else:
                end_time = assoc_at_time.time_range.end_timestamp
                num_tracks_needed += 1

                # If not yet at the end of the truth timestamps indices, move on to the next
                try:
                    # Move to next timestamp index after current association's end timestamp
                    timestamp_index = truth_timestamps.index(end_time, timestamp_index + 1) + 1
                except ValueError:
                    break
        return num_tracks_needed

    def rate_of_track_number_changes(self, manager, ground_truths):
        """:math:`R`. Calculate the average rate of track number changes for true objects held by
        `manager`.

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used
        ground_truths: set or list of :class:`~.GroundTruthPath` or :class:`~.Track` objects
            Containing the ground truth or track data to be used


        Returns
        -------
        float
            Average rate of track number changes
        """
        numerator = sum(self.min_num_tracks_needed_to_track(manager, truth) - 1
                        for truth in ground_truths)
        denominator = sum(self.total_time_tracked(manager, truth)
                          for truth in ground_truths)

        return numerator / denominator if denominator != 0 else 0

    @staticmethod
    def truth_lifetime(truth):
        """:math:`T`. Calculate how long `truth` exists for.

        Parameters
        ----------
        truth: GroundTruthPath
            True object

        Returns
        -------
        float
            Number of seconds that `truth` exists for
        """
        timestamps = [state.timestamp for state in truth.states]
        return (max(timestamps) - min(timestamps)).total_seconds()

    @staticmethod
    def longest_track_time_on_truth(manager, truth):
        """:math:`TL_j`. Calculate the longest time that a single track is associated to `truth`
        using associations held by `manager`.

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used
        truth: GroundTruthPath
            True object

        Returns
        -------
        float
            Number of seconds of longest association to `truth`
        """
        assocs = manager.association_set.associations_including_objects({truth})
        return max(assoc.time_range.duration.total_seconds() for assoc in assocs) if assocs else 0


class IDSIAPMetrics(SIAPMetrics):
    r"""ID-based SIAP Metrics

    SIAP metric generator that additionally computes ID-based SIAP metrics.

    ID-based Metrics:
        * ID Completeness (CID): ID-based SIAP. Fraction of true objects with an assigned ID. The
          output is in the range :math:`0:1`, with a target score of 1.
        * ID Correctness (IDC): ID-based SIAP. Fraction of true objects with correct ID
          assignment. The output is in the range :math:`0:1`, with a target score of 1.
        * ID Ambiguity (IDA): ID-based SIAP. Fraction of true objects with ambiguous ID
          assignment. The output is in the range :math:`0:1`, with a target score of 0.

    Notes
    -----
        * This implementation assumes that track and ground truth path IDs are implemented via
          metadata, whereby the strings :attr:`track_id` and :attr:`truth_id` are keys to track and
          truth metadata entries with ID data respectively.
        * :class:`~.Track` types store metadata outside of their `states` attribute. Therefore the
          ID SIAPs make metadata comparisons via the tracks' last ID metadata values (as calling
          `track.metadata` will return a track's metadata at the end of its life). To provide a
          better implementation, one might modify :class:`~.Track` types to contain a list of
          `state` types that hold their own metadata.

    Reference
        [1] Single Integrated Air Picture (SIAP) Metrics Implementation, Votruba et al, 29-10-2001
    """

    truth_id: str = Property(doc="Metadata key for ID of each ground truth path in data-set")
    track_id: str = Property(doc="Metadata key for ID of each track in data-set")

    def compute_metric(self, manager, **kwargs):
        r"""Compute metrics:

        .. math::
            \begin{alignat*}{3}
                \textrm{Name} &\quad \textrm{At Time} &&\quad \textrm{TimeRange}\\
                C &\quad \frac{JT({t})}{J({t})} &&\quad \frac{\sum_{t_{start}}^{t_{end}}JT({t})}
                {\sum_{t_{start}}^{t_{end}}J({t})}\\
                A &\quad \frac{N{A}({t})}{JT({t})} &&\quad \frac{\sum_{t_{start}}^{t_{end}}
                N{A}({t})}{\sum_{t_{start}}^{t_{end}}JT({t})}\\
                S &\quad \frac{N({t}) - N{A}({t})}{N({t})} &&\quad \frac{\sum_{t_{start}}^{t_{end}}
                [N({t}) - N{A}({t})]}{\sum_{t_{start}}^{t_{end}}N({t})}\\
                PA &\quad \frac{{\sum_{n\in tracks}PA_{n}(t)}}{NA(t)} &&\quad
                \frac{\sum_{t_{start}}^{t_{end}}{\sum_{n\in tracks}PA_{n}(t)}}
                {\sum_{t_{start}}^{t_{end}}{NA(t)}}\\
                VA &\quad \frac{{\sum_{n\in tracks}VA_{n}(t)}}{NA(t)} &&\quad
                \frac{\sum_{t_{start}}^{t_{end}}{\sum_{n\in tracks}VA_{n}(t)}}
                {\sum_{t_{start}}^{t_{end}}{NA(t)}}\\
                R &\quad -- &&\quad \frac{\sum_{j\in truths}NU_j-1}{\sum_{j\in truths}TT_j}\\
                LS &\quad -- &&\quad \frac{\sum_{j\in truths}T{L}_{j}}{\sum_{j\in truths}T_{j}}\\
                CID &\quad \frac{J{T}({t}) - J{U}({t})}{JT({t})}
                &&\quad \frac{\sum_{t_{start}}^{t_{end}}[J{T}({t}) - J{U}({T})]}
                {\sum_{t_{start}}^{t_{end}}J{T}({t})}\\
                IDC &\quad \frac{J{C}({t})}{JT({t})} &&\quad \frac{\sum_{t_{start}}^{t_{end}}
                J{C}({t})}{\sum_{t_{start}}^{t_{end}}J{T}({t})}\\
                IDA &\quad \frac{J{A}({t})}{JT({t})} &&\quad \frac{\sum_{t_{start}}^{t_{end}}
                J{A}({t})}{\sum_{t_{start}}^{t_{end}}J{T}({t})}
            \end{alignat*}

        Parameters
        ----------
        manager : MetricManager
            containing the data to be used to create the metric(s)

        Returns
        -------
        : list of :class:`~.Metric` objects
            Generated metrics
        """

        metrics = super().compute_metric(manager, **kwargs)

        timestamps = manager.list_timestamps(generator=self)

        ground_truths = self._get_data(manager, self.truths_key)

        id_completeness_at_times = list()
        id_correctness_at_times = list()
        id_ambiguity_at_times = list()

        JT_sum = JU_sum = JC_sum = JI_sum = JA_sum = 0

        for timestamp in timestamps:
            JTt = self.num_associated_truths_at_time(manager, ground_truths, timestamp)
            JT_sum += JTt
            JUt, JCt, JIt = self.num_id_truths_at_time(manager, ground_truths, timestamp)
            JU_sum += JUt
            JC_sum += JCt
            JI_sum += JIt
            JAt = JTt - JCt - JIt - JUt
            JA_sum += JAt

            id_completeness_at_times.append(
                SingleTimeMetric(title="SIAP ID Completeness at timestamp",
                                 value=(JTt - JUt) / JTt if JTt != 0 else 0,
                                 timestamp=timestamp,
                                 generator=self)
            )
            id_correctness_at_times.append(
                SingleTimeMetric(title="SIAP ID Correctness at timestamp",
                                 value=JCt / JTt if JTt != 0 else 0,
                                 timestamp=timestamp,
                                 generator=self)
            )
            id_ambiguity_at_times.append(
                SingleTimeMetric(title="SIAP Ambiguity at timestamp",
                                 value=JAt / JTt if JTt != 0 else 0,
                                 timestamp=timestamp,
                                 generator=self)
            )

        time_range = TimeRange(min(timestamps), max(timestamps))

        id_completeness = TimeRangeMetric(title="SIAP ID Completeness",
                                          value=(JT_sum - JU_sum) / JT_sum if JT_sum != 0 else 0,
                                          time_range=time_range,
                                          generator=self)
        id_correctness = TimeRangeMetric(title="SIAP ID Correctness",
                                         value=JC_sum / JT_sum if JT_sum != 0 else 0,
                                         time_range=time_range,
                                         generator=self)
        id_ambiguity = TimeRangeMetric(title="SIAP ID Ambiguity",
                                       value=JA_sum / JT_sum if JT_sum != 0 else 0,
                                       time_range=time_range,
                                       generator=self)

        id_completeness_at_times = TimeRangeMetric(title="SIAP ID Completeness at times",
                                                   value=id_completeness_at_times,
                                                   time_range=time_range,
                                                   generator=self)
        id_correctness_at_times = TimeRangeMetric(title="SIAP ID Correctness at times",
                                                  value=id_correctness_at_times,
                                                  time_range=time_range,
                                                  generator=self)
        id_ambiguity_at_times = TimeRangeMetric(title="SIAP ID Ambiguity at times",
                                                value=id_ambiguity_at_times,
                                                time_range=time_range,
                                                generator=self)

        metrics.extend([id_completeness, id_correctness, id_ambiguity,
                        id_completeness_at_times, id_correctness_at_times, id_ambiguity_at_times])
        return metrics

    def find_track_id(self, track, timestamp):
        """Find `track` ID at `timestamp`.

        Parameters
        ----------
        track: Track
            Track object
        timestamp: datetime.datetime
            Timestamp to retrieve ID data at

        Returns
        -------
        any
            `track` ID at `timestamp`
        """
        state = track[timestamp]
        index = track.index(state)
        metadata = track.metadatas[index]
        return metadata.get(self.track_id)

    def num_id_truths_at_time(self, manager, ground_truths, timestamp):
        """:math:`JU`, :math:`JC`, :math:`JI`. Calculate the number of true objects that are:
         Un-identified, correctly identified, incorrectly identified at `timestamp` according to
         associations held by `manager`.

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used
        ground_truths: set or list of :class:`~.GroundTruthPath` or :class:`~.Track` objects
            Containing the ground truth or track data to be used
        timestamp: datetime.datetime
            Timestamp at which to consider associations

        Returns
        -------
        int, int, int
            Number of true objects: Un-identified, correctly identified, incorrectly identified

        Note
        ----
            * A true object is considered to be un-identified if all tracks associated with it at
              `timestamp` have no ID.
            * A true object is considered to be correctly identified if all tracks associated with
              it at `timestamp` have the same ID as it.
            * A true object is considered to be incorrectly identified if all tracks associated
              with it at `timestamp` have a different ID to it.
            * A true object is considered to have ambiguous identification if tracks associated
              with it have differing ID (this value is calculated in the :meth:`~.compute_metric`
              method).
        """

        unknown_count = 0
        correct_count = 0
        incorrect_count = 0

        assocs = manager.association_set.associations_at_timestamp(timestamp)
        for truth in ground_truths:
            truth_id = truth.metadata.get(self.truth_id)
            track_ids = list()

            truth_assocs = [assoc for assoc in assocs if truth in assoc.objects]

            if len(truth_assocs) == 0:
                continue

            for assoc in truth_assocs:
                _, track = self.truth_track_from_association(assoc)

                track_ids.append(self.find_track_id(track, timestamp))

            if all(track_id is None for track_id in track_ids):
                unknown_count += 1
            elif (all(track_id == truth_id and track_id is not None for track_id in track_ids)
                  and truth_id is not None):
                correct_count += 1
            elif all(track_id != truth_id and track_id is not None for track_id in track_ids):
                incorrect_count += 1

        return unknown_count, correct_count, incorrect_count
