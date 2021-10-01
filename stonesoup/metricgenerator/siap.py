# -*- coding: utf-8 -*-
import datetime
from operator import attrgetter

import numpy as np

from .base import MetricGenerator
from ..base import Property
from ..measures import Measure
from ..types.metric import SingleTimeMetric, TimeRangeMetric
from ..types.time import TimeRange
from ..types.track import Track


class NewSIAPs(MetricGenerator):
    position_measure: Measure = Property(
        doc="Distance measure used in calculating position accuracy scores.")
    velocity_measure: Measure = Property(
        doc="Distance measure used in calculating velocity accuracy scores."
    )

    def compute_metric(self, manager, **kwargs):

        timestamps = manager.list_timestamps()

        completeness_at_times = list()
        ambiguity_at_times = list()
        spuriousness_at_times = list()
        position_accuracy_at_times = list()
        velocity_accuracy_at_times = list()

        J_sum = JT_sum = NA_sum = N_sum = PA_sum = VA_sum = 0

        for timestamp in timestamps:
            Jt = self.num_truths_at_time(manager, timestamp)
            J_sum += Jt
            JTt = self.num_associated_truths_at_time(manager, timestamp)
            JT_sum += JTt
            NAt = self.num_associated_tracks_at_time(manager, timestamp)
            NA_sum += NAt
            Nt = self.num_tracks_at_time(manager, timestamp)
            N_sum += Nt
            PAt = self.accuracy_at_time(manager, timestamp, self.position_measure)
            PA_sum += PAt
            VAt = self.accuracy_at_time(manager, timestamps, self.velocity_measure)
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
                SingleTimeMetric(title="SIAP Positional Accuracy at timestamp",
                                 value=PAt / NAt if NAt != 0 else 0,
                                 timestamp=timestamp,
                                 generator=self)
            )
            velocity_accuracy_at_times.append(
                SingleTimeMetric(title="SIAP Positional Accuracy at timestamp",
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
        R = self.average_excess_tracks(manager)
        track_breakage = TimeRangeMetric(title="SIAP track breakage",
                                         value=1 / R if R != 0 else np.inf,
                                         time_range=time_range,
                                         generator=self)
        TL_sum = sum(self.longest_track_time_on_truth(manager, truth)
                     for truth in manager.groundtruth_paths)
        T_sum = sum(self.truth_lifetime(truth) for truth in manager.groundtruth_paths)
        continuity = TimeRangeMetric(title="SIAP continuity",
                                     value=TL_sum / T_sum if T_sum != 0 else 0,
                                     time_range=time_range,
                                     generator=self)

        completeness_at_times = TimeRangeMetric(title="SIAP completeness at times",
                                                value=completeness_at_times,
                                                time_range=time_range,
                                                generator=self)
        ambiguity_at_times = TimeRangeMetric(title="SIAP ambiguity at times",
                                             value=ambiguity_at_times,
                                             time_range=time_range,
                                             generator=self)
        spuriousness_at_times = TimeRangeMetric(title="SIAP spuriousness at times",
                                                value=spuriousness_at_times,
                                                time_range=time_range,
                                                generator=self)
        position_accuracy_at_times = TimeRangeMetric(title="SIAP position_accuracy at times",
                                                     value=position_accuracy_at_times,
                                                     time_range=time_range,
                                                     generator=self)
        velocity_accuracy_at_times = TimeRangeMetric(title="SIAP velocity at times",
                                                     value=velocity_accuracy_at_times,
                                                     time_range=time_range,
                                                     generator=self)
        return [completeness, ambiguity, spuriousness, position_accuracy, velocity_accuracy,
                track_breakage, continuity, completeness_at_times, ambiguity_at_times,
                spuriousness_at_times, position_accuracy_at_times, velocity_accuracy_at_times]

    @staticmethod
    def num_truths_at_time(manager, timestamp):
        """J(t)."""
        return sum(
            1
            for path in manager.groundtruth_paths
            if timestamp in (state.timestamp for state in path))

    @staticmethod
    def num_associated_truths_at_time(manager, timestamp):
        """JT(t)."""
        associations = manager.association_set.associations_at_timestamp(timestamp)
        association_objects = set.union(*[assoc.objects for assoc in associations])
        return sum(
            1
            for truth in manager.groundtruth_paths
            if truth in association_objects)

    @staticmethod
    def num_associated_tracks_at_time(manager, timestamp):
        """NA(t)."""
        associations = manager.association_set.associations_at_timestamp(timestamp)
        association_objects = set.union(*[assoc.objects for assoc in associations])
        return sum(
            1
            for truth in manager.tracks
            if truth in association_objects)

    @staticmethod
    def num_tracks_at_time(manager, timestamp):
        """N(t)."""
        return sum(
            1
            for track in manager.tracks
            if timestamp in (state.timestamp for state in track.states))

    def accuracy_at_time(self, manager, timestamp, measure):
        """PA(t) or VA(t) (dependent on measure).

        Note:
            Should this method only consider each track, truth once? Or add each and every
            association for each object as below?
        """
        associations = manager.association_set.associations_at_timestamp(timestamp)
        error_sum = 0
        for association in associations:
            truth, track = self.truth_track_from_association(association)
            error_sum += measure(truth[timestamp], track[timestamp])
        return error_sum

    @staticmethod
    def truth_track_from_association(association):
        """Find truth and track from an association."""
        truth, track = association.objects
        # Sets aren't ordered, so need to ensure correct path is truth/track
        if isinstance(truth, Track):
            truth, track = track, truth
        return truth, track

    @staticmethod
    def total_time_tracked(manager, truth):
        """TT"""
        assocs = manager.association_set.associations_including_objects([truth])

        if len(assocs) == 0:
            return 0

        truth_timestamps = sorted(state.timestamp for state in truth.states)

        total_time = datetime.timedelta(0)
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
        """NU"""
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

    def average_excess_tracks(self, manager):
        """R"""
        numerator = sum(self.min_num_tracks_needed_to_track(manager, truth) - 1
                        for truth in manager.groundtruth_paths)
        denominator = sum(self.total_time_tracked(manager, truth)
                          for truth in manager.groundtruth_paths)

        return numerator / denominator if denominator != 0 else 0

    @staticmethod
    def truth_lifetime(truth):
        """T"""
        timestamps = [state.timestamp for state in truth.states]
        return (max(timestamps) - min(timestamps)).total_seconds()

    @staticmethod
    def track_time_on_truth(manager, track, truth):
        """Tn"""
        assocs = manager.association_set.associations_including_objects({track, truth})
        return sum(assoc.time_range.duration.total_seconds() for assoc in assocs)

    def longest_track_time_on_truth(self, manager, truth):
        """TLj"""
        return max(self.track_time_on_truth(manager, track, truth) for track in manager.tracks)
