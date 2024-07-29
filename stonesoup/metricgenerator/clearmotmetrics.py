import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union

import numpy as np

from ..base import Property
from ..measures.state import Measure
from ..types.association import AssociationSet, TimeRangeAssociation
from ..types.groundtruth import GroundTruthPath
from ..types.metric import Metric, TimeRangeMetric
from ..types.time import CompoundTimeRange, TimeRange
from ..types.track import Track
from .base import MetricGenerator
from .manager import MultiManager

MatchSetAtTimestamp = Set[Tuple[str, str]]


class ClearMotMetrics(MetricGenerator):
    """TODO
    
    Reference
        [1] Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics,
            Bernardin et al, 2008
    """
    tracks_key: str = Property(doc='Key to access set of tracks added to MetricManager',
                               default='tracks')
    truths_key: str = Property(doc="Key to access set of ground truths added to MetricManager. "
                                   "Or key to access a second set of tracks for track-to-track "
                                   "metric generation",
                               default='groundtruth_paths')

    distance_measure: Measure = Property(
        doc="Distance measure used in calculating position accuracy scores.")

    def compute_metric(self, manager: MultiManager, **kwargs) -> List[Metric]:

        timestamps = manager.list_timestamps(generator=self)
        tracks = self._get_data(manager, self.tracks_key)
        ground_truths = self._get_data(manager, self.truths_key)

        # TODO: CODE HERE
        motp_score = self.compute_motp(manager)

        mota_score = self.compute_mota(manager)

        time_range = TimeRange(min(timestamps), max(timestamps))

        motp = TimeRangeMetric(title="MOTP",
                               value=motp_score,
                               time_range=time_range,
                               generator=self)
        mota = TimeRangeMetric(title="MOTA",
                               value=mota_score,
                               time_range=time_range,
                               generator=self)
        return [motp, mota]

    def compute_motp(self, manager: MultiManager) -> float:

        associations: AssociationSet = manager.association_set

        timestamps = sorted(manager.list_timestamps(generator=self))

        timestamps_as_numpy_array = np.array(timestamps)

        associations: Set[TimeRangeAssociation] = manager.association_set.associations

        error_sum = 0.0
        num_associated_truth_timestamps = 0
        for association in associations:

            truth, track = self.truth_track_from_association(association)

            time_range = association.time_range

            if isinstance(time_range, CompoundTimeRange):
                time_ranges = time_range.time_ranges
            else:
                time_ranges = [time_range,]

            mask = np.zeros(len(timestamps_as_numpy_array), dtype=bool)
            for time_range in time_ranges:
                mask = mask | ((timestamps_as_numpy_array >= time_range.start_timestamp)
                               & (timestamps_as_numpy_array <= time_range.end_timestamp))

            timestamps_for_association = timestamps_as_numpy_array[mask]

            num_associated_truth_timestamps += len(timestamps_for_association)
            for t in timestamps_for_association:
                truth_state_at_t = truth[t]
                track_state_at_t = track[t]
                error = self.distance_measure(truth_state_at_t, track_state_at_t)
                error_sum += error
        if num_associated_truth_timestamps > 0:
            return error_sum / num_associated_truth_timestamps
        else:
            return float("inf")

    def compute_total_number_of_gt_states(self, manager: MultiManager) -> int:
        truth_state_set: Set[Track] = manager.states_sets[self.truths_key]
        total_number_of_gt_states = sum(len(truth_track) for truth_track in truth_state_set)
        return total_number_of_gt_states

    def create_matches_at_time_lookup(self, manager: MultiManager) -> Dict[datetime.datetime, MatchSetAtTimestamp]:
        timestamps = manager.list_timestamps(generator=self)

        matches_by_timestamp = defaultdict(set)

        for i, timestamp in enumerate(timestamps):

            associations = manager.association_set.associations_at_timestamp(timestamp)

            for association in associations:
                truth, track = self.truth_track_from_association(association)
                match_truth_track = (truth.id, track.id)
                matches_by_timestamp[timestamp].add(match_truth_track)
        return matches_by_timestamp

    def compute_mota(self, manager: MultiManager):

        timestamps = manager.list_timestamps(generator=self)

        truth_state_set = manager.states_sets[self.truths_key]
        tracks_state_set = manager.states_sets[self.tracks_key]

        truth_ids_at_time = create_ids_at_time_lookup(truth_state_set)
        track_ids_at_time = create_ids_at_time_lookup(tracks_state_set)

        matches_at_time_lookup = self.create_matches_at_time_lookup(manager)

        num_misses, num_false_positives, num_miss_matches = 0, 0, 0

        for i, timestamp in enumerate(timestamps):

            print(f"i={i}")

            # TODO: add lookup here!
            truths_ids_at_timestamp = truth_ids_at_time[timestamp]
            tracks_ids_at_timestamp = track_ids_at_time[timestamp]

            matches_current = matches_at_time_lookup[timestamp]
            matched_truth_ids_curr = {match[0] for match in matches_current}
            matched_tracks_at_timestamp = {match[1] for match in matches_current}

            unmatched_truth_ids = list(filter(lambda x: x not in matched_truth_ids_curr,
                                              truths_ids_at_timestamp))
            num_misses += len(unmatched_truth_ids)

            unmatched_track_ids = list(filter(lambda x: x not in matched_tracks_at_timestamp,
                                              tracks_ids_at_timestamp))
            num_false_positives += len(unmatched_track_ids)

            if i > 0:

                matches_prev = matches_at_time_lookup[timestamps[i-1]]

                num_miss_matches_current = self._compute_number_of_miss_matches_from_match_sets(
                    matches_prev, matches_current)

                num_miss_matches += num_miss_matches_current

        number_of_gt_states = self.compute_total_number_of_gt_states(manager)

        return 1 - (num_misses + num_false_positives + num_miss_matches)/number_of_gt_states

    def _compute_number_of_miss_matches_from_match_sets(self, matches_prev: MatchSetAtTimestamp,
                                                        matches_current: MatchSetAtTimestamp) -> int:
        num_miss_matches_current = 0

        matched_truth_ids_prev = {match[0] for match in matches_prev}
        matched_truth_ids_curr = {match[0] for match in matches_current}
        truths_ids_at_both_timestamps = matched_truth_ids_prev & matched_truth_ids_curr

        for truth_id in truths_ids_at_both_timestamps:
            prev_matches_with_truth_id = list(
                filter(lambda match: match[0] == truth_id, matches_prev))
            cur_matches_with_truth_id = list(
                filter(lambda match: match[0] == truth_id, matches_current))

            # if len(prev_matches_with_truth_id) > 1:
            #     warnings.warn("More than one track per truth is not supported!")
            #     continue

            # if len(cur_matches_with_truth_id) > 1:
            #     warnings.warn("More than one track per truth is not supported!")
            #     continue

            matched_track_id_prev = prev_matches_with_truth_id[0][1]
            matched_track_id_curr = cur_matches_with_truth_id[0][1]

            if matched_track_id_prev != matched_track_id_curr:
                num_miss_matches_current += 1
        return num_miss_matches_current

    @staticmethod
    def truth_track_from_association(association) -> Tuple[Track, Track]:
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


def create_ids_at_time_lookup(tracks_set: Set[Union[Track, GroundTruthPath]]) \
        -> Dict[datetime.datetime, Set[str]]:

    track_ids_by_time = defaultdict(set)
    for track in tracks_set:
        for state in track.last_timestamp_generator():
            track_ids_by_time[state.timestamp].add(track.id)

    return track_ids_by_time
