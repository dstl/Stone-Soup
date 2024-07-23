from typing import Set, Tuple

from stonesoup.measures.state import Measure
from stonesoup.types.track import Track

from ..base import Property
from ..types.association import AssociationSet, TimeRangeAssociation
from ..types.metric import TimeRangeMetric
from ..types.time import TimeRange
from .base import MetricGenerator
from .manager import MultiManager


class CLEARMOTMetrics(MetricGenerator):
    """TODO"""
    tracks_key: str = Property(doc='Key to access set of tracks added to MetricManager',
                               default='tracks')
    truths_key: str = Property(doc="Key to access set of ground truths added to MetricManager. "
                                   "Or key to access a second set of tracks for track-to-track "
                                   "metric generation",
                               default='groundtruth_paths')
    
    distance_measure: Measure = Property(
        doc="Distance measure used in calculating position accuracy scores.")

    def compute_metric(self, manager: MultiManager, **kwargs):

        timestamps = manager.list_timestamps(generator=self)
        tracks = self._get_data(manager, self.tracks_key)
        ground_truths = self._get_data(manager, self.truths_key)

        # TODO: CODE HERE
        motp_score = self.compute_motp(manager)

        time_range = TimeRange(min(timestamps), max(timestamps))

        motp = TimeRangeMetric(title="MOTP",
                               value=motp_score,
                               time_range=time_range,
                               generator=self)
        mota = TimeRangeMetric(title="MOTA",
                               value=0.0,
                               time_range=time_range,
                               generator=self)
        return [motp, mota]
    
    def compute_motp(self, manager: MultiManager) -> float:
        associations: AssociationSet = self.manager.association_set
        associations.associations_including_objects

        associations: AssociationSet = manager.association_set

        timestamps = manager.list_timestamps(generator=self)

        associations: Set[TimeRangeAssociation] = manager.association_set.associations

        error_sum = 0.0
        num_associated_truth_timestamps = 0
        for association in associations:

            truth, track = self.truth_track_from_association(association)

            time_range = association.time_range
            timestamps_for_association = timestamps[(timestamps >= time_range.start_timestamp) & 
                                                    (timestamps <= time_range.end_timestamp)]
            
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

        
    def compute_mota(self, manager: MultiManager):

        timestamps = manager.list_timestamps(generator=self)

        truth_state_set = manager.states_sets[self.truths_key]
        tracks_state_set = manager.states_sets[self.tracks_key]

        num_misses, num_false_positives, num_id_switches = 0, 0, 0

        for i, timestamp in enumerate(timestamps):

            truths_at_timestamp = []
            tracks_at_timestamp = []

            associations = manager.association_set.associations_at_timestamp(timestamp)

            matched_truths_at_timestamp = set()
            matched_tracks_at_timestamp = set()
            for association in associations:
                truth, track = self.truth_track_from_association(association)
                matched_truths_at_timestamp.add(truth.id)
                matched_tracks_at_timestamp.add(track.id)

            unmatched_truth_ids = list(filter(lambda x: x.id not in matched_truths_at_timestamp, 
                                                truths_at_timestamp))
            num_misses += len(unmatched_truth_ids)

            unmatched_track_ids = list(filter(lambda x: x.id not in matched_tracks_at_timestamp, 
                                                tracks_at_timestamp))
            num_false_positives += len(unmatched_track_ids)
            
            # TODO: num_id_switches
            # if i > 0:
            #     associations_prev = manager.association_set.associations_at_timestamp(timestamps[i-1])

            #     truths_at_prev_timestamp = set()
            #     for association in associations_prev:
            #         truth, track = self.truth_track_from_association(association)
            #         truths_at_prev_timestamp.add(truth.id)
                
            #     truths_ids_at_both_timestamps = truths_at_prev_timestamp.intersection(matched_truths_at_timestamp)

        number_of_gt_states = self.compute_total_number_of_gt_states(manager)
        return 1 - (num_misses + num_false_positives + num_id_switches)/number_of_gt_states
    
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
