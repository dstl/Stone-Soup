import uuid
from datetime import datetime
from typing import Collection
from typing import Set, Tuple, Sequence

from stonesoup.base import Property
from stonesoup.dataassociator.base import TrackToTrackAssociator
from stonesoup.feeder.base import MultipleTrackFeeder
from stonesoup.mixturereducer import MixtureReducer
from stonesoup.models.transition.linear import ConstantVelocity, \
    CombinedLinearGaussianTransitionModel
from stonesoup.predictor.base import Predictor
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.tracker import Tracker
from stonesoup.types.association import AssociationSet, Association
from stonesoup.types.track import Track, CompositeTrack


class TrackFusedTracker(Tracker):

    track_associator: TrackToTrackAssociator = Property()
    state_combiner: MixtureReducer = Property()
    multiple_track_feeder: MultipleTrackFeeder = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()

    @property
    def detector(self):
        # This allows the standard tracker loop to be used
        return self.multiple_track_feeder

    def update_tracker(self, time, input_tracks: Sequence[Set[Track]]) -> \
            Tuple[datetime, Set[Track]]:
        # print("Input Tracks:", *[[track.id[0:5] for track in track_set]
        #                          for track_set in input_tracks])
        associated_track_sets, unassociated_tracks = self.associate_tracks(input_tracks)
        combined_tracks = {self.combine_tracks(association) for association in
                           associated_track_sets}
        # print("Output Tracks:",
        #       "c", [track.id[0:5] for track in combined_tracks],
        #       "i", *[track.id[0:5] for track in unassociated_tracks])

        return time, {*combined_tracks, *unassociated_tracks}

    def associate_tracks(self, received_track_sets: Sequence[Set[Track]]) -> \
            Tuple[AssociationSet, Collection[Track]]:
        """
        This method determines which tracks should be associated together and returns a collection
        of AssociationSet objects which contain tracks and a collection of unassociated tracks
        :return:
        """

        associated_track_sets, unassociated_tracks_sets = \
            self.track_associator.associated_and_unassociated_tracks(*received_track_sets)

        all_unassociated_tracks = [track for unassociated_tracks in unassociated_tracks_sets
                                   for track in unassociated_tracks]

        return associated_track_sets, all_unassociated_tracks

    def combine_tracks(self, association: Association) -> Track:
        """

        :param association:
        :return:
        """
        states = (track.state for track in association.objects)
        new_state = self.state_combiner.merge_components(*states)

        new_track_id_tuple = tuple(sorted(track.id for track in association.objects))
        new_track_id = get_fused_id(*new_track_id_tuple)
        return CompositeTrack(id=new_track_id,
                              states=[new_state],
                              sub_tracks=list(association.objects)
                              )

    def tracks(self):
        return self._tracks


class AsynchronousTrackFusedTracker(TrackFusedTracker):

    def update_tracker(self, sim_time, input_tracks: Sequence[Set[Track]]) -> \
            Tuple[datetime, Set[Track]]:
        self.predict_input_tracks(sim_time, input_tracks)
        return super().update_tracker(sim_time, input_tracks)

    def predict_input_tracks(self, time: datetime, received_track_sets: Sequence[Set[Track]]):
        for tracks in received_track_sets:
            for track in tracks:
                self.predict_track(track, time)

    def predict_track(self, track: Track, timestamp: datetime) -> Track:
        state = track.state

        if state.timestamp == timestamp:
            pass
        elif state.timestamp < timestamp:
            predictor = self.get_predictor(track)
            track.append(predictor.predict(state, timestamp))
        elif state.timestamp > timestamp:
            pass

        return track

    @staticmethod
    def get_predictor(track: Track) -> Predictor:
        """
        This is a bad method but will do for now
        :param track:
        :return:
        """
        transition_model = [ConstantVelocity(0.0) for _ in range(track.state.ndim//2)]
        predictor = KalmanPredictor(CombinedLinearGaussianTransitionModel(transition_model))
        return predictor


def get_fused_id(*track_ids: str) -> str:
    """
    Combines 'n' track ids 128 bit hex str into another 128 bit hex str
    :param track_ids:
    :return:
    """
    #
    output_int = 0
    for track_id in track_ids:
        id_int = uuid.UUID(track_id).int
        output_int ^= id_int

    output_uuid = uuid.UUID(int=output_int)

    return str(output_uuid)
