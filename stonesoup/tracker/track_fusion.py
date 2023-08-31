import uuid
from datetime import datetime
from typing import Collection
from typing import Set, Tuple, Iterator, Sequence

from stonesoup.base import Property
from stonesoup.dataassociator.base import TrackToTrackAssociator
from stonesoup.feeder.base import MultipleTrackFeeder
from stonesoup.mixturereducer import MixtureReducer
from stonesoup.tracker import Tracker
from stonesoup.types.association import AssociationSet, Association
from stonesoup.types.track import Track, CompositeTrack


class TrackFusedTracker(Tracker):
    """
    Tracks from multiple sources arrive via the `multiple_track_feeder`. These tracks are
    associated together using the `track_associator`. Associated tracks are combined by the
    `state_combiner` to form a new state. A new :class:`~.CompositeTrack` is created with this
    state. The track ID of the new track is created by combining the source track IDs. The
    tracker yields (datetime, Set[Track]) with the tracks containing the new combined tracks and
    the unassociated tracks.
    """

    multiple_track_feeder: MultipleTrackFeeder = Property()
    track_associator: TrackToTrackAssociator = Property()
    state_combiner: MixtureReducer = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()

    def __iter__(self) -> Iterator[Tuple[datetime, Set[Track]]]:
        for time, sets_of_tracks in self.multiple_track_feeder:
            yield self.update_tracker(time, sets_of_tracks)

    def __next__(self):
        raise NameError

    def update_tracker(self, time, input_tracks: Sequence[Set[Track]]) \
            -> Tuple[datetime, Set[Track]]:

        associated_track_sets, unassociated_tracks = self.associate_tracks(input_tracks)
        combined_tracks = {self.combine_tracks(association) for association in
                           associated_track_sets}

        return time, {*combined_tracks, *unassociated_tracks}

    def associate_tracks(self, received_track_sets: Sequence[Set[Track]]) \
            -> Tuple[AssociationSet, Collection[Track]]:
        """
        This method determines which tracks should be associated together and returns a collection
        of AssociationSet objects which contain tracks and a collection of unassociated tracks
        """

        associated_track_sets, unassociated_tracks_sets = \
            self.track_associator.associate_tracks_plus(*received_track_sets)

        all_unassociated_tracks = [track for unassociated_tracks in unassociated_tracks_sets
                                   for track in unassociated_tracks]
        return associated_track_sets, all_unassociated_tracks

    def combine_tracks(self, association: Association) -> Track:
        """
        Three steps to combine the tracks:

         #. Combines the states from the associated tracks using the :attr:`.state_combiner`.
         #. Create a new track from this state
         #. Assign new track_id using :func:`.get_fused_id`

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


def get_fused_id(*track_ids: str) -> str:
    """ Combines `n` track ids 128 bit hex str into another 128 bit hex str """

    output_int = 0
    for track_id in track_ids:
        id_int = uuid.UUID(track_id).int
        output_int ^= id_int

    output_uuid = uuid.UUID(int=output_int)

    return str(output_uuid)
