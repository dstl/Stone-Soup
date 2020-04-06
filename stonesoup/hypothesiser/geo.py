# -*- coding: utf-8 -*-
import copy
from functools import lru_cache
from itertools import groupby

import utm

from .base import Hypothesiser
from ..base import Property
from ..types.multihypothesis import MultipleHypothesis
from ..types.track import Track


class DynamicUTMHypothesiserWrapper(Hypothesiser):
    """Distance hypothesiser aware of UTM zones

    This hypothesiser utilises :attr:`.Track.metadata` and :attr:`.Detection.metadata` `utm_zone`
    value to translate the :class:`~.Track` to the UTM zone of the :class:`~.Detection`. This will
    mean the track's state space will change zones over time.

    .. note::

        This currently only works for :class:`~.State` and :class:`~.GaussianState` based states.
    """

    hypothesiser = Property(Hypothesiser, doc="Hypothesiser that is being wrapped.")
    utm_mapping = Property((int, int), doc="Mapping of easting and northing")

    def _track_to_utm_zone(self, track, track_zone, detection_zone):
        return Track([self._state_to_utm_northern(
            track.state, self.utm_mapping,
            track_zone[0], track_zone[1] >= 'N', detection_zone[0], detection_zone[1] >= 'N')])

    @staticmethod
    @lru_cache()
    def _state_to_utm_northern(
            state, mapping,
            track_zone_number, track_northern, detection_zone_number, detection_northern):

        if track_zone_number != detection_zone_number:
            lat, long = utm.to_latlon(
                *state.state_vector[mapping, 0], track_zone_number, northern=track_northern)
            easting, northing, _, _ = utm.from_latlon(
                lat, long, detection_zone_number, force_northern=track_northern)
        else:
            # Same zone number, so not need to convert to lat, lon and back again.
            easting, northing = state.state_vector[mapping, 0]

        # Add/subtract equator offset for southern if northern value differs
        northing += 10000000 * (track_northern - detection_northern)

        # Shallow copy is fine...
        state = copy.copy(state)
        # as long as we also copy the state vector we are modifying
        state.state_vector = state.state_vector.copy()
        state.state_vector[mapping, 0] = easting, northing
        return state

    @staticmethod
    def _utm_zone_getter(state):
        return state.metadata['utm_zone']

    def hypothesise(self, track, detections, timestamp):
        hypotheses = list()

        track_zone = self._utm_zone_getter(track)
        track_zone_hit = False
        for detection_zone, utm_detections in groupby(
                sorted(detections, key=self._utm_zone_getter), self._utm_zone_getter):

            if track_zone == detection_zone:
                # Just run with existing track
                hypotheses.extend(self.hypothesiser.hypothesise(track, utm_detections, timestamp))
                track_zone_hit = True
            else:
                # TODO: Could gate detections out here using zone info
                # Get track in detections UTM zone
                utm_track = self._track_to_utm_zone(track, track_zone, detection_zone)
                utm_hypotheses = self.hypothesiser.hypothesise(
                    utm_track, utm_detections, timestamp)
                # Extend with all but missed detection hypotheses
                hypotheses.extend((hypothesis for hypothesis in utm_hypotheses if hypothesis))

        # Need to keep sure we process track zone to get missed detection
        if not track_zone_hit:
            hypotheses.extend(self.hypothesiser.hypothesise(track, {}, timestamp))

        return MultipleHypothesis(sorted(hypotheses, reverse=True))
