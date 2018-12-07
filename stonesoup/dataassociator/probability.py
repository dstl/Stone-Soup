# -*- coding: utf-8 -*-

from .base import DataAssociator
from ..base import Property
from ..hypothesiser import Hypothesiser
from ..types import Probability, MissedDetection


class SimplePDA(DataAssociator):
    """Simple Probabilistic Data Associatoion (PDA)

    Given a set of detections and a set of tracks, each detection has a
    probability that it is associated each specific track.  For each track,
    associate the highest probability (remaining) detection hypothesis with
    that track.

    This particular data associator assumes no gating; all detections have the
    possibility to be associated with any track.  This can lead to excessive
    computation time.
    """

    hypothesiser = Property(
        Hypothesiser,
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, time):
        """Associate detections with predicted states.

        Parameters
        ----------
        tracks : list of :class:`Track`
            Current tracked objects
        detections : list of :class:`Detection`
            Retrieved measurements
        time : datetime
            Detection time to predict to

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
        """

        # Generate a set of hypotheses for each track on each detection
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        associations = {}
        associated_measurements = set()
        while tracks > associations.keys():
            # Define a 'greedy' association
            highest_probability_detection = None
            highest_probability = Probability(0)
            for track in tracks - associations.keys():
                for weighted_measurement in \
                        hypotheses[track].weighted_measurements:
                    # A measurement may only be associated with a single track
                    current_probability = weighted_measurement["weight"]
                    if weighted_measurement["measurement"] in \
                            associated_measurements:
                        continue
                    # best_hypothesis is 'greater than' other
                    if (highest_probability_detection is None
                            or current_probability > highest_probability):
                        highest_probability_detection = \
                            weighted_measurement["measurement"]
                        highest_probability = current_probability
                        highest_probability_track = track

            hypotheses[highest_probability_track].\
                set_selected_measurement(highest_probability_detection)
            associations[highest_probability_track] = \
                hypotheses[highest_probability_track]
            if not isinstance(highest_probability_detection, MissedDetection):
                associated_measurements.add(highest_probability_detection)

        return associations
