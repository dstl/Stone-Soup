# -*- coding: utf-8 -*-

from .base import DataAssociator
from ..base import Property
from ..hypothesiser import Hypothesiser


class NearestNeighbour(DataAssociator):
    """Nearest Neighbour Associator

    Gates, scores and associates detections to a predicted state using the
    Nearest Neighbour method.
    """

    hypothesiser = Property(
        Hypothesiser,
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, time):
        """Associate detections with predicted states.

        Parameters
        ----------


        Returns
        -------

        """

        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        # 'Greedy' associator
        associations = {}
        associated_detections = set()
        while tracks > associations.keys():
            best_hypothesis = None
            for track in tracks - associations.keys():
                for hypothesis in hypotheses[track]:
                    if (hypothesis.detection not in associated_detections
                            and best_hypothesis is None
                            or hypothesis > best_hypothesis):
                        best_hypothesis = hypothesis
                        best_hypothesis_track = track

            associations[best_hypothesis_track] = best_hypothesis
            if best_hypothesis.detection is not None:
                associated_detections.add(best_hypothesis.detection)

        return associations
