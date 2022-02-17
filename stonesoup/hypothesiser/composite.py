# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Sequence

from .base import Hypothesiser
from ..base import Property
from ..predictor.composite import CompositePredictor
from ..types.hypothesis import CompositeProbabilityHypothesis
from ..types.multihypothesis import MultipleHypothesis


class CompositeHypothesiser(Hypothesiser):
    """Composite hypothesiser type

        A composition of ordered sub-hyposisers (:class:`~.Hypothesiser`). Hypothesises each
        sub-state of a track-detection pair using a corresponding sub-hypothesiser.
    """

    sub_hypothesisers: Sequence[Hypothesiser] = Property(
        doc="Sequence of sub-hypothesisers comprising the composite hypothesiser. Must not be "
            "empty. These must be hypothesisers that return probability-weighted hypotheses, in "
            "order for composite hypothesis weights to be calculated.")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if len(self.sub_hypothesisers) == 0:
            raise ValueError("Cannot create an empty composite hypothesiser")

        if any(not isinstance(sub_hypothesiser, Hypothesiser)
               for sub_hypothesiser in self.sub_hypothesisers):
            raise ValueError("All sub-hypothesisers must be a hypothesiser type")

        # create predictor as composition of sub-hypothesisers' predictors
        sub_predictors = list()
        for sub_hypothesiser in self.sub_hypothesisers:
            sub_predictors.append(sub_hypothesiser.predictor)
        self.predictor = CompositePredictor(sub_predictors)

    def hypothesise(self, track, detections, timestamp):
        """Evaluate and return all track association hypotheses.

        For a given track and a set of N available detections, return a MultipleHypothesis object
        composed of N+1 :class:`~.CompositeProbabilityHypothesis` (including a null hypothesis),
        each with an associated probability.

        Parameters
        ----------
        track: :class:`~.Track`
            The track object to hypothesise on, existing in a composite state space
        detections: :class:`set`
            A set of :class:`~CompositeDetection` objects, representing the available detections.
        timestamp: :class:`datetime.datetime`
            A timestamp used when evaluating the state and measurement predictions. Note that if a
            given detection has a non empty timestamp, then prediction will be performed according
            to the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~CompositeHypothesis` objects, each of which containing a
            sequence of :class:`~.SingleHypothesis` objects
        """

        all_hypotheses = list()

        # Common state & measurement prediction
        prediction = self.predictor.predict(track, timestamp=timestamp)

        null_sub_hypotheses = list()

        # as each detection is composite, it will have a set of sub-hypotheses paired to it
        detections_hypotheses = defaultdict(list)

        # loop over the sub-states of the track and sub-hypothesisers
        for sub_state_index, (sub_state, sub_hypothesiser) in enumerate(
                zip(track[-1].sub_states, self.sub_hypothesisers)):
            # store all sub-detections produced from sub-state index i
            sub_state_detections = set()
            # need way to get whole composite detections back from their i-th components
            # create dictionary, keyed by the i-th components, where the value is whole detection
            # will keep track of all detections that have an i-th component
            relevant_detections = dict()
            for detection in detections:
                try:
                    sub_detection_index = detection.mapping.index(sub_state_index)
                except ValueError:
                    continue
                sub_detection = detection[sub_detection_index]
                sub_state_detections.add(sub_detection)
                relevant_detections[sub_detection] = detection

            # get all hypotheses for the i-th component, considering i-th component of track state
            sub_hypotheses = sub_hypothesiser.hypothesise(sub_state, sub_state_detections,
                                                          timestamp=timestamp)
            # get the set of single hypotheses back
            sub_hypotheses = sub_hypotheses.single_hypotheses

            # Store sub-null-hypothesis for detections that didn't have i-th component
            sub_null_hypothesis = None
            while sub_hypotheses:
                sub_hypothesis = sub_hypotheses.pop()
                if not sub_hypothesis:
                    sub_null_hypothesis = sub_hypothesis
                else:
                    # get whole detection back, using
                    relevant_detection = relevant_detections[sub_hypothesis.measurement]
                    # Add hypothesis to detection's hypotheses container
                    detections_hypotheses[relevant_detection].append(sub_hypothesis)

            # For detections without i-th component, use sub-missed detection hypothesis
            for detection in detections - set(relevant_detections.values()):
                detections_hypotheses[detection].append(sub_null_hypothesis)

            # Add sub-null-hypothesis to composite null hypothesis
            null_sub_hypotheses.append(sub_null_hypothesis)

        # add a composite hypothesis for each detection
        for detection in detections:
            # get all sub-hypotheses for detection
            sub_hypotheses = detections_hypotheses[detection]

            all_hypotheses.append(CompositeProbabilityHypothesis(prediction=prediction,
                                                                 measurement=detection,
                                                                 sub_hypotheses=sub_hypotheses))

        # add null-hypothesis
        all_hypotheses.append(CompositeProbabilityHypothesis(prediction=prediction,
                                                             measurement=None,
                                                             sub_hypotheses=null_sub_hypotheses))

        return MultipleHypothesis(all_hypotheses, normalise=True, total_weight=1)

    def __contains__(self, item):
        return self.sub_hypothesisers.__contains__(item)

    def __getitem__(self, index):
        """Can be indexed as a list, or sliced, in which case a new composite hypothesiser will be
        created from the sub-list of sub-hypothesisers."""
        if isinstance(index, slice):
            return self.__class__(self.sub_hypothesisers.__getitem__(index))
        return self.sub_hypothesisers.__getitem__(index)

    def __iter__(self):
        return self.sub_hypothesisers.__iter__()

    def __len__(self):
        return self.sub_hypothesisers.__len__()
