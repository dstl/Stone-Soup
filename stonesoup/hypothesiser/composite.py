# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Sequence

from .base import Hypothesiser
from ..base import Property
from ..predictor.composite import CompositePredictor
from ..types.detection import MissedDetection, CompositeMissedDetection
from ..types.hypothesis import CompositeProbabilityHypothesis
from ..types.multihypothesis import MultipleHypothesis


class CompositeHypothesiser(Hypothesiser):
    """A composition of multiple sub-hypothesisers.

    Generate track predictions at detection times and calculate probabilities for all
    prediction-detection pairs for single prediction and multiple detections.

    Assumes tracks and measurements exist in composite state spaces.
    Utilises the output hypotheses of its :attr:`sub_hypothesisers` to attain a
    :class:`~.CompositeHypothesis` for each prediction-detection pair.
    """
    sub_hypothesisers: Sequence[Hypothesiser] = Property(
        doc="A sequence of sub-hypothesisers. These must be hypothesisers that return "
            "probability-weighted hypotheses, in order for composite hypothesis weights to be "
            "calculated.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # create predictor as composition of sub-hypothesisers' predictors
        sub_predictors = list()
        for sub_hypothesiser in self.sub_hypothesisers:
            sub_predictors.append(sub_hypothesiser.predictor)
        self.predictor = CompositePredictor(sub_predictors)

    def hypothesise(self, track, detections, timestamp):
        """ Evaluate and return all track association hypotheses.

        For a given track and a set of N available detections, return a MultipleHypothesis object
        with N+1 detections (first detection is a 'CompositeMissedDetection'), each with an
        associated probability.

        A resultant composite hypothesis' probability is calculated as the weighted product of

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

        Notes
        ----
        * Detections are required to have a timestamp, in order for the composite null-hypothesis
        to have a timestamp.
        * A 'weightings' attribute should be implemented to adjust sub-hypotheses' contributions
        to the composite hypothesis weights.

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
                if isinstance(sub_hypothesis.measurement, MissedDetection):
                    sub_null_hypothesis = sub_hypothesis
                else:
                    # get whole detection back, using
                    relevant_detection = relevant_detections[sub_hypothesis.measurement]
                    # Add hypothesis to detection's hypotheses container
                    detections_hypotheses[relevant_detection].append(sub_hypothesis)

            # For detections without i-th component, use sub-missed detection hypothesis
            for detection in detections - set(relevant_detections.values()):
                # TODO: should this be an 'un-measured detection hypothesis?
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

        # add missed-detection and corresponding null-hypothesis to dictionary
        missed_detection = CompositeMissedDetection(default_timestamp=timestamp)

        all_hypotheses.append(CompositeProbabilityHypothesis(prediction=prediction,
                                                             measurement=missed_detection,
                                                             sub_hypotheses=null_sub_hypotheses))

        return MultipleHypothesis(all_hypotheses, normalise=True, total_weight=1)
