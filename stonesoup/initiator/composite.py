# -*- coding: utf-8 -*-
import warnings
from typing import Sequence

from .base import Initiator
from ..base import Property
from ..types.detection import MissedDetection
from ..types.hypothesis import SingleHypothesis, CompositeHypothesis
from ..types.state import CompositeState
from ..types.track import Track
from ..types.update import CompositeUpdate, Update


class CompositeUpdateInitiator(Initiator):
    """Composite initiator type

    A composition of sub-initiators (:class:`~.Initiator`).

    Requires that all sub-initiators have a defined prior state in order to compose its own
    composite prior state
    """

    sub_initiators: Sequence[Initiator] = Property(
        doc="Sequence of sub-initiators comprising the composite initiator. Must not be empty."
    )

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if len(self.sub_initiators) == 0:
            raise ValueError("Cannot create an empty composite initiator")

        if any(not isinstance(sub_initiator, Initiator) for sub_initiator in self.sub_initiators):
            raise ValueError("All sub-initiators must be an initiator type")

    @property
    def prior_state(self):
        return CompositeState([sub_initiator.prior_state for sub_initiator in self.sub_initiators])

    def initiate(self, detections, timestamp, **kwargs):
        """Utilises its sub-initiators to attempt to initiate a track for a
        :class:`~.CompositeDetection` in each sub-state space, then combines the resultant track
        states in to a :class:`~.CompositeState` track. If any sub-state is missing from the
        detection, or a sub-initiator fails to initiate in its sub-state space, the corresponding
        sub-state of the :attr:`prior_state` will be used instead.
        It is required that the sub-initiators initiate on a single measurement, in order for
        the individual sub-states of a track to be linked to one another."""

        tracks = set()

        # Store sub-tracks for each composite detection
        detection_hyps_states = dict()
        for detection in detections:
            detection_hyps_states[detection] = {'sub-hypotheses': list(), 'sub-states': list()}

        for sub_state_index, sub_initiator in enumerate(self.sub_initiators):

            # Store all sub-detections produced from sub-state index
            sub_state_detections = set()

            # Get all composite detections that have sub-detection for this index
            relevant_detections = dict()

            # Get sub-prior state for this index
            sub_prior = self.prior_state[sub_state_index]

            for detection in detections:
                if sub_state_index in detection.mapping:
                    sub_detection_index = detection.mapping.index(sub_state_index)
                    sub_detection = detection[sub_detection_index]
                    sub_state_detections.add(sub_detection)
                    # link sub-detection back to composite
                    relevant_detections[sub_detection] = detection
                else:
                    # Consider it a missed detection otherwise
                    # Add null hypothesis to its sub-hypotheses list
                    detection_hyps_states[detection]['sub-hypotheses'].append(
                        SingleHypothesis(None, MissedDetection(timestamp=detection.timestamp))
                    )
                    # Add sub-prior to its sub-states list
                    detection_hyps_states[detection]['sub-states'].append(sub_prior)

            if relevant_detections:

                sub_tracks = sub_initiator.initiate(sub_state_detections, timestamp=timestamp)

                while sub_tracks:

                    sub_track = sub_tracks.pop()

                    # Get detection that initiated this sub_track
                    # Expecting single measurement initiation
                    sub_track_detections = {state.hypothesis.measurement
                                            for state in sub_track
                                            if isinstance(state, Update)}

                    if len(sub_track_detections) != 1:
                        # Ambiguity in which detection caused this track
                        # Should not have case where == 0 as this would imply track initiated on
                        # no measurement
                        warnings.warn(
                            "Attempted to initiate sub-track with more than one detection"
                        )

                    sub_track_detection = sub_track_detections.pop()

                    # retrieve composite detection that contains sub-track's detection
                    full_detection = relevant_detections[sub_track_detection]

                    update = sub_track[-1]

                    # Add to sub-hypotheses list for this composite detection
                    detection_hyps_states[full_detection]['sub-hypotheses'].append(
                        SingleHypothesis(None, sub_track_detection)
                    )
                    # Add update to the sub-states list for this composite detection
                    detection_hyps_states[full_detection]['sub-states'].append(update)

        # For each composite detection, create a track from its corresponding sub-hypotheses and
        # sub-states
        for detection, hyps_states in detection_hyps_states.items():
            # Create composite hypothesis from list of sub-hypotheses
            hypothesis = CompositeHypothesis(prediction=None,
                                             sub_hypotheses=hyps_states['sub-hypotheses'],
                                             measurement=detection)
            # Create composite update from list of sub-states
            composite_update = CompositeUpdate(sub_states=hyps_states['sub-states'],
                                               hypothesis=hypothesis)

            tracks.add(Track([composite_update]))
        return tracks

    def __contains__(self, item):
        return self.sub_initiators.__contains__(item)

    def __getitem__(self, index):
        """Can be indexed as a list, or sliced, in which case a new composite initiator will be
        created from the sub-list of sub-initiators."""
        if isinstance(index, slice):
            return self.__class__(self.sub_initiators.__getitem__(index))
        return self.sub_initiators.__getitem__(index)

    def __iter__(self):
        return self.sub_initiators.__iter__()

    def __len__(self):
        return self.sub_initiators.__len__()
