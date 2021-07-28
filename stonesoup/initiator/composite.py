# -*- coding: utf-8 -*-
from typing import Sequence
import warnings

from .base import Initiator
from ..base import Property
from ..types.detection import MissedDetection
from ..types.hypothesis import SingleHypothesis, CompositeHypothesis
from ..types.state import CompositeState
from ..types.track import Track
from ..types.update import CompositeUpdate


class CompositeUpdateInitiator(Initiator):
    """A composition of initiators, initiating tracks in a composite state space.

    Utilises its sub-initiators to attempt to initiate a track for a :class:`~.CompositeDetection`
    in each corresponding sub-state space, then combines the resultant track states in to a
    :class:`~.CompositeState` track. If any sub-state is missing from the detection, or a
    sub-initiator fails to initiate in its sub-state space, the corresponding sub-state of the
    :attr:`prior_state` will be used instead.
    """
    sub_initiators: Sequence[Initiator] = Property()
    prior_state: CompositeState = Property(
        default=None,
        doc="Prior state information. Defaults to `None`, in which case all sub-initiators must "
            "have their own `prior_state` attribute.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.prior_state is None:
            for sub_initiator in self.sub_initiators:
                try:
                    sub_prior = getattr(sub_initiator, 'prior_state')
                except AttributeError:
                    raise ValueError("If no default prior state is defined, all sub-initiators "
                                     "must have a `prior_state` attribute")
                else:
                    if sub_prior is None:
                        raise ValueError("If no default prior state is defined, all "
                                         "sub-initiators require a defined `prior_state`")

    @property
    def _prior_state(self):
        if self.prior_state:
            return self.prior_state
        else:
            return [sub_initiator.prior_state for sub_initiator in self.sub_initiators]

    def initiate(self, detections, timestamp, **kwargs):
        tracks = set()

        # Store sub-tracks for each detection
        detection_hyps_states = dict()
        for detection in detections:
            detection_hyps_states[detection] = {'sub-hypotheses': list(), 'sub-states': list()}

        for sub_state_index, sub_initiator in enumerate(self.sub_initiators):

            # Store all sub-detections produced from sub-state index
            sub_state_detections = set()

            # Get all detections that have sub-detection for this index
            relevant_detections = dict()

            # Get sub-prior state for this index
            sub_prior = self._prior_state[sub_state_index]

            for detection in detections:
                try:
                    sub_detection_index = detection.mapping.index(sub_state_index)
                except ValueError:
                    # Consider it a missed detection otherwise
                    # Add null hypothesis to its sub-hypotheses list
                    detection_hyps_states[detection]['sub-hypotheses'].append(
                        SingleHypothesis(None, MissedDetection(timestamp=detection.timestamp))
                    )
                    # Add sub-prior to its sub-states list
                    detection_hyps_states[detection]['sub-states'].append(sub_prior)
                else:
                    sub_detection = detection[sub_detection_index]
                    sub_state_detections.add(sub_detection)
                    relevant_detections[sub_detection] = detection

            if relevant_detections:

                sub_tracks = sub_initiator.initiate(sub_state_detections, timestamp=timestamp)

                while sub_tracks:

                    sub_track = sub_tracks.pop()

                    # Get detection that initiated this sub_track
                    sub_track_detections = list()
                    for state in sub_track:
                        try:
                            sub_track_detections.append(state.hypothesis.measurement)
                        except AttributeError:
                            # Must be prediction
                            pass

                    if len(sub_track_detections) != 1:
                        # Ambiguity in which detection caused this track
                        warnings.warn(
                            "Attempted to initiate sub-track with more than one detection"
                        )

                    track_detection = sub_track_detections[-1]

                    full_detection = relevant_detections[track_detection]

                    update = sub_track[-1]

                    # Add to sub-hypotheses list for this detection
                    detection_hyps_states[full_detection]['sub-hypotheses'].append(
                        SingleHypothesis(None, track_detection)
                    )
                    # Add update to the sub-states list for this detection
                    detection_hyps_states[full_detection]['sub-states'].append(update)

        # For each detection, create a track from its corresponding sub-hypotheses and sub-states
        for detection, hyps_states in detection_hyps_states.items():
            hypothesis = CompositeHypothesis(prediction=None,
                                             sub_hypotheses=hyps_states['sub-hypotheses'],
                                             measurement=detection)
            composite_update = CompositeUpdate(sub_states=hyps_states['sub-states'],
                                               hypothesis=hypothesis)

            tracks.add(Track([composite_update]))
        return tracks
