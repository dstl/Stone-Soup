# -*- coding: utf-8 -*-
from typing import Sequence

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

        for detection in detections:

            mapping = detection.mapping

            hypotheses = list()
            states = list()

            for i, sub_initiator in enumerate(self.sub_initiators):

                try:
                    # Check if detection has sub-detection for this state index
                    detection_index = mapping.index(i)

                    # Get sub-detection and initiate a (sub)track with it
                    sub_detection = detection[detection_index]
                    sub_tracks = sub_initiator.initiate({sub_detection}, timestamp=timestamp)
                    track = sub_tracks.pop()  # Set of 1 track

                except (ValueError, StopIteration):
                    # Either no sub-detection for this index, or sub-initiator could not initiate
                    # from the sub-detection
                    # Instead initiate sub-state as the sub-initiator's prior
                    prior = self._prior_state[i]
                    states.append(prior)

                    # Add missed detection hypothesis to composite hypothesis
                    hypotheses.append(
                        SingleHypothesis(None, MissedDetection(timestamp=detection.timestamp)))
                else:
                    update = track[-1]  # Get first state of track
                    states.append(update)
                    # Add detection hypothesis to composite hypothesis
                    hypotheses.append(SingleHypothesis(None, sub_detection))

            hypothesis = CompositeHypothesis(prediction=None,
                                             sub_hypotheses=hypotheses,
                                             measurement=detection)
            composite_update = CompositeUpdate(sub_states=states,
                                               hypothesis=hypothesis)

            tracks.add(Track([composite_update]))
        return tracks
