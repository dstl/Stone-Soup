# -*- coding: utf-8 -*-
import uuid
from typing import MutableSequence, MutableMapping, Sequence

from .state import State, StateMutableSequence, CategoricalState, CompositeState
from ..base import Property


class GroundTruthState(State):
    """Ground Truth State type"""
    metadata: MutableMapping = Property(
        default=None, doc='Dictionary of metadata items for Detections.')

    def __init__(self, state_vector, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)
        if self.metadata is None:
            self.metadata = {}


class CategoricalGroundTruthState(GroundTruthState, CategoricalState):
    """Categorical Ground Truth State type"""


class GroundTruthPath(StateMutableSequence):
    """Ground Truth Path type

    A :class:`~.StateMutableSequence` representing a track.
    """

    states: MutableSequence[GroundTruthState] = Property(
        default=None,
        doc="List of groundtruth states to initialise path with. Default "
            "`None` which initialises with an empty list.")
    id: str = Property(
        default=None,
        doc="The unique path ID. Default `None` where random UUID is "
            "generated.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())


class CompositeGroundTruthState(CompositeState):
    """Composite ground truth state type.

    A composition of ordered sub-states (:class:`GroundTruthState`) existing at the same timestamp,
    representing a true object with a state for (potentially) multiple, distinct state spaces.
    """

    sub_states: Sequence[GroundTruthState] = Property(
        doc="Sequence of sub-states comprising the composite state. All sub-states must have "
            "matching timestamp and `metadata` attributes. Must not be empty.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def metadata(self):
        """Combined metadata of all sub-detections."""
        metadata = dict()
        for sub_state in self.sub_states:
            metadata.update(sub_state.metadata)
        return metadata


GroundTruthState.register(CompositeGroundTruthState)  # noqa: E305
