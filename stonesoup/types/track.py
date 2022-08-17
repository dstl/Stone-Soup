import copy
import uuid
from typing import MutableSequence, MutableMapping

from .multihypothesis import MultipleHypothesis
from .state import State, StateMutableSequence
from .update import Update
from ..base import Property


class Track(StateMutableSequence):
    """Track type

    A :class:`~.StateMutableSequence` representing a track.

    Notes:
        Any manual modifications to :attr:`metadata` or :attr:`metadatas` will be overwritten if a
        state is inserted at a point prior to where the modifications are made.
        For example, inserting a state at the start of :attr:`states` will result in a
        :attr:`metadatas` update that will update all subsequent metadata values, resulting in
        manual metadata modifications being lost.
    """

    states: MutableSequence[State] = Property(
        default=None,
        doc="The initial states of the track. Default `None` which initialises with empty list.")

    id: str = Property(default=None, doc="The unique track ID")

    init_metadata: MutableMapping = Property(
        default={}, doc="Initial dictionary of metadata items for track. Default `None` which "
                        "initialises track metadata as an empty dictionary.")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.metadatas = list()

        for state in self.states:
            self._update_metadata_from_state(state)
        if self.id is None:
            self.id = str(uuid.uuid4())

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        if index < 0:
            index = len(self.states) + index
        self._update_metadatas(index)

    def __copy__(self):
        inst = super().__copy__()
        inst.__dict__['metadatas'] = copy.copy(self.__dict__['metadatas'])
        return inst

    def insert(self, index, value):
        """Insert value at index of :attr:`states`.

        Parameters
        ----------
        index: int
            Index of :attr:`states` to insert value at.
        value: State
            A state object to be inserted at the specified index of :attr:`states`.
        """
        super().insert(index, value)

        if index < 0:
            if index < -len(self.states):
                index = 0
            else:
                index += len(self.states) - 1
        elif index >= len(self.states):
            index = len(self.states) - 1
        self._update_metadatas(index)

    def append(self, value):
        """Add value at end of :attr:`states`.

        Parameters
        ----------
        value: State
            A state object to be added at the end of :attr:`states`.
        """
        # Update metadata
        self._update_metadata_from_state(value)
        return self.states.append(value)

    @property
    def metadata(self):
        """Current metadata dictionary of track. If track contains no states, this is the initial
        metadata dictionary :attr:`init_metadata`."""
        if self.metadatas:
            return self.metadatas[-1]
        else:
            return self.init_metadata

    def _update_metadatas(self, index):
        """Update track :attr:`metadatas` property, starting at specified index.

        Parameters
        ----------
        index: Int
            Index of :attr:`metadatas` to update from.
        """
        # Plus one for 0th initial track meta data
        self.metadatas = self.metadatas[:index]

        for future_state in self.states[index:]:
            self._update_metadata_from_state(future_state)

    def _update_metadata_from_state(self, state):
        """Update :attr:`metadatas` with an updated metadata entry, accounting for extracted
        metadata from state.

        Parameters
        ----------
        state: State
            A state object from which to extract metadata. Metadata can only be extracted from
            Update (or subclassed) objects. Calling this method with a non-Update (subclass) object
            will NOT raise an error, but will have no effect on the metadata.
        """
        self.metadatas.append(self.metadata.copy())

        if isinstance(state, Update):
            if isinstance(state.hypothesis, MultipleHypothesis):
                # Sort and iterate through multiple hypotheses such that most
                # likely hypothesis comes last. This ensures that metadata
                # from all hypotheses are retained, but more likely
                # hypotheses will over-write the metadata set by less likely
                # ones.
                for hypothesis in sorted(state.hypothesis, reverse=True):
                    if hypothesis \
                            and hypothesis.measurement.metadata is not None:
                        self.metadata.update(hypothesis.measurement.metadata)
            else:
                hypothesis = state.hypothesis
                if hypothesis and hypothesis.measurement.metadata is not None:
                    self.metadata.update(hypothesis.measurement.metadata)
