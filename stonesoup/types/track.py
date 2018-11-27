# -*- coding: utf-8 -*-
import uuid

from ..base import Property
from .state import State, StateMutableSequence
from .update import Update


class Track(StateMutableSequence):
    """Track type

    A :class:`~.StateMutableSequence` representing a track.
    """

    states = Property(
        [State],
        default=None,
        doc="The initial states of the track. Default `None` which initialises"
            "with empty list.")

    id = Property(
        str,
        default=None,
        doc="The unique track ID")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())

    @property
    def metadata(self):
        """Returns metadata associated with a track.

        Parameters
        ----------
        None

        Returns
        -------
        : :class:`dict` of variable size
            All metadata associate with this track.
        """

        metadata = {}

        for state in self:
            if isinstance(state, Update) \
                    and state.hypothesis.measurement.metadata is not None:
                metadata.update(state.hypothesis.measurement.metadata)

        return metadata
