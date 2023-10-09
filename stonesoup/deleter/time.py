"""Contains collection of time based deleters"""
from datetime import timedelta

from ..base import Property
from ..types.update import Update
from .base import Deleter


class UpdateTimeStepsDeleter(Deleter):
    """Update Time based deleter

    Identify tracks for deletion which an :class:`~.Update` has occurred in
    last :attr:`time_steps_since_update`.
    """

    time_steps_since_update: int = Property(doc="Maximum time steps since last update")

    def check_for_deletion(self, track, **kwargs):
        """Delete track without update with measurements within time steps

        Parameters
        ----------
        track : Track
            Track to check for deletion

        Returns
        -------
        bool
            `False` if track has an :class:`~.Update` with measurements within
            time steps; `True` otherwise.
        """
        timestamps_set = set()

        for state in track[::-1]:
            timestamps_set.add(state.timestamp)
            if len(timestamps_set) > self.time_steps_since_update:
                return True

            if isinstance(state, Update) and state.hypothesis:
                return False

        return False


class UpdateTimeDeleter(Deleter):
    """Update Time based deleter

    Identify tracks for deletion which time of last :class:`~.Update` with
    measurements is greater than :attr:`time_since_update`.
    """

    time_since_update: timedelta = Property(doc="Maximum time since last update")

    def check_for_deletion(self, track, timestamp=None, **kwargs):
        """Delete track based on time of last update with measurements

        Parameters
        ----------
        track : Track
            Track to check for deletion
        timestamp : datetime.datetime, optional
            Timestamp to calculate deletion time from. Default `None`, where
            the track's latest timestamp will be used.

        Returns
        -------
        bool
            `False` if track has an :class:`~.Update` with measurements within
            time; `True` otherwise.
        """
        if timestamp is None:
            timestamp = track.timestamp
        return not any(isinstance(state, Update) and state.hypothesis
                       for state in track[timestamp - self.time_since_update:])
