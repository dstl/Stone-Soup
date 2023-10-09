import copy
import datetime
import warnings
from typing import Union, List, Iterable

import numpy as np

from ..types.array import StateVectors
from ..types.state import StateMutableSequence, State


def time_range(start_time: datetime.datetime, end_time: datetime.datetime,
               timestep: datetime.timedelta = datetime.timedelta(seconds=1)) \
        -> Iterable[datetime.datetime]:
    """
    Produces a range of datetime object between ``start_time`` (inclusive) and ``end_time``
    (inclusive)

    Parameters
    ----------
    start_time: datetime.datetime   time range start (inclusive)
    end_time: datetime.datetime     time range end (inclusive)
    timestep: datetime.timedelta    default value is 1 second

    Returns
    -------
    Generator[datetime.datetime]

    """
    duration = end_time - start_time
    n_time_steps = duration / timestep
    for x in range(int(n_time_steps) + 1):
        yield start_time + x * timestep


def interpolate_state_mutable_sequence(sms: StateMutableSequence,
                                       times: Union[datetime.datetime, List[datetime.datetime]],
                                       ) -> Union[StateMutableSequence, State]:
    """
    This function performs linear interpolation on a :class:`~.StateMutableSequence`. The function
    has two slightly different forms:

    If an individual :class:`~datetime.datetime` is inputted for the variable ``times`` then a
    :class:`~.State` is returned corresponding to ``times``.

    If a list of :class:`~datetime.datetime` is inputted for the variable ``times`` then a
    :class:`~.StateMutableSequence` is returned with the states in the sequence corresponding to
    ``times``.

    Note
    ----
    This function does **not** extrapolate. Times outside the range of the time range of ``sms``
    are discarded and warning is given. If all ``times`` values are outside the time range of
    ``sms`` then an ``IndexError`` is raised.

    Unique states for each time are required for interpolation. If there are multiple states with
    the same time in ``sms`` the later state in the sequence is used.

    For :class:`~.Track` inputs the *metadatas* is removed as it can't be interpolated.
    """

    # If single time is used, insert time into list and run again.
    # A StateMutableSequence is produced by the inner function call.
    # The single state is taken from that StateMutableSequence
    if isinstance(times, datetime.datetime):
        new_sms = interpolate_state_mutable_sequence(sms, [times])
        return new_sms.state

    # Track metadata removed and no interpolation can be performed on the metadata
    new_sms = copy.copy(sms)
    if hasattr(new_sms, "metadatas"):
        new_sms.metadatas = list()

    # This step ensure unique states for each timestamp. The last state for a timestamp is used
    # with earlier states not being used.
    time_state_dict = {state.timestamp: state
                       for state in sms}

    # Filter times if required
    max_state_time = sms[-1].timestamp
    min_state_time = sms[0].timestamp
    if max(times) > max_state_time or min(times) < min_state_time:
        new_times = [time
                     for time in times
                     if min_state_time <= time <= max_state_time]

        if len(new_times) == 0:
            raise IndexError(f"All times are outside of the state mutable sequence's time range "
                             f"({min_state_time} -> {max_state_time})")

        removed_times = set(times).difference(new_times)
        warnings.warn(f"Trying to interpolate states which are outside the time range "
                      f"({min_state_time} -> {max_state_time}) of the state mutable sequence. The "
                      f"following times aren't included in the output {removed_times}")

        times = new_times

    # Find times that require interpolation
    times_to_interpolate = sorted(set(times).difference(time_state_dict.keys()))

    if len(times_to_interpolate) > 0:
        # Only interpolate if required
        state_vectors = StateVectors([state.state_vector for state in time_state_dict.values()])
        state_timestamps = [time.timestamp() for time in time_state_dict.keys()]
        interp_timestamps = [time.timestamp() for time in times_to_interpolate]

        interp_output = np.empty((sms.state.ndim, len(times_to_interpolate)))
        for element_index in range(sms.state.ndim):
            interp_output[element_index, :] = np.interp(x=interp_timestamps,
                                                        xp=state_timestamps,
                                                        fp=state_vectors[element_index, :])

        for state_index, time in enumerate(times_to_interpolate):
            time_state_dict[time] = State(interp_output[:, state_index], timestamp=time)

    new_sms.states = [time_state_dict[time] for time in times]

    return new_sms
