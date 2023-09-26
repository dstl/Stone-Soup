import copy
import datetime
import timeit
import warnings
from typing import Union, List, Iterable

import numpy as np
from scipy.interpolate import interp1d

from stonesoup.types.array import StateVector
from stonesoup.types.state import StateMutableSequence, State


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
                                       option: int) -> Union[StateMutableSequence, State]:

    if isinstance(times, datetime.datetime):
        new_sms = interpolate_state_mutable_sequence(sms, [times])
        return new_sms.state

    new_sms = copy.copy(sms)

    # Removes multiple states with the same timestamp. Not needed for most sequences
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

    # Return short-cut method if no interpolation is required
    if set(times).issubset(time_state_dict.keys()):
        # All states are present. No interpolation is required
        new_sms.states = [time_state_dict[time]
                          for time in times]

        return new_sms

    if option == 1:
        return option_1(sms, times)
    elif option == 2:
        return option_2(sms, times)
    elif option == 3:
        return option_3(sms, times)
    elif option == 4:
        return option_4(sms, times)


def option_1(sms: StateMutableSequence, times: List[datetime.datetime]) -> StateMutableSequence:
    new_sms = copy.copy(sms)
    if hasattr(new_sms, "metadatas"):
        new_sms.metadatas = list()
    new_sms.states = list()

    state_iter = iter(sms)
    aft = bef = next(state_iter)

    for timestamp in times:

        # want `bef` to be state just before timestamp, and `aft` to be state just after
        while aft.timestamp < timestamp:
            bef = aft
            while bef.timestamp == aft.timestamp:
                aft = next(state_iter)

        if aft.timestamp == timestamp:
            # if `aft` happens to have timestamp exactly equal, just take its state vector
            sv = aft.state_vector
        else:
            # otherwise, get the point on the line connecting `bef` and `aft`
            # that lies exactly where `timestamp` would be (assuming constant velocity)
            bef_sv = bef.state_vector
            aft_sv = aft.state_vector

            frac = (timestamp - bef.timestamp) / (aft.timestamp - bef.timestamp)

            sv = bef_sv + ((aft_sv - bef_sv) * frac)

        # use `bef` as template for new state (since metadata might change in the future,
        # at `aft`)
        new_state = State.from_state(bef, state_vector=sv, timestamp=timestamp)
        new_state.state_vector = sv
        new_sms.append(new_state)

    return new_sms


def option_2(sms: StateMutableSequence, times: List[datetime.datetime]) -> StateMutableSequence:
    new_sms = copy.copy(sms)
    if hasattr(new_sms, "metadatas"):
        new_sms.metadatas = list()
    new_sms.states = list()

    time_state_dict = {state.timestamp: state
                       for state in sms}

    float_times = [state.timestamp.timestamp() for state in sms]
    float_state_vectors = [[np.double(state.state_vector[i])
                            for state in sms]
                           for i, _ in enumerate(sms[0].state_vector)]

    for timestamp in times:
        if timestamp in time_state_dict:
            new_sms.append(time_state_dict[timestamp])
        else:

            output = np.zeros(len(sms[0].state_vector))
            for i in range(len(output)):
                a_states = float_state_vectors[i]
                output[i] = np.interp(timestamp.timestamp(), float_times, a_states)

            new_sms.append(State(StateVector(output), timestamp=timestamp))

    return new_sms


def option_3(sms: StateMutableSequence, times: List[datetime.datetime]) -> StateMutableSequence:

    new_sms = copy.copy(sms)
    if hasattr(new_sms, "metadatas"):
        new_sms.metadatas = list()

    # Removes multiple states with the same timestamp. Not needed for most sequences
    time_state_dict = {state.timestamp: state
                       for state in sms}

    svs = [state.state_vector for state in time_state_dict.values()]
    state_times = [time.timestamp() for time in time_state_dict.keys()]
    interpolate_object = interp1d(state_times, np.stack(svs, axis=0), axis=0)

    interp_output = interpolate_object([time.timestamp() for time in times])

    new_sms.states = [State(interp_output[idx, :, :], timestamp=time)
                      for idx, time in enumerate(times)]

    return new_sms


def option_4(sms: StateMutableSequence, times: List[datetime.datetime]) -> StateMutableSequence:
    new_sms = copy.copy(sms)
    if hasattr(new_sms, "metadatas"):
        new_sms.metadatas = list()

    # Removes multiple states with the same timestamp. Not needed for most sequences
    time_state_dict = {state.timestamp: state
                       for state in sms}

    times_to_interpolate = sorted(set(times).difference(time_state_dict.keys()))

    svs = [state.state_vector for state in time_state_dict.values()]
    state_times = [time.timestamp() for time in time_state_dict.keys()]
    interpolate_object = interp1d(state_times, np.stack(svs, axis=0), axis=0)

    interp_output = interpolate_object([time.timestamp() for time in times_to_interpolate])

    for idx, time in enumerate(times_to_interpolate):
        time_state_dict[time] = State(interp_output[idx, :, :], timestamp=time)

    new_sms.states = [time_state_dict[time] for time in times]

    return new_sms


def state_vector_fun(time: datetime.datetime) -> State:
    today = datetime.datetime(datetime.datetime.today().year,
                              datetime.datetime.today().month,
                              datetime.datetime.today().day)
    num = (time-today).seconds
    return State(
        timestamp=time,
        state_vector=[
            num*1.3e-4,
            num*4.7e-5,
            num/43,
            # np.sin(num*1e-6)
        ]
    )


def gen_test_data():
    t0 = datetime.datetime.today()
    max_seconds = 1e3
    t_max = t0 + datetime.timedelta(seconds=max_seconds)

    sms = StateMutableSequence([state_vector_fun(time)
                                for time in time_range(t0, t_max, datetime.timedelta(seconds=0.25))
                                ])

    interp_times = list(time_range(t0, t_max, datetime.timedelta(seconds=0.1)))

    return sms, interp_times


def individual_speed_test(i):
    sms, interp_times = gen_test_data()
    interpolate_state_mutable_sequence(sms, interp_times, i)


def run_all_speed_tests():
    for i in [1, 2, 3, 4]:
        time_taken = timeit.timeit(setup="from __main__ import individual_speed_test",
                                   stmt=f"individual_speed_test({i})", number=10)
        print(f"Option {i} took {time_taken}")


def check_correct():
    sms, interp_times = gen_test_data()
    for i in [1, 2, 3, 4]:
        new_sms = interpolate_state_mutable_sequence(sms, interp_times, i)
        for state in new_sms:
            interp_sv = state.state_vector
            true_sv = state_vector_fun(state.timestamp).state_vector
            if not np.all(np.isclose(interp_sv, true_sv, rtol=1e-3)):
                print(f"Option {i} failed")
                break
        print(f"Option {i} worked")


if __name__ == '__main__':
    run_all_speed_tests()
    check_correct()
