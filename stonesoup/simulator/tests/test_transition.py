from datetime import datetime, timedelta

import numpy as np
import pytest

from ..transition import create_smooth_transition_models, ConstantJerkSimulator
from ...types.array import StateVector
from ...types.state import State

start = datetime.now()
times = [start + timedelta(seconds=10 * i) for i in range(3)]


def get_coords_list():

    x0 = np.zeros(3)
    x1 = np.array([0, 10, 20])
    x2 = -x1
    x3 = np.array([0, 10, 30])
    x4 = -x3
    y0 = x0
    y1 = np.array([0, 0, 20])
    y2 = -y1
    y3 = x1
    y4 = -y3

    coords_list = [(x0, x1, x2, x3, x4), (y0, y1, y2, y3, y4)]
    return [(x, y) for x in coords_list[0] for y in coords_list[1]]


def test_coordinate_length():

    x_coords = np.zeros(3)
    y_coords = np.zeros(2)
    times = [start + timedelta(seconds=10*i) for i in range(3)]

    x = x_coords[0]
    y = y_coords[0]
    vx = (x_coords[1] - x_coords[0]) / 10  # set sensible initial velocity
    vy = (y_coords[1] - y_coords[0]) / 10

    turn_rate = 5

    state_vector = StateVector([x, vx, y, vy])
    target = State(state_vector=state_vector, timestamp=start)

    with pytest.raises(ValueError):
        #  x and y lists not equal in length (x longer)
        create_smooth_transition_models(initial_state=target,
                                        x_coords=x_coords,
                                        y_coords=y_coords,
                                        times=times,
                                        turn_rate=turn_rate)
    with pytest.raises(ValueError):
        #  x and y lists not equal in length (y longer)
        create_smooth_transition_models(initial_state=target,
                                        x_coords=y_coords,
                                        y_coords=x_coords,
                                        times=times,
                                        turn_rate=turn_rate)
    times = [start + timedelta(seconds=10*i) for i in range(4)]
    with pytest.raises(ValueError):
        #  times not equal to coordinates lists
        create_smooth_transition_models(initial_state=target,
                                        x_coords=x_coords,
                                        y_coords=x_coords,
                                        times=times,
                                        turn_rate=turn_rate)


@pytest.mark.parametrize('coords', get_coords_list())
@pytest.mark.parametrize('turn_rate', [5, 10, 45, 90])
def test_transitions_list(coords, turn_rate):

    x_coords = coords[0]
    y_coords = coords[1]

    x = x_coords[0]
    y = y_coords[0]
    vx = (x_coords[1] - x_coords[0]) / 10  # set sensible initial velocity
    vy = (y_coords[1] - y_coords[0]) / 10

    state_vector = StateVector([x, vx, y, vy])
    target = State(state_vector=state_vector, timestamp=start)

    transition_models, transition_times = create_smooth_transition_models(initial_state=target,
                                                                          x_coords=x_coords,
                                                                          y_coords=y_coords,
                                                                          times=times,
                                                                          turn_rate=turn_rate)

    for transition_model in transition_models:
        assert transition_model.ndim_state == 4

    for i in range(len(transition_models)):
        target.state_vector = transition_models[i].function(state=target,
                                                            time_interval=transition_times[i])
        target.timestamp += transition_times[i]
        if target.timestamp in times:
            index = times.index(target.timestamp)
            # expect target to be at next coordinates at each time-step
            assert np.isclose(target.state_vector[0], x_coords[index])
            assert np.isclose(target.state_vector[2], y_coords[index])


def test_constant_jerk():

    start = datetime.now()
    position_mapping = [0, 2]

    # test state dimension disparity error
    with pytest.raises(ValueError,
                       match="Initial and final states must share the same number of dimensions. "
                             "Initial state has ndim = 4 but final state has ndim = 3"):
        ConstantJerkSimulator(position_mapping=position_mapping,
                              velocity_mapping=None,
                              init_state=State([1, 2, 3, 4], start),
                              final_state=State([1, 2, 3], start + timedelta(minutes=1)))

    # test default velocity mapping
    model = ConstantJerkSimulator(position_mapping=position_mapping, velocity_mapping=None,
                                  init_state=State([1, 2, 3, 4], start),
                                  final_state=State([5, 6, 7, 8], start + timedelta(minutes=1)))
    assert model.velocity_mapping == [1, 3]

    for _ in range(5):
        ndim_state = np.random.randint(4, 6)
        init_state = State(100 * np.random.rand(ndim_state, 1), start)
        final_state = State(100 * np.random.rand(ndim_state, 1),
                            start + timedelta(minutes=1))
        if ndim_state > 4:
            for i in range(4, ndim_state):
                # non-kinematic elements stay the same
                final_state.state_vector[i] = init_state.state_vector[i]

        model = ConstantJerkSimulator(position_mapping=position_mapping, velocity_mapping=None,
                                      init_state=init_state, final_state=final_state)

        # test transitioned to correct state
        new_vector = model.function(init_state, timedelta(minutes=1))

        assert np.allclose(new_vector, final_state.state_vector)

    # test create models
    ndim_state = np.random.randint(4, 6)
    states = [State(100 * np.random.rand(ndim_state, 1), start + timedelta(minutes=i))
              for i in range(5)]
    if ndim_state > 4:
        for state in states[1:]:
            for i in range(4, ndim_state):
                # non-kinematic elements stay the same
                state.state_vector[i] = states[0].state_vector[i]

    transition_models, transition_times = \
        ConstantJerkSimulator.create_models(states, position_mapping)

    assert len(transition_models) == len(transition_times) == 4  # one less than len(states)
    for prev_state, next_state, model, duration in zip(states[:-1], states[1:],
                                                       transition_models, transition_times):

        exp_duration = next_state.timestamp - prev_state.timestamp
        assert exp_duration == duration
        assert model.init_state == prev_state
        assert model.final_state == next_state

    current_state = states[0]
    for i, (model, duration) in enumerate(zip(transition_models, transition_times)):
        new_vector = model.function(current_state, duration)
        exp_state = states[i+1]
        assert np.allclose(new_vector, exp_state.state_vector)
        current_state = exp_state
