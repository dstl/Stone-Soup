# coding: utf-8

from datetime import datetime, timedelta

import numpy as np
import pytest

from stonesoup.simulator.transition import create_smooth_transition_models
from stonesoup.types.state import State
from stonesoup.types.array import StateVector

start = datetime.now()
times = [start + timedelta(seconds=10*i) for i in range(3)]


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
