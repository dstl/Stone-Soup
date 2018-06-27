# -*- coding: utf-8 -*-
import datetime

import pytest
import numpy as np

from ...types import State, GaussianState
from ..simple import SingleTargetGroundTruthSimulator,\
    MultiTargetGroundTruthSimulator


@pytest.fixture(params=[datetime.timedelta(seconds=1),
                        datetime.timedelta(seconds=10),
                        datetime.timedelta(minutes=1)])
def timestep(request):
    return request.param


def test_single_target_ground_truth_simulator(transition_model, timestep):
    initial_state = State(np.array([[1]]), timestamp=datetime.datetime.now())
    simulator = SingleTargetGroundTruthSimulator(transition_model,
                                                 initial_state, timestep)

    for step, (time, groundtruth_paths) in \
            enumerate(simulator.groundtruth_paths_gen()):
        # Check single ground truth track
        assert len(groundtruth_paths) == 1

        # Check length of path is equal to number of steps
        gt_path = groundtruth_paths.pop()
        assert len(gt_path) == step + 1

        # Check time is now + steps
        timedelta = simulator.timestep * step
        assert gt_path[-1].timestamp == initial_state.timestamp + timedelta

        # Check ground truth object has moved
        assert gt_path[-1].state_vector == initial_state.state_vector +\
            timedelta.total_seconds()

    # Check that the number of steps is equal to the simulation steps
    assert step + 1 == simulator.number_steps


def test_multitarget_ground_truth_simulator(transition_model, timestep):
    initial_state = GaussianState(np.array([[1]]), np.array([[0]]),
                                  timestamp=datetime.datetime.now())
    simulator = MultiTargetGroundTruthSimulator(transition_model,
                                                initial_state, timestep)

    total_paths = set()
    for step, (time, groundtruth_paths) in\
            enumerate(simulator.groundtruth_paths_gen()):
        total_paths |= groundtruth_paths

        # Check time is now + steps
        assert time == initial_state.timestamp + simulator.timestep * step

    # Check number of steps is equal to simulation steps
    assert step + 1 == simulator.number_steps

    # Check that there are multiple ground truth paths
    assert len(total_paths) > 1

    # Check that ground truth paths die
    assert len(groundtruth_paths) < len(total_paths)

    # Check that ground truth paths vary in length
    assert len({len(path) for path in total_paths}) > 1
