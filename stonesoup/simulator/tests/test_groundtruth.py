import datetime

import pytest
import numpy as np

from ...types.state import GaussianState, State
from ..simple import (
    SingleTargetGroundTruthSimulator, MultiTargetGroundTruthSimulator,
    SwitchOneTargetGroundTruthSimulator, SwitchMultiTargetGroundTruthSimulator)


@pytest.fixture(params=[datetime.timedelta(seconds=1),
                        datetime.timedelta(seconds=10),
                        datetime.timedelta(minutes=1)])
def timestep(request):
    return request.param


def test_single_target_ground_truth_simulator(transition_model1, timestep):
    initial_state = State(np.array([[1]]), timestamp=datetime.datetime.now())
    simulator = SingleTargetGroundTruthSimulator(transition_model1,
                                                 initial_state, timestep)

    for step, (time, groundtruth_paths) in enumerate(simulator):
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


def test_multitarget_ground_truth_simulator(transition_model1, timestep):
    initial_state = GaussianState(np.array([[1]]), np.array([[0]]),
                                  timestamp=datetime.datetime.now())
    simulator = MultiTargetGroundTruthSimulator(transition_model1,
                                                initial_state, timestep)

    total_paths = set()
    for step, (time, groundtruth_paths) in enumerate(simulator):
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

    # Check random seed gives consistent results
    simulator1 = MultiTargetGroundTruthSimulator(transition_model1,
                                                 initial_state, timestep,
                                                 seed=1)
    simulator2 = MultiTargetGroundTruthSimulator(transition_model1,
                                                 initial_state, timestep,
                                                 seed=1)

    for (_, truth1), (_, truth2) in zip(simulator1, simulator2):
        state_vectors1 = tuple(tuple(gt.state_vector) for gt in truth1)
        state_vectors2 = tuple(tuple(gt.state_vector) for gt in truth2)
        for sv in state_vectors1:
            assert sv in state_vectors2


def test_one_target_ground_truth_simulator_switch(transition_model1,
                                                  transition_model2,
                                                  timestep):
    initial_state = State(np.array([[1]]), timestamp=datetime.datetime.now())
    model_probs = [[0.5, 0.5], [0.5, 0.5]]
    simulator = SwitchOneTargetGroundTruthSimulator(
        transition_models=[transition_model1, transition_model2],
        model_probs=model_probs,
        initial_state=initial_state,
        timestep=timestep)

    for step, (time, groundtruth_paths) in enumerate(simulator):
        # Check single ground truth track
        assert len(groundtruth_paths) == 1

        # Check length of path is equal to number of steps
        gt_path = groundtruth_paths.pop()
        assert len(gt_path) == step + 1

        # Check time is now + steps
        timedelta = simulator.timestep * step
        assert gt_path[-1].timestamp == initial_state.timestamp + timedelta

        record = []
        for state in gt_path:
            record.append(state.metadata.get("index")+1)
        total = sum(record[1:])

        # Check ground truth object has moved
        assert gt_path[-1].state_vector == initial_state.state_vector +\
            timestep.total_seconds()*total

    # Check that the number of steps is equal to the simulation steps
    assert step + 1 == simulator.number_steps

    # Check random seed gives consistent results
    simulator1 = SwitchOneTargetGroundTruthSimulator(
        transition_models=[transition_model1, transition_model2],
        model_probs=model_probs,
        initial_state=initial_state,
        timestep=timestep,
        seed=1)
    simulator2 = SwitchOneTargetGroundTruthSimulator(
        transition_models=[transition_model1, transition_model2],
        model_probs=model_probs,
        initial_state=initial_state,
        timestep=timestep,
        seed=1)

    for (_, truth1), (_, truth2) in zip(simulator1, simulator2):
        state_vectors1 = tuple(tuple(gt.state_vector) for gt in truth1)
        state_vectors2 = tuple(tuple(gt.state_vector) for gt in truth2)
        for sv in state_vectors1:
            assert sv in state_vectors2


def test_multitarget_ground_truth_simulator_witch(transition_model1,
                                                  transition_model2,
                                                  timestep):
    initial_state = GaussianState(np.array([[1]]), np.array([[0]]),
                                  timestamp=datetime.datetime.now())
    model_probs = [[0.5, 0.5], [0.5, 0.5]]
    simulator = SwitchMultiTargetGroundTruthSimulator(
        transition_models=[transition_model1, transition_model2],
        model_probs=model_probs,
        initial_state=initial_state,
        timestep=timestep)

    total_paths = set()
    for step, (time, groundtruth_paths) in enumerate(simulator):
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

    for path in total_paths:
        indices = []
        for state in path:
            indices.append(state.metadata.get("index"))
        if len(path) > 9:
            assert indices.count(1) < len(path)

    # Check random seed gives consistent results
    simulator1 = SwitchMultiTargetGroundTruthSimulator(
        transition_models=[transition_model1, transition_model2],
        model_probs=model_probs,
        initial_state=initial_state,
        timestep=timestep,
        seed=1)
    simulator2 = SwitchMultiTargetGroundTruthSimulator(
        transition_models=[transition_model1, transition_model2],
        model_probs=model_probs,
        initial_state=initial_state,
        timestep=timestep,
        seed=1)

    for (_, truth1), (_, truth2) in zip(simulator1, simulator2):
        state_vectors1 = tuple(tuple(gt.state_vector) for gt in truth1)
        state_vectors2 = tuple(tuple(gt.state_vector) for gt in truth2)
        for sv in state_vectors1:
            assert sv in state_vectors2
