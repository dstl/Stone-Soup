import numpy as np

from ..groundtruth import GroundTruthState, GroundTruthPath, CategoricalGroundTruthState, \
    CompositeGroundTruthState

from datetime import datetime


def test_groundtruthpath():
    empty_path = GroundTruthPath()

    assert len(empty_path) == 0

    groundtruth_path = GroundTruthPath([
        GroundTruthState(np.array([[0]])) for _ in range(10)])

    assert len(groundtruth_path) == 10

    state1 = GroundTruthState(np.array([[1]]))
    groundtruth_path.append(state1)
    assert groundtruth_path[-1] is state1
    assert groundtruth_path.states[-1] is state1

    state2 = GroundTruthState(np.array([[2]]))
    groundtruth_path[0] = state2
    assert groundtruth_path[0] is state2

    groundtruth_path.remove(state1)
    assert state1 not in groundtruth_path

    del groundtruth_path[0]
    assert state2 not in groundtruth_path


def test_composite_groundtruth():
    sub_state1 = GroundTruthState([0], metadata={'colour': 'red'})
    sub_state2 = GroundTruthState([1], metadata={'speed': 'fast'})
    sub_state3 = CategoricalGroundTruthState([0.6, 0.4], metadata={'shape': 'square'})
    state = CompositeGroundTruthState(sub_states=[sub_state1, sub_state2, sub_state3])
    assert state.metadata == {'colour': 'red', 'speed': 'fast', 'shape': 'square'}


def test_available_at_time():
    state_1 = GroundTruthState(np.array([[1]]), timestamp=datetime(2023, 3, 28, 16, 54, 1, 868643))
    state_2 = GroundTruthState(np.array([[1]]), timestamp=datetime(2023, 3, 28, 16, 54, 2, 868643))
    state_3 = GroundTruthState(np.array([[1]]), timestamp=datetime(2023, 3, 28, 16, 54, 3, 868643))
    path = GroundTruthPath([state_1, state_2, state_3])
    path_0 = path.available_at_time(datetime(2023, 3, 28, 16, 54, 0, 868643))
    assert len(path_0) == 0
    path_2 = path.available_at_time(datetime(2023, 3, 28, 16, 54, 2, 868643))
    assert len(path_2) == 2
    path_2_1 = path.available_at_time(datetime(2023, 3, 28, 16, 54, 2, 999643))
    assert len(path_2_1) == 2
    path_3 = path.available_at_time(datetime(2023, 3, 28, 16, 54, 20, 868643))
    assert len(path_3) == 3
