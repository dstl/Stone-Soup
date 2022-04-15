# -*- coding: utf-8 -*-
import numpy as np

from ..groundtruth import GroundTruthState, GroundTruthPath, CategoricalGroundTruthState, \
    CompositeGroundTruthState


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
