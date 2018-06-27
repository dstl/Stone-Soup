# -*- coding: utf-8 -*-
import numpy as np

from ..groundtruth import GroundTruthState, GroundTruthPath


def test_groundtruthpath():
    empty_path = GroundTruthPath()

    assert len(empty_path) == 0

    groundtruth_path = GroundTruthPath([
        GroundTruthState(np.array([[0]])) for _ in range(10)])

    assert len(groundtruth_path) == 10

    state1 = GroundTruthState(np.array([[1]]))
    groundtruth_path.append(state1)
    assert groundtruth_path[-1] is state1

    state2 = GroundTruthState(np.array([[2]]))
    groundtruth_path[0] = state2
    assert groundtruth_path[0] is state2

    groundtruth_path.remove(state1)
    assert state1 not in groundtruth_path

    del groundtruth_path[0]
    assert state2 not in groundtruth_path
