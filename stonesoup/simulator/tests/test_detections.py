# -*- coding: utf-8 -*-
import datetime

import pytest
import numpy as np

from ...types import State
from ..simple import SimpleDetectionSimulator, SingleTargetGroundTruthSimulator


@pytest.fixture(params=[datetime.timedelta(seconds=1),
                        datetime.timedelta(seconds=10),
                        datetime.timedelta(minutes=1)])
def timestep(request):
    return request.param


def test_simple_detection_simulator(
        transition_model, measurement_model, timestep):
    initial_state = State(
        np.array([[0], [0], [0], [0]]), timestamp=datetime.datetime.now())
    groundtruth = SingleTargetGroundTruthSimulator(
        transition_model, initial_state, timestep)
    meas_range = np.array([[-1, 1], [-1, 1]]) * 5000
    detector = SimpleDetectionSimulator(
        groundtruth, measurement_model, meas_range)

    total_detections = set()
    clutter_detections = set()
    for step, (time, detections) in enumerate(detector.detections_gen()):
        total_detections |= detections
        clutter_detections |= detector.clutter_detections

        # Check time increments correctly
        assert time == initial_state.timestamp + step * timestep

    # Check both real and clutter detections are generated
    assert len(total_detections) > len(clutter_detections)

    # Check clutter is generated within specified bounds
    for clutter in clutter_detections:
        assert (meas_range[:, 0] <= clutter.state_vector.ravel()).all()
        assert (meas_range[:, 1] >= clutter.state_vector.ravel()).all()
