# -*- coding: utf-8 -*-
import datetime

import pytest
import numpy as np

from ...types.state import State
from ..simple import SimpleDetectionSimulator, SwitchDetectionSimulator,\
    SingleTargetGroundTruthSimulator, SwitchOneTargetGroundTruthSimulator


@pytest.fixture(params=[datetime.timedelta(seconds=1),
                        datetime.timedelta(seconds=10),
                        datetime.timedelta(minutes=1)])
def timestep(request):
    return request.param


def test_simple_detection_simulator(
        transition_model1, measurement_model, timestep):
    initial_state = State(
        np.array([[0], [0], [0], [0]]), timestamp=datetime.datetime.now())
    groundtruth = SingleTargetGroundTruthSimulator(
        transition_model1, initial_state, timestep)
    meas_range = np.array([[-1, 1], [-1, 1]]) * 5000
    detector = SimpleDetectionSimulator(
        groundtruth, measurement_model, meas_range, clutter_rate=3)

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

    assert detector.clutter_spatial_density == 3e-8


def test_switch_detection_simulator(
        transition_model1, transition_model2, measurement_model, timestep):
    initial_state = State(
        np.array([[0], [0], [0], [0]]), timestamp=datetime.datetime.now())
    model_probs = [[0.5, 0.5], [0.5, 0.5]]
    groundtruth = SwitchOneTargetGroundTruthSimulator(
        transition_models=[transition_model1, transition_model2],
        model_probs=model_probs,
        initial_state=initial_state,
        timestep=timestep)
    meas_range = np.array([[-1, 1], [-1, 1]]) * 5000

    detector = SwitchDetectionSimulator(
        groundtruth, measurement_model, meas_range, clutter_rate=3,
        detection_probabilities=[0, 1])

    test_detector = SimpleDetectionSimulator(
        groundtruth, measurement_model, meas_range, clutter_rate=3,
        detection_probability=1
    )

    total_detections = set()
    clutter_detections = set()
    for step, (time, detections) in enumerate(detector.detections_gen()):
        total_detections |= detections
        clutter_detections |= detector.clutter_detections

        # Check time increments correctly
        assert time == initial_state.timestamp + step * timestep

    test_detections = set()
    for step, (time, detections) in enumerate(test_detector.detections_gen()):
        test_detections |= detections

    # Check both real and clutter detections are generated
    assert len(total_detections) > len(clutter_detections)

    # Check clutter is generated within specified bounds
    for clutter in clutter_detections:
        assert (meas_range[:, 0] <= clutter.state_vector.ravel()).all()
        assert (meas_range[:, 1] >= clutter.state_vector.ravel()).all()

    assert detector.clutter_spatial_density == 3e-8

    assert len(total_detections) < len(test_detections)
