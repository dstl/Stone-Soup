# -*- coding: utf-8 -*-

import datetime

import numpy as np

from ...models.transition.linear import \
    CombinedLinearGaussianTransitionModel, ConstantVelocity
from ...platform.simple import SensorPlatform
from ...types.state import State
from ..platform import PlatformDetectionSimulator
from ..simple import SingleTargetGroundTruthSimulator


def build_platform(sensors, x_velocity):
    state = State(state_vector=[[0], [x_velocity], [0], [0]],
                  timestamp=datetime.datetime(2020, 4, 1))
    model_1d = ConstantVelocity(0.0)  # zero noise so pure movement
    trans_model = CombinedLinearGaussianTransitionModel([model_1d] * 2)
    mounting_offsets = np.zeros((len(sensors), 2))
    mounting_mappings = np.array([[0, 2]])
    platform = SensorPlatform(state=state, sensors=sensors,
                              transition_model=trans_model,
                              mounting_offsets=mounting_offsets,
                              mounting_mappings=mounting_mappings)
    return platform


def test_platform_detection_simulator(sensor_model1,
                                      sensor_model2,
                                      transition_model1):

    platform1 = build_platform([sensor_model1], 1)
    platform2 = build_platform([sensor_model2], 2)

    ground_truth = ((datetime.datetime(2020, 4, 1) +
                     datetime.timedelta(seconds=i),
                     set()) for i in range(10))
    detector = PlatformDetectionSimulator(ground_truth,
                                          [platform1, platform2])

    for n, (time, detections) in enumerate(detector):
        # Detection count at each step.
        assert len(detections) == 1
        # platform1 position.
        assert platform1.state.state_vector[0, 0] == int(n/2)
        # platform2 position.
        assert platform2.state.state_vector[0, 0] == int(n/2)*2

        for detection in detections:
            assert detection.state_vector.shape == (2*(n % 2) + 2, 1)
        for detection in detections:
            if n % 2 == 0:
                # platform1 detects platform2.
                assert detection.state_vector[0, 0] == \
                       platform2.state.state_vector[0, 0]
            else:
                # platform2 detects platform1.
                assert detection.state_vector[0, 0] == \
                       platform1.state.state_vector[0, 0]
        if n > 10:
            break


def test_platform_ground_truth_detection_simulator(sensor_model1,
                                                   sensor_model2,
                                                   transition_model1):

    platform = build_platform([sensor_model1,
                               sensor_model2],
                              1)

    initial_state = State(np.array([[0], [0], [0], [0]]),
                          timestamp=datetime.datetime(2020, 4, 1))
    ground_truth = SingleTargetGroundTruthSimulator(transition_model1,
                                                    initial_state,
                                                    number_steps=10)

    detector = PlatformDetectionSimulator(ground_truth, [platform])

    for n, (time, detections) in enumerate(detector):
        assert len(detections) == 1  # Detection count at each step.
        for detection in detections:
            assert detection.state_vector.shape == (2*(n % 2) + 2, 1)
        for detection in detections:
            for i in range(0, len(detection.state_vector)):
                # Detection at location of ground truth.
                assert int(detection.state_vector[i][0]) == int(n/2)


def test_detection_simulator(sensor_model1,
                             sensor_model2,
                             transition_model1):

    platform1 = build_platform([sensor_model1, sensor_model2], 1)
    platform2 = build_platform([sensor_model1], 2)

    initial_state = State(np.array([[0], [0], [0], [0]]),
                          timestamp=datetime.datetime(2020, 4, 1))
    ground_truth = SingleTargetGroundTruthSimulator(transition_model1,
                                                    initial_state,
                                                    number_steps=10)

    detector = PlatformDetectionSimulator(ground_truth, [platform1, platform2])

    for n, (time, detections) in enumerate(detector):
        assert len(detections) == 2  # Detection count at each step.
        for detection in detections:
            # Detection at location of ground truth or at platform.
            assert int(detection.state_vector[0][0]) in (int(n/3), 2*int(n/3))
