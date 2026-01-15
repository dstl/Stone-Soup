import pytest
from datetime import datetime
import numpy as np

from ..reward import (RewardFunction, AdditiveRewardFunction, MultiplicativeRewardFunction,
                      FOVInteractionRewardFunction)

from stonesoup.base import Property
from stonesoup.types.state import State
from stonesoup.types.track import Track
from stonesoup.sensor.sensor import Sensor


class DummyRewardFunction(RewardFunction):
    def __init__(self, *args, **kwargs):
        self.score = kwargs["score"]

    def __call__(self, config, tracks, metric_time, *args, **kwargs):
        return self.score


class DummyUpdater:
    pass


class DummySensor(Sensor):
    position: State = Property()

    def add_actions(self, actions):
        pass

    def act(self, metric_time, noise=False):
        pass

    def measure(self, ground_truths, noise, **kwargs):
        pass

    def measurement_model(self):
        pass


class DummyAction:
    pass


class DummyPredictor:
    def predict(self, track, timestamp=None):
        # Just return a dummy state (as a list or object)
        return State([10, 0, 10])


@pytest.mark.parametrize(
        "reward_function, score_list, weights, output",
        [
            (
                DummyRewardFunction,
                [1, 2],
                None,
                3
            ),
            (
                DummyRewardFunction,
                [0, -2],
                None,
                -2
            ),
            (
                DummyRewardFunction,
                [0, -2],
                [0.5, 0.5],
                -1
            ),
            (
                DummyRewardFunction,
                [3, 2],
                [0.4, 0.6],
                2.4
            ),
            (
                DummyRewardFunction,
                [3, 2, 1],
                [0.5, 0.4, 0.1],
                2.4
            )
        ]
)
def test_additive(reward_function, score_list, weights, output):
    additive = AdditiveRewardFunction(
        reward_function_list=[reward_function(score=score) for score in score_list],
        weights=weights)
    assert np.allclose(additive(config=None, tracks=None, metric_time=None), output)


@pytest.mark.parametrize(
        "reward_function, score_list, weights, output",
        [
            (
                DummyRewardFunction,
                [1, 2],
                None,
                2
            ),
            (
                DummyRewardFunction,
                [2, 3],
                [1, 1],
                6
            ),
            (
                DummyRewardFunction,
                [-2, 5],
                [0.5, 0.5],
                -2.5
            )
        ]
)
def test_multiplicative(reward_function, score_list, weights, output):
    multiplicative = MultiplicativeRewardFunction(
        reward_function_list=[reward_function(score=score) for score in score_list],
        weights=weights)
    assert np.allclose(multiplicative(config=None, tracks=None, metric_time=None), output)


def test_unequal_multiplicative():
    multiplicative = MultiplicativeRewardFunction(
        reward_function_list=[DummyRewardFunction(score=1), DummyRewardFunction(score=2)],
        weights=[1, 2, 3])
    with pytest.raises(IndexError):
        multiplicative(config=None, tracks=None, metric_time=None)


def test_unequal_additive():
    additive = AdditiveRewardFunction(
        reward_function_list=[DummyRewardFunction(score=1), DummyRewardFunction(score=2)],
        weights=[1, 2, 3])
    with pytest.raises(IndexError):
        additive(config=None, tracks=None, metric_time=None)


@pytest.fixture
def fov_reward():
    return FOVInteractionRewardFunction(
        predictor=DummyPredictor(),
        updater=DummyUpdater(),
        sensor_fov_radius=20.0,
        target_fov_radius=10.0,
        sensor_mapping=[0, 2],
        target_mapping=[0, 2])


def test_target_in_sensor_fov_and_not_in_target_fov(reward_function=fov_reward):
    sensor = DummySensor(State([10, 0, 10, 0]))
    config = {sensor: [DummyAction()]}
    tracks = {Track([10, 0, 10, 0])}
    metric_time = datetime.now()
    reward = reward_function(config, tracks, metric_time)
    assert reward == -1.0


def test_target_outside_sensor_fov(reward_function=fov_reward):
    sensor = DummySensor(State([100, 0, 100, 0]))
    config = {sensor: [DummyAction()]}
    tracks = {Track([0, 0, 0, 0])}
    metric_time = datetime.now()
    reward = reward_function(config, tracks, metric_time)
    assert reward == -1.0


def test_target_in_sensor_fov_and_in_target_fov(reward_function=fov_reward):
    sensor = DummySensor(State([9, 0, 9, 0]))
    config = {sensor: [DummyAction()]}
    tracks = {Track([10, 0, 10, 0])}
    metric_time = datetime.now()
    reward = reward_function(config, tracks, metric_time)
    assert reward == -1.0


def test_target_in_sensor_fov_but_not_in_target_fov(reward_function=fov_reward):
    sensor = DummySensor(State([0, 0, 0, 0]))
    config = {sensor: [DummyAction()]}
    tracks = {Track([15, 0, 0, 0])}
    metric_time = datetime.now()
    reward = reward_function(config, tracks, metric_time)
    assert reward == 1.0
