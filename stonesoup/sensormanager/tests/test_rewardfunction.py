import pytest
import numpy as np

from ...types.shape import AreaOfInterest
from ..reward import (RewardFunction, AdditiveRewardFunction,
                      MultiplicativeRewardFunction, AOIRewardFunction2D)


class DummyRewardFunction(RewardFunction):
    def __init__(self, *args, **kwargs):
        self.score = kwargs["score"]

    def __call__(self, config, tracks, metric_time, *args, **kwargs):
        return self.score


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


class DummyTrack:
    def __init__(self, state_vector):
        self.state_vector = state_vector


@pytest.mark.parametrize(
    "area_kwargs, interest_thresholds, access_thresholds, default_score, expected_score",
    [
        (
            {"xmin": 0.0, "xmax": 10.0, "ymin": 0.0, "ymax": 10.0,
             "interest": 5, "access": 4},
            {1: DummyRewardFunction(score=10)},
            {1: DummyRewardFunction(score=20)},
            1,
            30,
        ),
        (
            {"xmin": 0.0, "xmax": 10.0, "ymin": 0.0, "ymax": 10.0,
             "interest": 5, "access": 0},
            {1: DummyRewardFunction(score=7)},
            None,
            1,
            7,
        ),
        (
            {"xmin": 0.0, "xmax": 10.0, "ymin": 0.0, "ymax": 10.0,
             "interest": 0, "access": 5},
            None,
            {1: DummyRewardFunction(score=11)},
            1,
            11,
        ),
        (
            {"xmin": 0.0, "xmax": 10.0, "ymin": 0.0, "ymax": 10.0,
             "interest": 0, "access": 0},
            {1: DummyRewardFunction(score=7)},
            {1: DummyRewardFunction(score=11)},
            2,
            2,
        ),
    ],
)
def test_aoi_reward2d_threshold_matching(area_kwargs, interest_thresholds,
                                         access_thresholds, default_score,
                                         expected_score):
    area = AreaOfInterest(**area_kwargs)
    aoi_reward = AOIRewardFunction2D(
        interest_thresholds=interest_thresholds,
        access_thresholds=access_thresholds,
        default_reward=DummyRewardFunction(score=default_score),
        areas=[area],
        target_mapping=(0, 1),
    )

    assert aoi_reward(config=None, tracks={DummyTrack([5.0, 5.0])}, metric_time=None) == expected_score  # noqa: E501
