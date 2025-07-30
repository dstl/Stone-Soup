import pytest
import numpy as np

from ..reward import RewardFunction, AdditiveRewardFunction, MultiplicativeRewardFunction


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
    print(additive(config=None, tracks=None, metric_time=None),
          score_list, output)
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
    print(multiplicative(config=None, tracks=None, metric_time=None),
          score_list, output)
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
