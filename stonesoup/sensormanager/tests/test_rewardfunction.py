import pytest

from ..reward import RewardFunction, AdditiveRewardFunction, MultiplicativeRewardFunction


class DummyRewardFunction(RewardFunction):
    def __init__(self, *args, **kwargs):
        self.score = kwargs["score"]

    def __call__(self, config, tracks, metric_time, *args, **kwargs):
        return self.score


@pytest.mark.parametrize(
        "reward_function, score_list, output",
        [
            (
                DummyRewardFunction,
                [1, 2],
                3
            ),
            (
                DummyRewardFunction,
                [0, -2],
                -2
            )
        ]
)
def test_additive(reward_function, score_list, output):
    additive = AdditiveRewardFunction(
        reward_function_list=[reward_function(score=score) for score in score_list])
    print(additive(config=None, tracks=None, metric_time=None),
          score_list, output)
    assert additive(config=None, tracks=None, metric_time=None) == output


@pytest.mark.parametrize(
        "reward_function, score_list, output",
        [
            (
                DummyRewardFunction,
                [1, 2],
                2
            ),
            (
                DummyRewardFunction,
                [2, 3],
                6
            ),
            (
                DummyRewardFunction,
                [-2, 5],
                -10
            )
        ]
)
def test_multiplicative(reward_function, score_list, output):
    multiplicative = MultiplicativeRewardFunction(
        reward_function_list=[reward_function(score=score) for score in score_list])
    print(multiplicative(config=None, tracks=None, metric_time=None),
          score_list, output)
    assert multiplicative(config=None, tracks=None, metric_time=None) == output
