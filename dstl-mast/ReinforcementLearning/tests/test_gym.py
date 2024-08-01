import os
import sys

import gymnasium

sys.path.insert(0, os.getcwd())
from ReinforcementLearning.environment import gym


def test_gym(
    gym_env=gym.StoneSoupEnv(
        scenario_config="ReinforcementLearning/configs/scenario_config.yaml"
    ),
):
    """
    Test step function and gym env.

    Takes in gym environment and tests
    functions over multiple steps and
    resets.

    Tests include, existence checks,
    type checks, length checks, and
    range checks.
    """

    # Check spaces exist
    assert gym_env.action_space is not None
    assert gym_env.observation_space is not None

    act = 1

    # Iterate for thorough testing
    for _ in range(10):
        gym_env.reset()
        for _ in range(5):
            obs, reward, terminated, truncated, info = gym_env.step(act)

            test_obs_space(gym_env)
            test_process_obs(gym_env)
            test_action_space(gym_env)
            test_reward(gym_env)
            test_terminate(gym_env)
            test_info(gym_env)

            # Get observation space bounds
            low = gym_env.observation_space.low
            high = gym_env.observation_space.high

            # Check range of observations
            for idx in range(len(obs)):
                assert obs[idx] <= high[idx]
                assert obs[idx] >= low[idx]


def test_process_obs(
    env=gym.StoneSoupEnv(
        scenario_config="ReinforcementLearning/configs/scenario_config.yaml"
    ),
):
    """
    Test processed observations.

    This tests the generated observations
    of the provided scenario. Checking that
    the observations using type checks (list), and
    dynamic length checks.
    """

    # Test 5 times
    for j in range(5):
        act = 1
        obs, reward, terminated, truncated, info = env.step(act)

        # Type test
        assert isinstance(obs, list)

        # Dynamic length test
        assert len(obs) == (
            (
                len(env.scenario_items["targets"])
                + len(env.scenario_items["unknown_targets"])
            )
            * 4
            * 5
        ) + ((len(env.scenario_items["sensor_manager"]) - 1) * 4) + (
            len(env.scenario_items["sensor_manager"]["platform"].sensors)
        ) + (
            len(env.scenario_items["targets"]) * 2
        )


def test_obs_space(
    env=gym.StoneSoupEnv(
        scenario_config="ReinforcementLearning/configs/scenario_config.yaml"
    ),
):
    """
    Test observation space.

    Checks the observation space
    using both type checks (Box) and
    dynamic length checks.
    """

    # Type test
    assert isinstance(env.observation_space, gymnasium.spaces.Box)

    # Dynamic length check
    assert env.observation_space.shape[0] == (
        (
            len(env.scenario_items["targets"])
            + len(env.scenario_items["unknown_targets"])
        )
        * 4
        * 5
    ) + ((len(env.scenario_items["sensor_manager"]) - 1) * 4) + (
        len(env.scenario_items["sensor_manager"]["platform"].sensors)
    ) + (
        len(env.scenario_items["targets"]) * 2
    )


def test_action_space(
    env=gym.StoneSoupEnv(
        scenario_config="ReinforcementLearning/configs/scenario_config.yaml"
    ),
):
    """
    Test action space

    Tests the action space,
    checking to see it it of
    the correct type (Discrete).
    """
    # Type check
    assert isinstance(env.action_space, gymnasium.spaces.Discrete)


def test_reward(
    env=gym.StoneSoupEnv(
        scenario_config="ReinforcementLearning/configs/scenario_config.yaml"
    ),
):
    """
    Test reward calculation

    Checks to make sure reward is
    and acceptable type (int or float).
    """
    # Calculate reward
    reward = env._compute_rewards(env._compute_obs())

    # Type test
    assert isinstance(reward, int) or isinstance(reward, float)


def test_terminate(
    env=gym.StoneSoupEnv(
        scenario_config="ReinforcementLearning/configs/scenario_config.yaml"
    ),
):
    """
    Test terminated value

    Tests to make sure the terminated
    value is of the correct type (bool).
    """

    # Generate terminated value
    terminated = env._compute_done()

    # Type test
    assert isinstance(terminated, bool)


def test_info(
    env=gym.StoneSoupEnv(
        scenario_config="ReinforcementLearning/configs/scenario_config.yaml"
    ),
):
    """
    Tests the info

    Tests to see if info is
    of the expected type (dict).
    """

    act = 0

    # Generate info
    obs, reward, terminated, truncated, info = env.step(act)

    assert isinstance(info, dict)
