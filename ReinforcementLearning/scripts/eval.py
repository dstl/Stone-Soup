import argparse
import os
import random
import sys

import ray
from tqdm import trange

sys.path.insert(0, os.getcwd())

# Ray imports
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

# ENVIRONMENT IMPORTS
from ReinforcementLearning.environment.gym import StoneSoupEnv

DEFAULT_RENDERING = False
DEAFULT_SCENARIO_CONFIG = "ReinforcementLearning/configs/scenario_config.yaml"
DEFAULT_LOG_DIR = "ReinforcementLearning/logs/renders/"


def main(chkp: str, render: bool):

    env = StoneSoupEnv(
        scenario_config=DEAFULT_SCENARIO_CONFIG,
        render_episodes=render,
        log_dir=DEFAULT_LOG_DIR,
    )

    env2 = StoneSoupEnv(
        scenario_config=DEAFULT_SCENARIO_CONFIG,
        render_episodes=False,
        log_dir=DEFAULT_LOG_DIR,
    )
    # Register custom environment with RLlib
    register_env(name="StoneSoupEnv-v1", env_creator=lambda config_dict: env2)

    # Check if dir exists, else create one
    if os.path.exists(os.getcwd() + "/ReinforcementLearning/logs/"):
        chkp_path = os.getcwd() + "/ReinforcementLearning/logs/ray_runs/" + chkp
    else:
        os.mkdir(os.getcwd() + "/ReinforcementLearning/logs/")
        chkp_path = os.getcwd() + "/ReinforcementLearning/logs/ray_runs/" + chkp

    try:
        agent = Algorithm.from_checkpoint(chkp_path)
    except ValueError:
        print("Agent not found!")
        return False
    except AttributeError:
        print("AttributeError")
        return False

    for ep in range(5):
        trailing_reward = 0
        terminated = truncated = False
        obs, info = env.reset(seed=random.seed(10))
        with trange(100, disable=True) as t:
            for j in t:
                action = agent.compute_single_action(obs)
                if not terminated:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print("Total reward: ", trailing_reward + reward)
                    trailing_reward += reward

                else:
                    print("############# NEXT EPISODE ##############")
                    env.reset()
    env.close()
    ray.shutdown()


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        prog="ray_load_eval", description="Load trained agent", epilog=""
    )
    PARSER.add_argument(
        "--chkp",
        default="PPO_StoneSoupEnv-v1__08.13.2024_16.10.16/PPO_StoneSoupEnv-v1_25e22_00000_0_2024-08-13_16-10-16/checkpoint_000593",
        help="Pass primary chkp directory which can be found in logs/ray_runs/{}",
    )
    PARSER.add_argument(
        "-r", "--render", action="store_true", default=DEFAULT_RENDERING
    )

    ARGS = PARSER.parse_args()

    main(**vars(ARGS))  # Convert Namespace -> dict, then pass as kwargs
