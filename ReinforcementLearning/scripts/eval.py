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

    # Register custom environment with RLlib
    register_env(name="StoneSoupEnv-v1", env_creator=lambda config_dict: env)

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

    for ep in range(100):
        terminated = truncated = False
        obs, info = env.reset(seed=random.seed(10))
        with trange(500, disable=True) as t:
            for j in t:
                action = agent.compute_single_action(obs)
                print("Actions from trained policy: ", action)
                if not terminated:
                    obs, reward, terminated, truncated, info = env.step(action)
                else:
                    print("############# NEXT EPISODE ##############")
                    env.reset()

    ray.shutdown()


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        prog="ray_load_eval", description="Load trained agent", epilog=""
    )
    PARSER.add_argument(
        "--chkp",
        default="PPO_StoneSoupEnv-v1__06.27.2024_00.30.45/PPO_StoneSoupEnv-v1_8166c9e5_0_2024-06-27_09-14-37/checkpoint_000565",
        help="Pass primary chkp directory which can be found in logs/ray_runs/{}",
    )
    PARSER.add_argument(
        "-r", "--render", action="store_true", default=DEFAULT_RENDERING
    )

    ARGS = PARSER.parse_args()

    main(**vars(ARGS))  # Convert Namespace -> dict, then pass as kwargs
