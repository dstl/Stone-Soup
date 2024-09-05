import argparse
import os
import sys
from os import path as osp

sys.path.insert(0, os.getcwd())

# Ray RLLib imports
from ray import tune
from ray.tune.registry import register_env

# Environment imports
from ReinforcementLearning.utils.rl_utils import env_creator, get_configs


def main(chkp: str, local_dir: str):
    """Restores Tuner after a previously failed run.
    Specifically modified to allow resuming run with a different (or same) environment,
    specified via {new_config_name}.yaml config
    Note: we can not change any parameters using tune.Tuner.restore


    Args:
        new_config_name: config file contaning env config we want to resume from
        config_path: The absolute path where the previous failed run is checkpointed.
            Usually ~/Documents/~/Documents/Github-DL/leonardo-GCAP-S1/logs/ray_runs/{}
        local_dir: Logging directory for RLlib
    """
    # Check if dir exists, else create one
    if os.path.exists(os.getcwd() + "/ReinforcementLearning/logs/"):
        params_dir = osp.abspath(
            osp.join(
                os.getcwd() + "/ReinforcementLearning/logs/ray_runs/" + chkp, "../"
            )
        )
    else:
        os.mkdir(os.getcwd() + "/ReinforcementLearning/logs/")
        params_dir = osp.abspath(
            osp.join(
                os.getcwd() + "/ReinforcementLearning/logs/ray_runs/" + chkp, "../"
            )
        )

    config = get_configs(params_dir)
    param_config = config

    # Create env, might need to close
    env_name = param_config["env"]  # Env name taken from defined config
    env_config = param_config["env_config"]

    # Register custom environment with RLLib
    register_env(
        name=env_name, env_creator=lambda config_dict: env_creator(**env_config)
    )

    local_dir = (
        osp.abspath(osp.join(os.getcwd()))
        + "/ReinforcementLearning/logs/ray_runs/"
        + chkp.split("/")[0]
    )

    tuner = tune.Tuner.restore(
        local_dir,
        trainable="PPO",
        restart_errored=True,
    )

    tuner.fit()


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--chkp",
        type=str,
        help="Path to interrupted experiment. Usually ./logs/ray_runs/{}. \
        Like so: PPO_StoneSoupEnv-v1__06.19.2024_11.17.02/PPO_StoneSoupEnv-v1_1237d_00000_0_2024-06-19_11-17-02/checkpoint_000000 ",
        default="PPO_StoneSoupEnv-v1__07.25.2024_16.32.06/PPO_StoneSoupEnv-v1_0cbb7_00000_0_2024-07-25_16-32-06/checkpoint_000000",
    )
    PARSER.add_argument(
        "--local-dir",
        default="ReinforcementLearning/logs/ray_runs/",
        help="Logging dir for RLlib",
    )
    ARGS = PARSER.parse_args()

    main(**vars(ARGS))  # Convert Namespace -> dict, then pass as kwargs
