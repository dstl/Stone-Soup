import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from hyperopt import hp
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

sys.path.insert(0, os.getcwd())

import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.tune.registry import register_env

from ReinforcementLearning.scripts.callbacks import CustomCallback
from ReinforcementLearning.utils.rl_utils import env_creator, get_configs


def main(
    config_path: str,
    local_dir: str,
    workers: int,
    add_to_name: str,
    use_tune,
    logging_level: str,
):
    """Read in configs and perform a `tune.run()`

    Args:
        config_path: Path to directory containing.
        local_dir: Logging dir for RLlib.
        workers: Number of rollout workers to use while training
        add_to_name: Additional info to add to file name
        use_tune: True if we want hyperparameter tuning. Default is false
        logging_level: Level of logging for ray to perform. E.g. DEBUG, INFO, WARN
    """
    # Initialise ray session with logging level etc.
    ray.init(configure_logging=True, logging_level=logging_level)

    config: dict = get_configs(config_path)

    # Configure tune settings if use_tune set to True
    if use_tune:
        asha_scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="episode_reward_mean",
            mode="max",
            max_t=100,
            grace_period=10,
            reduction_factor=3,
            brackets=1,
        )

        space = {
            "lr": hp.loguniform("lr", 1.0e-6, 1.0e-3),
            "gamma": hp.uniform("gamma", 0.9, 0.999),
        }

        hyperopt_search = HyperOptSearch(
            space, metric="episode_reward_mean", mode="max"
        )

        config.update(
            {
                "tune_config": {
                    "search_alg": hyperopt_search,
                    "scheduler": asha_scheduler,
                    "num_samples": 5,
                }
            }
        )

        tune_config = config.get("tune_config", None)
        tune_config = tune.TuneConfig(**tune_config)
    else:
        tune_config = None

    param_config = config["param_config"]
    run_config = config["run_config"]
    chkp_config = run_config["checkpoint_config"]

    # Create env, might need to close
    env_name = param_config["env"]  # Env name taken from defined config
    env_config = param_config["env_config"]

    # Update number of rollout workers to be used if specified
    if workers:
        param_config.update({"num_rollout_workers": workers})

    # Register custom environment with RLlib
    register_env(
        name=env_name, env_creator=lambda config_dict: env_creator(**env_config)
    )

    param_config.update(
        {
            "env_config": param_config["env_config"],
            "callbacks": CustomCallback,  # change callback if needed
            "policy_mapping_fn": getattr(AlgorithmConfig, "DEFAULT_POLICY_MAPPING_FN"),
        }
    )

    # pprint(config, sort_dicts=False)

    run_config.update(
        {
            "name": config["trainable"]
            + "_"
            + param_config["env"]
            + "_"
            + add_to_name
            + "_"
            + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
            "storage_path": Path(
                local_dir
            ).resolve(),  # as Ray >2.7.0 only supports absolute paths
            "checkpoint_config": air.CheckpointConfig(**chkp_config),
        }
    )

    # Initiate training run
    tuner = tune.Tuner(
        config.get("trainable", "PPO"),
        param_space=param_config,
        run_config=air.RunConfig(**run_config),
        tune_config=tune_config,
    )

    results = tuner.fit()

    checkpoints = results.get_best_result().checkpoint
    with open(local_dir + run_config["name"] + "/checkpoint.txt", "w+") as f:
        f.write(checkpoints[0][0])

    ray.shutdown()


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--config-path",
        default="ReinforcementLearning/configs/",
        help="Path to config file",
    )
    PARSER.add_argument(
        "--local-dir",
        default="ReinforcementLearning/logs/ray_runs/",
        help="Logging dir for RLlib",
    )
    PARSER.add_argument(
        "--workers", default=1, type=int, help="Number of RLLib rollout workers"
    )
    PARSER.add_argument(
        "--use-tune", default=False, type=bool, help="Use tune settings"
    )
    PARSER.add_argument(
        "--logging-level",
        default="INFO",
        help="Level of logging for ray to perform. E.g. DEBUG, INFO, WARN",
    )
    PARSER.add_argument(
        "--add-to-name", default="", help="Additional info to add to file name"
    )
    ARGS = PARSER.parse_args()

    main(**vars(ARGS))  # Convert Namespace -> dict, then unroll as kwargs
