import json
import os
import os.path as osp
from typing import Any

import yaml

from ReinforcementLearning.environment.gym import StoneSoupEnv


def load_tune_yaml(config_path: str) -> dict[str, Any]:

    """Load yaml file as dict.
    Args:
        config_path: Path to .yaml file.
    Returns:
        tune_kwargs: Dict containing k-v corresponding to kwargs for `tune.run()`
    """
    _, ext = os.path.splitext(config_path)
    with open(config_path, "r") as stream:
        if ext in [".yml", ".yaml"]:
            tune_kwargs = yaml.full_load(stream)
        else:
            raise RuntimeError("No yaml supplied")
    return tune_kwargs


def load_dict(dict_path):

    _, ext = osp.splitext(dict_path)
    with open(dict_path, "r") as stream:
        dict_str = stream.read()

        if ext in [".json"]:
            yaml_dict = json.loads(dict_str)
        elif ext in [".yml", ".yaml"]:
            yaml_dict = yaml.load(dict_str, Loader=yaml.FullLoader)
        else:
            raise RuntimeError("No configs found")
    return yaml_dict


def env_creator(**kwargs):

    """
    Creates a custom Stone Soup Environment
    """
    env = StoneSoupEnv(
        scenario_config="ReinforcementLearning/configs/scenario_config.yaml",
        render_episodes=False,
        log_dir="ReinforcementLearning/logs/renders/",
    )
    return env


def get_configs(log_dir):

    assert osp.isdir(log_dir), "Log dir does not exist."

    if osp.exists(osp.join(log_dir, "config.yaml")):
        print("CONF YAML BEING RETURNED")
        exp_configs = load_dict(osp.join(log_dir, "config.yaml"))
        return exp_configs

    if osp.exists(osp.join(log_dir, "params.json")):
        exp_configs = load_dict(osp.join(log_dir, "params.json"))
        print("PARAMS JSON BEING RETURNED")
        return exp_configs

    raise RuntimeError("No configs found")
