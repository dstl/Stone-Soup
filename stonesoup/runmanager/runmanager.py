import logging
import sys
from .runmanagercore import RunManagerCore
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="A configuration .yaml file", type=str)
    parser.add_argument("--parameter", "-p", help="A Parameter .json file", type=str)
    parser.add_argument("--groundtruth", "-p", help="Ground truth setting. 1 for ground truth, 0 for no ground_truth", type=int)
    parser.add_argument("--dir", "-d", help="A config & parameter directory", type=str)
    args = parser.parse_args()
    print("args directory {0}", args.config_dir)

    try:
        configInput = args.config

    except Exception as e:
        configInput = ""
        logging.error(e)

    try:
        parametersInput = args.parameters
    except Exception as e:
        parametersInput = ""
        logging.error(e)
        # parametersInput= "C:\\Users\\gbellant\\Documents\\Projects\\Serapis\\dummy3.json"

    try:
        groundtruthSettings = args.groundtruth
    except Exception as e:
        groundtruthSettings = 1
        logging.error(e)

    rmc = RunManagerCore()

    rmc.run(configInput, parametersInput, groundtruthSettings)
