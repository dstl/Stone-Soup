from stonesoup.runmanager.runmanagercore import RunManagerCore
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="A configuration .yaml file", type=str)
    parser.add_argument("--parameter", "-p", help="A Parameter .json file", type=str)
    parser.add_argument("--groundtruth", "-g", help="Ground truth setting. 1 for ground truth, 0 for no ground_truth", type=int)
    parser.add_argument("--dir", "-d", help="A config & parameter directory", type=str)
    args = parser.parse_args()
    dir = args.dir
    
    try:
        configInput = args.config
    except Exception as e:
        configInput = "C:\\Users\\hayden97\\Documents\\Projects\\Serapis\\Data\\2021_Nov_24_10_23_40_560750.yaml"
        print(e)

    try:
        parametersInput = args.parameter
    except Exception as e:
        parametersInput = "C:\\Users\\hayden97\\Documents\\Projects\\Serapis\\Data\\2021_Nov_24_10_23_40_560750_parameters.json"
        print(e)

    try:
        groundtruthSettings = args.groundtruth
    except Exception as e:
        groundtruthSettings = 1
        print(e)

    rmc = RunManagerCore(configInput, parametersInput, groundtruthSettings, dir)
    rmc.run(nprocesses=3)