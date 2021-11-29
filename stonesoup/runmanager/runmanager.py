from stonesoup.runmanager.runmanagercore import RunManagerCore
import sys
import argparse

if __name__ == "__main__":
    rmc = RunManagerCore()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", help="Specify a directoy with config & parameter files.", type=str)
    parser.add_argument("--config", "-c", help="A configuration .yaml file", type=str)
    parser.add_argument("--parameter", "-p", help="A Parameter .json file", type=str)
    parser.add_argument("--groundtruth", "-g", help="Ground truth setting. True for ground truth, False for no ground_truth. Default is False", type=bool)
    parser.add_argument("--nruns", "-n", help="Specify the number of monte carlo runs you want to perform for each simulation", type=int)
    parser.add_argument("--processes", "-pc", help="Specify the number of processing cores to use", type=int)
    args = parser.parse_args()
    
    if args.dir:
        dir = args.dir
        print(f"Directory: {dir} selected.")
    else:
        dir = None
        print(f"No Directory selected.")
        
    if args.config:
        config = args.config
        print(f"Configuration file {config} selected.")
    else:
        config = None
        print(f"No Configuration file selected.")
    
    if args.parameter:
        parameter = args.parameter
        print(f"Parameter file {parameter} selected.")
    else:
        parameter = None
        print(f"No parameter file selected.")

    if args.groundtruth:
        groundtruth = args.groundtruth
        print(f"Ground truth is {groundtruth}.")
    else: 
        groundtruth = False
        print(f"No ground truth specified.")
    
    if args.nruns:
        nruns = args.nruns
        print(f"number of monte carlo runs: {nruns} selected.")
    else: 
        nruns = 1
        print(f"Defaulted to number of runs {nruns}, will use parameter file if present.")
        
    if args.processes:
        nprocesses = args.processes
        print(f'Using {nprocesses} processing cores.')
    else: 
        nprocesses = None
        print('Using default processing cores.')
    
    rmc.run(config, parameter, groundtruth, dir, nruns, nprocesses)