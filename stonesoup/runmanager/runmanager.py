from stonesoup.runmanager.runmanagercore import RunManagerCore
import sys
import argparse

def manage_if(arg):
    if arg:
        dir = arg
    else:
        dir = None
    return dir
    
if __name__ == "__main__":
    rmc = RunManagerCore()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="A configuration .yaml file", type=str)
    parser.add_argument("--parameter", "-p", help="A Parameter .json file", type=str)
    parser.add_argument("--groundtruth", "-g", help="Ground truth setting. True for ground truth, False for no ground_truth. Default is False", type=bool)
    parser.add_argument("--dir", "-d", help="Specify a directoy with config & parameter files.", type=str)
    parser.add_argument("--nruns", "-n", help="Specify the number of monte carlo runs you want to perform for each simulation", type=int)
    parser.add_argument("--processes", "-pc", help="Specify the number of processing cores to use", type=int)
    args = parser.parse_args()
    
    config=manage_if(args.config)
    parameter=manage_if(args.parameter)
    groundtruth=manage_if(args.groundtruth)
    dir=manage_if(args.dir)
    nruns=manage_if(args.nruns)
    nprocesses=manage_if(args.processes)

    rmc.run(config, parameter, groundtruth, dir, nruns, nprocesses)