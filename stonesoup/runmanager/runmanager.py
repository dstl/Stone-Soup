from stonesoup.runmanager.runmanagercore import RunManagerCore
import argparse


def manage_if(arg):
    if arg:
        dir = arg
    else:
        dir = None
    return dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c",
                        help="A configuration .yaml file",
                        type=str)
    parser.add_argument("--parameter", "-p",
                        help="A Parameter .json file",
                        type=str)
    parser.add_argument("--groundtruth", "-g",
                        help="""Ground truth setting. Set True if a groundtruth has been appended
                         to the config after the track and you wish to explicitly use that
                         groundtruth. Otherwise, set to False if
                         no ground_truth explicitly added config. Default is False""",
                        type=bool)
    parser.add_argument("--dir", "-d",
                        help="Specify a directory with config & parameter files.",
                        type=str)
    parser.add_argument("--nruns", "-n",
                        help="""Specify the number of monte carlo runs you
                        want to perform for each simulation""",
                        type=int)
    parser.add_argument("--processes", "-pc",
                        help="Specify the number of processing cores to use",
                        type=int)
    parser.add_argument("--montecarlo", "-mc",
                        help="""NOT YET IMPLEMENTED. Specify the type of Monte-Carlo distribution you want.
                        0: Equal 1: Logarithmic, 2: Exponential, 3: Random Distributed""",
                        type=int)
    args = parser.parse_args()

    config = manage_if(args.config)
    parameter = manage_if(args.parameter)
    groundtruth = manage_if(args.groundtruth)
    dir = manage_if(args.dir)
    nruns = manage_if(args.nruns)
    nprocesses = manage_if(args.processes)
    montecarlo = manage_if(args.montecarlo)

    rmc = RunManagerCore(config, parameter, groundtruth, dir, montecarlo, nruns, nprocesses)
    rmc.run()
