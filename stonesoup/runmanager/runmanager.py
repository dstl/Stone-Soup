#!/usr/bin/env python3

from stonesoup.runmanager.runmanagercore import RunManagerCore, info_logger
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
    # parser.add_argument("--montecarlo", "-mc",
    #                    help="""NOT YET IMPLEMENTED. Specify the type of Monte-Carlo
    #                    distribution you want.
    #                    0: Equal 1: Logarithmic, 2: Exponential, 3: Random Distributed""",
    #                    type=int)
    parser.add_argument("--slurm", "-s",
                        help="""Slurm setting, set True if using a HPC and need to schedule RunManager
                        executions/jobs using slurm. Default is False""",
                        type=bool)
    parser.add_argument("--slurm_dir", "-sd",
                        help="""Only used with slurm scheduler. Directory name to store all RunManager
                        output files and directories.""",
                        type=str)
    parser.add_argument("--node", "-nd",
                        help="""Optional. The name of the node/pc the RunManager is running on. Is
                        automatically set when slurm scheduling is used.""",
                        type=str)
    args = parser.parse_args()

    config = manage_if(args.config)
    parameter = manage_if(args.parameter)
    groundtruth = manage_if(args.groundtruth)
    dir = manage_if(args.dir)
    nruns = manage_if(args.nruns)
    nprocesses = manage_if(args.processes)
    # montecarlo = manage_if(args.montecarlo)
    slurm = manage_if(args.slurm)
    slurm_dir = manage_if(args.slurm_dir)
    node = manage_if(args.node)

    rm_args = {
        "config": config,
        "parameter": parameter,
        "groundtruth": groundtruth,
        "dir": dir,
        # "montecarlo": montecarlo,
        "nruns": nruns,
        "processes": nprocesses,
        "slurm": slurm,
        "slurm_dir": slurm_dir,
        "node": node
        }

    rmc = RunManagerCore(rm_args)
    if rmc.slurm:
        rmc.run_manager_scheduler.schedule_jobs(rmc)
    else:
        rmc.run()

    for handler in info_logger.handlers:
        handler.close()
