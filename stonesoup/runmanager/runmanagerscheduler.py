from .base import RunManager

import subprocess
import numpy as np
from datetime import datetime
import os
import socket


class RunManagerScheduler(RunManager):

    def __init__(self, rm_args, logger):
        """The constructor for scheduling jobs for when using slurm.
        Asks the user for slurm required slurm arguments.
        """
        # Initialise run manager parameters for slurm
        self.rm_args = rm_args
        self.logger = logger

        try:
            self.n_nodes = int(input("How many nodes would you like to use? "))
        except Exception as e:
            self.logger.error(f"Invalid input for number of nodes, must be a numerical value. {e}")

        # if self.rm_args['parameter'] is None:
        #     self.logger.info("No parameter json given, defaulting to splitting across runs.")
        #     self.split_sims = False
        # else:
        #     try:
        #         self.split_sims=bool(int(input("Split runs(0) or simulations(1) over nodes? ")))
        #     except Exception as e:
        #         self.logger.error(f"Invalid input for node split option, must be a 0 or 1. {e}")

        # Emailing did not seem to work for me
        # self.email = str(input("Enter email address to notify on each job update: "))
        self.split_sims = False
        self.logger.info(f"Number of nodes: {self.n_nodes}")

    def schedule_jobs(self, run_manager):
        """Simply run runmanager number of times with same parameters on different nodes
        for node in n_nodes.

        Parameters
        ---------
        run_manager : RunManagerCore
            NOT USED. The RunManagerCore instance for scheduling simulations.

        """


        if self.split_sims:
            self.logger.info("Splitting simulations across nodes...")
            self.schedule_simulations(run_manager)
        elif not self.split_sims:
            self.logger.info("Splitting runs across nodes...")
            if self.rm_args['nruns'] is None:
                self.rm_args['nruns'] = 1
            self.schedule_runs()
        else:
            self.logger.info("Invalid split option.")

    def schedule_runs(self):
        """
        Creates n_nodes new run manager instances and splits total runs into even splits per node,
        runs node_split times on each node.
        """
        if self.rm_args['nruns'] is not None:
            self.node_split = np.array_split(range(self.rm_args['nruns']), self.n_nodes)
            self.rm_args['nruns'] = self.node_split
            self.logger.info(f"Number of runs per node:\
                            {[len(self.rm_args['nruns'][i]) for i in range(self.n_nodes)]}")

        # Reset command line arguments for running on nodes
        rm_args_str = ""
        for key, arg in self.rm_args.items():
            if arg is not None and key != 'nruns':
                if type(arg) is str:
                    arg = f'"{arg}"'
                rm_args_str += f"--{str(key)} {str(arg)} "

        datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        slurm_output_dir = f"runmanager-slurm_{datetime_str}/"
        os.mkdir(slurm_output_dir)
        hostname = socket.gethostname()

        for node_n in range(self.n_nodes):
            node_nruns = len(self.rm_args['nruns'][node_n])
            if node_nruns > 0:
                self.logger.info(f"Running on node: {node_n}")
                nruns_arg = f"--nruns {node_nruns} "
                # rm_args_new = f"python3 stonesoup/runmanager/runmanager.py\
                #  {rm_args_str} {nruns_arg}"
                output_str = f"{slurm_output_dir}node{node_n}-output.out"
                node_str = f"_{hostname}_{node_n}"
                rm_args = f"{rm_args_str} {nruns_arg}" + \
                    f" --slurm_dir {slurm_output_dir} --node {node_str}"
                sb_comm = f"sbatch --output={output_str}"
                rm_comm = f"{sb_comm} stonesoup/runmanager/runmanager.py {rm_args}"
                subprocess.run(rm_comm, shell=True)

    def schedule_simulations(self, run_manager):
        """NOT YET USED. Simulations split handled in the run manager core run function"""
        run_manager.run()
