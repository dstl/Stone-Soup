from .base import RunManager

import subprocess
import numpy as np

class RunManagerScheduler(RunManager):
    
    def __init__(self, rm_args, logger):
        # Initialise run manager parameters for slurm
        self.rm_args = rm_args
        self.logger = logger

        try:
            self.n_nodes = int(input("How many nodes would you like to use? "))
        except Exception as e:
            self.logger.error(f"Invalid input for number of nodes, must be a numerical value. {e}")

        if self.rm_args['parameter'] is None:
            self.logger.info("No parameter json given, defaulting to splitting across runs.")
            self.split_sims = False
        else:
            try:
                self.split_sims = bool(int(input("Split runs(0) or simulations(1) across nodes? ")))
            except Exception as e:
                self.logger.error(f"Invalid input for node split option, must be a 0 or 1. {e}")

        self.logger.info(f"Number of nodes: {self.n_nodes}")

    def schedule_jobs(self, run_manager):
        # Simply run runmanager number of times with same parameters on different nodes
        # for node in n_nodes: sbatch runmanager.py with jobarray=1-self.rm_args['nruns'],
        # slurm=False, rmargs

        # If using slurm job-array, replace nruns parameter
        # job_array_size = self.rm_args['nruns']
        # self.rm_args['nruns'] = None

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
        Creates n_nodes new run manager instances, runs node_split times
        on each node.
        """
        if self.rm_args['nruns'] is not None:
            self.node_split = np.array_split(range(self.rm_args['nruns']), self.n_nodes)
            self.rm_args['nruns'] = self.node_split
            self.logger.info(f"Number of runs per node: {[len(self.rm_args['nruns'][i]) for i in range(self.n_nodes)]}")

        # Reset command line arguments for running on nodes
        rm_args_str = ""
        for key, arg in self.rm_args.items():
            if arg != None and key != 'nruns':
                if type(arg) is str:
                    arg = f'"{arg}"'
                rm_args_str += f"--{str(key)} {str(arg)} "

        for node_n in range(self.n_nodes):
            node_nruns = len(self.rm_args['nruns'][node_n])
            if node_nruns > 0:
                self.logger.info(f"Running on node: {node_n}")
                nruns_arg = f"--nruns {node_nruns} "
                rm_args_new = f"python stonesoup/runmanager/runmanager.py {rm_args_str} {nruns_arg}"
                # rm_args_new = f"sbatch --array=1-{self.rm_args['nruns']} python stonesoup/runmanager/runmanager.py {rm_args_str} {nruns_arg}"
                subprocess.run(rm_args_new)

    def schedule_simulations(self, run_manager):
        """ Simulations split handled in the run manager core run function"""
        run_manager.run()