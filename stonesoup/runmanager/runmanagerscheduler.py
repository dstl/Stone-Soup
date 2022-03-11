from .base import RunManager

import subprocess

class RunManagerScheduler(RunManager):
    
    def __init__(self, rm_args):
        # Initialise run manager parameters for slurm
        self.rm_args = rm_args
        try:
            self.n_nodes = int(input("How many nodes would you like to use? "))
        except Exception as e:
            print("Invalid input for number of nodes, must be a numerical value. ", e)

        print("Number of nodes:", self.n_nodes)
        self.node_split = self.rm_args['nruns']//self.n_nodes
        self.rm_args['nruns'] = self.node_split
        print("Number of runs per node: ", self.rm_args['nruns'])

    def schedule_jobs(self):
        # Simply run runmanager number of times with same parameters on different nodes
        # for node in n_nodes: sbatch runmanager.py with jobarray=1-self.rm_args['nruns'],
        # slurm=False, rmargs

        # If using slurm job-array, replace nruns parameter
        # job_array_size = self.rm_args['nruns']
        # self.rm_args['nruns'] = None

        # Reset command line arguments for running on nodes
        rm_args_str = ""
        for key, arg in self.rm_args.items():
            if arg is not None:
                rm_args_str += '--' + str(key) + ' ' + str(arg) + ' '

        rm_args_new = f"python stonesoup/runmanager/runmanager.py {rm_args_str}"
        # rm_args_new = f"sbatch --array=1-{self.rm_args['nruns']} python stonesoup/runmanager/runmanager.py {rm_args_str}"

        # for node_n in range(self.n_nodes):
        #     print("Running on node: ", node_n)
        subprocess.run(rm_args_new)
        pass
