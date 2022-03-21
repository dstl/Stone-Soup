# Runmanager

The runmanager is a command line interface that allows the user to run monte-carlo simulations on various configuration files. The runmanager can run a single
configuration file, configuration file along with a parameter file or a directory containing multiple configurations & parameters.

---

## Setup

**Requires pathos module to work. This is a dependancy for the multiprocessing feature. (BSD LICENSE)**

To use the runmanager you must make a copy of the runmanager.py folder in the directory you wish to run from i.e your root folder.

---

## Running

The run manager has a few options for running these can be found using the command `python runmanager.py --help` or `-h`.

The following arguments are accepted.

- `-h` or `--help`: Shows the help message.
- `-c` or `--config`: Path to a configuration .yaml file (_May be used for a single run_)
- `-p` or `--parameter`: Path to a paramater .json file (_List of a all parameters required for the monte-carlo simulations._)
- `-g` or `--groundtruth`: Ground truth setting. True if a ground truth is present, False if no ground truth is present. (_Default False_)
- `-d` or `--dir`: Directory to a set of configuration files & parameter files. Used to run multiple trackers in a directory.
- `-n` or `--nruns`: Number of monte-carlo runs. This feature supersedes if nruns is also set in the parameter.json file. (_Default 1 run_)
- `-pc` or `--processes`: Number of cpu processing cores to use. This feature supersedes if num*proc is also set in the parameter.json file. (\_Default 1 core*)

The runmanager will work with 3 different options:

1. A single configuration file with single run. Example: `python runmanager.py -c "path/to/config.yaml"`
2. A single configuration file with a parameter file. This will run n amount of times based on amount of parameter combinations. Example: `python runmanager.py -c "path/to/config.yaml -p "path/to/parameters.json"`
3. A directory of configuration files with parameter files. This will run n amount of times based on amount of parameter combinations across all configs. Example: `python runmanager.py -d "path/to/configdirectory"`

## Config File.

The config file can have 3 components:

- Tracker
- Ground Truth
- Metric Manager

If the file has 2 components, they will be recognise as:

- Tracker
- Metric Manager

The groundtruth will be set as `tracker.detection.groundtruth`.

If the file has 1 component, it will be recognise as:

- Tracker

The groundtruth will be set as `tracker.detection.groundtruth`.
No metric manager.

If the command line has the argument `-g` the components will be recognise as:

- Tracker
- Ground Truth

## Ground truth

In the case where the ground truth is a csv datafile make sure this is in the directory which is specified in the configuration file. The runmanager will automatically pick up the ground truth from this location if it is present.

## Number of runs

Number of runs or `-n` specifies the number of monte-carlo runs that you want wish to execute for each configuration file.

## Multiprocessing

To use multiprocessing you simply need to use the -p command with the number of cores you wish to use. This will run multiple simulations in parallel. If there are a large number of simulations to be ran please use this option. The multiprocessing module will batch process the simulations.

Example for system with 16 CPU Cores: `python runmanager.py -d "path/to/configdirectory" -pc 16`

---

## Output

The simulation output files will be created with a timestamp in your root directory. This will contain a simulation folder for each different parameter combination ran and sub folders for each monte-carlo run with those specific parameters.

If a configuration file has both a ground truth and metric manager, the following folders should appear in the directory

- `config.yaml` - Configuration file for this specific run.
- `detections.csv` - CSV file containing the detections.
- `groundtruth.csv` - CSV file containing the ground truth.
- `metrics.csv` - CSV file containing the metrics.
- `tracks.csv` - CSV file containing the tracks.
- `parameters.json` - Easy to view json file with the parameters which have changed in this specific run.

---

## Averaging metrics

Once all simulations have ran the runmanager will average all of the monte-carlo run metric files and collate them into a single metrics file per simulation. This will allow the user to compare results of different parameter combinations. The average is across all runs per simulation on a cell level in order to retain the timestamp.

The metrics averaging will only work with real ground truth samples or a ground truth simulator where there is a fixed seed as the metrics.csv files need to be of the same length.

Within each simulation folder a file named `average.csv`  will appear. This is the average metric value of all monte-carlo runs for this simulation.

## Log file

The run manager will produce a `simulation.log` file at your root directory. This logs any errors which may occur in the runmanager.

## HPC/AWS functionality
Stonesoup RunManager slurm guide:

1. Running the RunManager locally:

Example commands: 
-- runmanager.py -c config.yaml -p param.json -g True -n 8 -pc 4

- You may want to run the RunManager locally first with less runs or computer intensive parameters to test
- your simulations first, in which case make sure to omit the '-s' flag or set it to False when running your command,
- as this flag expects slurm scheduling to be used when it is set to True. In the case where this has been left in,
- the simulation will fail and will output an error similar to:
- ''sbatch' is not recognized as an internal or external command, operable program or batch file.'
- if slurm is not install on your local machine.
- In another case where slurm is installed on your local machine and this flag is left as True, the simulation can still
- run without error and the RunManager will execute on the available nodes only, which will just be the local machine.

2. Running the RunManager on a compute cluster with Slurm scheduling:

Example commands: 
-- runmanager.py -c config.yaml -p param.json -g True -n 8 -pc 4 -s True

- When the '-s' flag is set to True, the RunManager will assume the user wishes to execute the command across a number
- of jobs on compute nodes that are scheduled and managed by Slurm.
- When the command is run with this flag as True, the RunManager will prompt the user to input how many compute nodes
- they would like to use for the simulations. The RunManager will then only use this number of nodes during execution,
- even if there are other available nodes. If the user inputs too many nodes than there are available, the RunManager will
- only use the available nodes and slurm will automatically schedule the remaining jobs until more nodes are free.
- The total number of runs intended to be ran will then be evenly divided across the number of compute nodes intended to
- be used. For example, if the user wants to execute 80 runs across 4 nodes, each node will execute 20 runs. In cases where
- there is not an even division, some nodes may do 1 more run that some of the others.
- Here, the '-n' or '--nruns' flag substitutes the need for a user to set a job-array slurm command that would
- define how many times the command will be run. It is important to remember '-n' is the TOTAL number of runs you wish to do,
- not runs per node.
- The RunManagerScheduler will then create n_node number of new RunManager instances to run on each node
- the divided n number of times.
- In order to organise the outputs of the slurm jobs, a new directory with the name pattern:
- 'runmanager-slurm_YYYY_MM_DD_hh_mm_ss'
- which will contain the regular RunManager output directories for each node containing each simulation output for each run,
- as well as the slurm.out output files where the RunManager command line logs are written to.
- All of the logs for the simulations on all nodes can also be found in the 'simulation_info.log' file the same way as
- local RunManager executions.

-- Example RunManager with Slurm workflow:
- Setup
1. Run run manager on a small set of runs locally to test the success of the simulations.
2. Once the execution has been run locally and the user is happy with the simulation runs, login to HPC/AWS:
   - e.g using ssh
3. Make sure Stonesoup is installed with HPC ready version of Run Manager
4. Make sure all configuration/parameter/csv files needed for the RunManager are on the HPC storage
5. May want to check everything is okay by running the same local RunManager command on HPC/AWS (Optional)

- Execution
1. Run 'sinfo' command to see information of available nodes and their types on the HPC
2. Run the Run Manager from the top of the Stone-Soup directory with '-s / --slurm' flag set to True and '-n / --nruns' flag
   set to the TOTAL number of runs you wish the Run Manager to execute. These runs will be split across the number of nodes
   you wish to use.
   - e.g  'python stonesoup/runmanager/runmanager.py -c config.yaml -p param.json -n 500 -s True'
3. The Run Manager will then ask the user for the number of nodes they wish to split the runs over. User must input a number.
4. The Run Manager will then run the 'sbatch' command n_node number of times to create a new instance for each node and run
   the split simulations in each node.
5. All of the output for the executions across the nodes will be stored in a directory with the following pattern: 'runmanager-slurm_YYYY_MM_DD_hh_mm_ss',
   including the node.out files, simulation and run output files and metrics, tracking information and averages across runs for each node.

---

## Known Issues

### Errors

The terminal and simulation will sometimes log ERROR with certain parameter combinations. This is likely due to a parameter combination that is generated in the monte-carlo runs which isn't compatible with the configuration of StoneSoup. Typically it shouldn't cause much of a problem it just means that these simulations can be ignored as they have invalid parameter combinations.

### Custom initiator

The current runmanager system will not work if the configuration file contains a custom initiator class.
