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

## Log file

The run manager will produce a `simulation.log` file at your root directory. This logs any errors which may occur in the runmanager.

---

## Known Issues

### Errors

The terminal and simulation will sometimes log ERROR with certain parameter combinations. This is likely due to a parameter combination that is generated in the monte-carlo runs which isn't compatible with the configuration of StoneSoup. Typically it shouldn't cause much of a problem it just means that these simulations can be ignored as they have invalid parameter combinations.

### Custom initiator

The current runmanager system will not work if the configuration file contains a custom initiator class.
