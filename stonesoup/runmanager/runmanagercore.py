import copy
import json
import logging
import time
import glob
import numpy as np
import subprocess
import pickle

import os
import multiprocessing

from pathos.multiprocessing import ProcessingPool as Pool
from datetime import datetime
from stonesoup.serialise import YAML
from .inputmanager import InputManager
from .runmanagermetrics import RunmanagerMetrics
from .runmanagerscheduler import RunManagerScheduler
from .base import RunManager


class RunManagerCore(RunManager):
    """
    Core RunManager class that contains all functionality for loading configuration and parameter
    files, generating and setting parameters as well as running a number of montecarlo simulations,
    either with or without multiprocessing.

    Parameters
    ----------
    Runmanager : Class
        Run manager base class
    """
    TRACKER = "tracker"
    GROUNDTRUTH = "ground_truth"
    METRIC_MANAGER = "metric_manager"

    def __init__(self, rm_args={
        "config": None,
        "parameter": None,
        "groundtruth": None,
        "dir": None,
        "montecarlo": None,
        "nruns": None,
        "processes": None,
        "slurm": None,
        "slurm_dir": None,
        "node": ""
    }):
        """The init function for RunManagerCore, initiating the key settings to allow
        the running of simulations.

        Parameters
        ----------
        config_path : str
            The path to the configuration file containing the tracker
        parameters_path : str
            The path to the parameters json file containing the parameters for a configuration
        groundtruth_setting : bool
            A boolean flag to indicate whether the user has included the groundtruth in the
            configuration file or not
        montecarlo : bool
            A boolean to indicate if montecarlo simulations are to be used
        dir : str
            The path to the directory containing configuration and parameter pairs
        nruns : int, optional
            number of monte-carlo runs, by default 1
        nprocesses : int, optional
            number of processing cores to use, by default 1
        """

        self.config_path = rm_args["config"]
        self.parameters_path = rm_args["parameter"]
        self.groundtruth_setting = rm_args["groundtruth"]
        self.montecarlo = rm_args["montecarlo"]  # Not used yet
        self.dir = rm_args["dir"]
        self.nruns = rm_args["nruns"]
        self.nprocesses = rm_args["processes"]
        self.slurm = rm_args["slurm"]
        self.slurm_dir = rm_args["slurm_dir"]
        self.node = rm_args["node"]

        if self.slurm_dir is None:
            self.slurm_dir = ""
        if self.node is None:
            self.node = ""

        self.total_trackers = 0
        self.current_run = 0
        self.current_trackers = 0

        self.input_manager = InputManager()
        self.run_manager_metrics = RunmanagerMetrics()
        # If using slurm hpc, setup scheduler here
        if self.slurm:
            info_logger.info("Slurm scheduler enabled.")
            rm_args['slurm'] = None
            if self.parameters_path:
                rm_args['nruns'] = self.set_runs_number(self.nruns,
                                                        self.read_json(self.parameters_path))
            self.run_manager_scheduler = RunManagerScheduler(rm_args, info_logger)

        # logging.basicConfig(filename='simulation.log', encoding='utf-8', level=logging.INFO)
        # self.info_logger = self.setup_logger('self.info_logger', 'simulation_info.log')
        # self.info_logger.info(f'RunManagerCore started. {datetime.now()}')
        info_logger.info(f'RunManagerCore started. {datetime.now()}')

    def read_json(self, json_input):
        """Opens and reads a json file from a given path.

        Parameters
        ----------
        json_input : str
            path to json file

        Returns
        -------
        dict
            returns a dictionary of json data
        """
        with open(json_input) as json_file:
            json_data = json.load(json_file)
            logging.info(f'{datetime.now()} Accessed jsonfile {json_file}')
            return json_data

    def run(self):
        """Handles the running of single file configurations,
        multiple files and defines the structure of the run.
        """

        # Start timer
        start = time.time()
        # Single simulation. No param file detected
        if self.config_path and self.parameters_path is None:
            if self.nruns is None:
                self.nruns = 1
            if self.nprocesses is None:
                self.nprocesses = 1
            self.prepare_single_simulation()

        else:
            pairs = self.config_parameter_pairing()
            if not pairs and not self.config_path:
                print(f"{datetime.now()} No files in current directory: {self.dir}")
                info_logger.info(f"{datetime.now()} No files in " +
                                 f"current directory: {self.dir}")

            for path in pairs:
                # Read the param data
                self.config_path, param_path = path[0], path[1]
                json_data = self.read_json(param_path)

                self.nruns = self.set_runs_number(self.nruns, json_data)
                nprocesses = self.set_processes_number(self.nprocesses, json_data)
                combo_dict = self.prepare_monte_carlo(json_data)
                # if self.slurm:
                #     self.schedule_simulations(combo_dict, nprocesses)
                # else:
                self.prepare_monte_carlo_simulation(combo_dict, self.nruns,
                                                    nprocesses, self.config_path)

        # End timer
        end = time.time()
        info_logger.info(f"{datetime.now()} Finished all simulations in " +
                         f"--- {end - start} seconds ---")
        # Average all of the metrics at the end
        self.average_metrics()

    def average_metrics(self):
        """Handles the averaging of the metric files for both single simulations
        and multi simulations.

        In future updates better memory handling should be implemented to automatically
        load as many dataframes as possible into memory to provide a more efficient process.

        Parameters
        ----------
        batch_size : int
            Size of the batches to split the dataframes.
            May need adjusting for very large datasets to save memory space.
        """
        batch_size = 200
        start = time.time()
        path, config = os.path.split(self.config_path)

        try:
            info_logger.info(f"{datetime.now()} Averaging metrics for all Monte-Carlo Simuatlions")
            directory = glob.glob(f'./{self.slurm_dir}{config}_{self.config_starttime}*/simulation*',
                                  recursive=False)
            if directory:
                for simulation in directory:
                    summed_df, sim_amt = self.run_manager_metrics.sum_simulations(simulation,
                                                                                  batch_size)
                    df = self.run_manager_metrics.average_simulations(summed_df, sim_amt)
                    df.to_csv(f"./{self.slurm_dir}{simulation}/average.csv", index=False)
            else:
                directory = glob.glob(f'{config}_{self.config_starttime}*', recursive=False)
                summed_df, sim_amt = self.run_manager_metrics.sum_simulations(directory,
                                                                              batch_size)
                df = self.run_manager_metrics.average_simulations(summed_df, sim_amt)
                df.to_csv(f"./{self.slurm_dir}{config}_{self.config_starttime}/average.csv", index=False)
            end = time.time()
            info_logger.info(f"{datetime.now()} Finished Averaging in " +
                             f"--- {end - start} seconds ---")

        except Exception as e:
            info_logger.error(f"{datetime.now()} Failed to average simulations.")
            info_logger.error(f"{datetime.now()} {e}")
            print(f"{datetime.now()} Failed to average simulations.")

    def schedule_simulations(self, combo_dict, nprocesses):
        # Split generated parameter combinations into n_node batches
        combo_dict_split = np.array_split(combo_dict, self.run_manager_scheduler.n_nodes)
        combo_batch_i = 0
        # For each node, run monte carlo simulations on a batch
        for combo_dict_batch in combo_dict_split:
            info_logger.info(f"Running parameter batch: {combo_batch_i+1}")
            # Pickle this RunManager so it is the same instance
            # for each batch/node and can pass same parameters
            pickle_batch_params = pickle.dumps([self, combo_dict_batch, self.nruns,
                                                nprocesses, self.config_path])
            # subprocess.run(
            #     f'python3 -c\
            #          "from stonesoup.runmanager.runmanagercore import RunManagerCore as rmc;\
            #               rmc.load_batch_params(rmc, {pickle_batch_params})"', shell=True)
            subprocess.run(
                f'sbatch "#!/usr/bin/python3\
                 from stonesoup.runmanager.runmanagercore import RunManagerCore as rmc;\
                      rmc.load_batch_params(rmc, {pickle_batch_params})"', shell=True)
            combo_batch_i += 1

    @staticmethod
    def load_batch_params(rmc, params):
        params_list = pickle.loads(params)
        rmc.prepare_monte_carlo_simulation(params_list[0], params_list[1], params_list[2],
                                           params_list[3], params_list[4])

    def set_runs_number(self, nruns, json_data):
        """Sets the number of runs.

        Parameters
        ----------
        nruns : int
            number of run from the terminal
        json_data : object
            json parameter object

        Returns
        -------
        int
            number of runs
        """
        if nruns is None:
            try:
                nruns = json_data['configuration']['runs_num']
                self.set_runs_number(nruns, None)
            except Exception as e:
                info_logger.error(e, "runs_num value from json not found, defaulting to 1")
                nruns = 1
        elif nruns > 1:
            pass
        else:
            nruns = 1

        return nruns

    def set_processes_number(self, nprocess, json_data):
        """Sets the number of processes.

        Parameters
        ----------
        nprocess : int
            number of process from the terminal
        json_data : object
            json parameter object

        Returns
        -------
        int
            number of process
        """
        if nprocess is None:
            try:
                nprocess = json_data['configuration']['proc_num']
                self.set_processes_number(nprocess, None)
            except Exception as e:
                info_logger.error(e, "proc_num value from json not found, defaulting to 1")
                nprocess = 1
        elif nprocess > 1:
            pass
        else:
            nprocess = 1
        return nprocess

    def prepare_monte_carlo(self, json_data):
        """Prepares the combination of parameters for a monte carlo run.

        Parameters
        ----------
        json_data : string in json format
            data from the parameter file

        Returns
        -------
        dict
            combination of all the parameters to run a monte-carlo
        """
        # Generate all the parameters for the monte carlo run
        trackers_combination_dict = self.input_manager.generate_parameters_combinations(
            json_data["parameters"])
        # Generate all the the possible combinations with the parameters
        combo_dict = self.input_manager.generate_all_combos(trackers_combination_dict)
        return combo_dict

    def config_parameter_pairing(self):
        """Pairs the config file with the parameter file.

        Returns
        -------
        array
            array that contains of config path and parameter path
            [config_path, parameter_path]
        """
        pairs = []
        if self.dir:
            paths = self.get_filepaths(self.dir)
            pairs = self.get_config_and_param_lists(paths)

        elif self.config_path and self.parameters_path:
            pairs = [[self.config_path, self.parameters_path]]

        elif self.dir and self.config_path and self.parameters_path:
            paths = self.get_filepaths(self.dir)
            pairs = self.get_config_and_param_lists(paths)
            if [self.config_path, self.parameters_path] not in pairs:
                pairs.append([self.config_path, self.parameters_path])

        return pairs

    def check_ground_truth(self, ground_truth):
        """Check if the groundtruth has generate path or not.
        If yes return the generate path and if not simply return groundtruth.

        Parameters
        ----------
        ground_truth : stonesoup object
            ground_truth

        Returns
        -------
        groundtruth
            groundtruth object after the checking
        """
        try:
            ground_truth = ground_truth.groundtruth_paths
        except Exception:
            ground_truth = ground_truth
        return ground_truth

    def run_simulation(self, simulation_parameters, dir_name):
        """Runs a single simulation on a tracker, groundtruth and metric manager

        Parameters
        ----------
        simulation_parameters : dict
            contains the tracker, the ground_truth and the metric manager
        dir_name : str
            output directory for metrics
        """
        tracker = simulation_parameters['tracker']
        ground_truth = simulation_parameters['ground_truth']
        metric_manager = simulation_parameters['metric_manager']

        try:
            log_time = datetime.now()
            self.logging_starting(log_time)
            for _, ctracks in tracker.tracks_gen():
                self.run_manager_metrics.tracks_to_csv(dir_name, ctracks)
                self.run_manager_metrics.detection_to_csv(dir_name, tracker.detector.detections)
                self.run_manager_metrics.groundtruth_to_csv(dir_name,
                                                            self.check_ground_truth(ground_truth))

                if metric_manager is not None:
                    # Generate the metrics
                    metric_manager.add_data(self.check_ground_truth(ground_truth), ctracks,
                                            tracker.detector.detections,
                                            overwrite=False)
            if metric_manager is not None:
                metrics = metric_manager.generate_metrics()
                self.run_manager_metrics.metrics_to_csv(dir_name, metrics)
            self.logging_success(log_time)

        except Exception as e:
            os.rename(dir_name, dir_name + "_FAILED")
            self.logging_failed_simulation(log_time, e)

        finally:
            # Clear manager after run to stop subsequent runs slowing down
            del metric_manager
            del ground_truth
            del tracker

            print('--------------------------------')

    def set_trackers(self, combo_dict, tracker, ground_truth, metric_manager):
        """Set the trackers, groundtruths and metricmanagers list (stonesoup objects)

        Parameters
        ----------
        combo_dict : dict
            dictionary of all the possible combinations of values
        tracker : tracker
            stonesoup tracker
        ground_truth : groundtruth
            stonesoup groundtruth
        metric_manager : metricmanager
            stonesoup metric_manager

        Returns
        -------
        list:
            list of trackers
        list:
            list of groundtruths
        list:
            list of metric managers
        """

        trackers = []
        ground_truths = []
        metric_managers = []

        for parameter in combo_dict:
            tracker_copy, ground_truth_copy, metric_manager_copy = copy.deepcopy(
                (tracker, ground_truth, metric_manager))

            self.set_tracker_parameters(parameter, tracker_copy)
            trackers.append(tracker_copy)
            ground_truths.append(ground_truth_copy)
            metric_managers.append(metric_manager_copy)

        return trackers, ground_truths, metric_managers

    def set_tracker_parameters(self, parameter, tracker):
        """Sets the paramater value to the tracker.

        Parameters
        ----------
        parameter:
            the value to set
        tracker:
            the tracker to set the parameter to
        """
        for k, v in parameter.items():
            split_path = k.split('.')
            if len(split_path) > 1:
                split_path = split_path[1::]
            self.set_param(split_path, tracker, v)

    def set_param(self, split_path, el, value):
        """Sets the paramater value to the attribute in the stone soup object

        Parameters
        ----------
        split_path : str
            path to object attribute
        el : str
            tracker
        value : str
            attribute value
        """
        if len(split_path) > 1:
            newEl = getattr(el, split_path[0])
            self.set_param(split_path[1::], newEl, value)
        else:
            if len(split_path) > 0:
                setattr(el, split_path[0], value)

    def read_config_file(self, config_file):
        """
        Reads and loads configuration data from given config.yaml file.
        If user has added a groundtruth or metric manager in the config file,
        assume the second object is metric manager unless the user has set
        the groundtruth flag argument (-g True), in which case the second object
        is set as groundtruth. If there are three objects in the config file,
        always assume groundtruth is second object and metric manager is third.

        Parameters
        ----------
        config_file : str
            file path to configuration file
        Returns
        -------
        object dictionary with the loaded tracker, groundtruth and metric_manager
        """
        config_string = config_file.read()
        tracker, groundtruth, metric_manager = None, None, None

        try:
            config_data = YAML('safe').load(config_string)
        except Exception as e:
            print(f"{datetime.now()} Failed to load config data: {e}")
            info_logger.error(f"{datetime.now()} Failed to load config data: {e}")
            config_data = [None, None, None]

        tracker = config_data[0]

        # If config has more than just tracker (len > 1)
        #   If user has set flag for gt to be in config
        #     Set gt = config_data[1]
        #   Else set metric manager = config_data[1]
        # If groundtruth is None
        #   throw exception groundtruth not found

        if len(config_data) > 1:
            if self.groundtruth_setting is True:
                groundtruth = config_data[1]
            else:
                metric_manager = config_data[1]
        if len(config_data) > 2:
            groundtruth = config_data[1]
            metric_manager = config_data[len(config_data)-1]

        # Try to find groundtruth in tracker if not set
        if groundtruth is None:
            try:
                groundtruth = tracker.detector.groundtruth
            except Exception as e:
                print("Ground truth not found, error: ", e)
                print("Check -g command is set to True and Groundtruth is in config file")
                pass

        # User has set a flag to use the groundtruth added in config file
        # if len(config_data) > 1:
        #     if self.groundtruth_setting is True:
        #         groundtruth = config_data[1]
        #     else:
        #         # Try to find groundtruth and metric manager if user has not flagged
        #         try:
        #             if len(config_data) > 2:
        #                 groundtruth = config_data[1]
        #                 metric_manager = config_data[len(config_data)-1]
        #             else:
        #                 # groundtruth = tracker.detector.groundtruth
        #                 metric_manager = config_data[1]
        #         except Exception as e:
        #             print("Could not find groundtruth, error: ", e)
        #             print("Check -g command is set to True and Groundtruth is in config file")
        #             pass

        # print("TRACKER: ", tracker)
        # print("GROUNDTRUTH: ", groundtruth)
        # print("METRIC MANAGER: : ", metric_manager)

        return {self.TRACKER: tracker,
                self.GROUNDTRUTH: groundtruth,
                self.METRIC_MANAGER: metric_manager}

    def read_config_dir(self, config_dir):
        """Reads a directory and returns a list of all of the file paths

        Parameters
        ----------
        config_dir : str
            directory location

        Returns
        -------
        list
            returns a list of files paths from specified directory
        """
        if os.path.exists(config_dir):
            files = os.listdir(config_dir)
        else:
            return None
        return files

    def get_filepaths(self, directory):
        """Returns the filepaths for a specific directory

        Parameters
        ----------
        directory : str
            path to directory

        Returns
        -------
        list
            list of all file paths from specified directory
        """
        file_paths = []
        if os.path.exists(directory):
            for root, directories, files in os.walk(directory):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    file_paths.append(filepath)
                break
        return file_paths

    def get_config_and_param_lists(self, files):
        """Matches the config file and parameter file by name and pairs them together
        within a list

        Parameters
        ----------
        files : list
            List of file paths

        Returns
        -------
        List
            List of file paths pair together
        """
        pair = []
        pairs = []

        for file in files:
            if not pair:
                pair.append(file)
            elif file.startswith(pair[0].split('.', 1)[0]):
                pair.append(file)
                pairs.append(pair)
                pair = []
            else:
                pair = []

        for idx, path in enumerate(pairs):
            path = self.order_pairs(path)
            pairs[idx] = path

        return pairs

    def order_pairs(self, path):
        if path[0].endswith('yaml'):
            config_path = path[0]
            param_path = path[1]
        else:
            config_path = path[1]
            param_path = path[0]

        return [config_path, param_path]

    def prepare_single_simulation(self):
        """Prepares a single simulation run by setting tracker,
        groundtruth and metric manager with their components.
        Also sets up a multiprocessing pool of processes if
        multiprocessing is being used.
        """
        try:
            now = datetime.now()
            dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
            self.config_starttime = dt_string
            dt_string = dt_string + self.node
            components = self.set_components(self.config_path)
            tracker = components[self.TRACKER]
            ground_truth = components[self.GROUNDTRUTH]
            metric_manager = components[self.METRIC_MANAGER]

            if self.nprocesses > 1:
                # Execute runs in separate processes
                range_nruns = list(range(0, self.nruns))
                trackers = [tracker] * self.nruns
                ground_truths = [ground_truth] * self.nruns
                metric_managers = [metric_manager] * self.nruns
                dt_string_ = [dt_string] * self.nruns
                pool = Pool(self.nprocesses)
                pool.map(self.run_single_simulation, trackers, ground_truths, metric_managers,
                         range_nruns, dt_string_)
            else:
                for runs in range(self.nruns):
                    self.run_single_simulation(tracker, ground_truth, metric_manager,
                                               runs, dt_string)
                    # Each tracker object needs to be reset
                    components = self.set_components(self.config_path)
                    tracker = components[self.TRACKER]
                    ground_truth = components[self.GROUNDTRUTH]
                    metric_manager = components[self.METRIC_MANAGER]

        except Exception as e:
            print(f'{datetime.now()} Preparing simulation error: {e}')
            info_logger.error(f'Could not run simulation. error: {e}')

    def run_single_simulation(self, tracker, ground_truth, metric_manager, runs_num, dt_string):
        """Finallising setting current run parameters for a single simulation and then
        executes the simulation. Is ran in its own process if multiprocessing is used.

        Parameters
        ----------
        tracker : list
            a single tracker used in the simulation run
        ground_truth:
            the ground truth for the simulation run
        metric_manager:
            the metric manager for the simulation run
        runs_num : int
            the index of the current run
        dt_string : str
            string of the datetime for the metrics directory name
        """
        path, config = os.path.split(self.config_path)
        dir_name = f"{self.slurm_dir}{config}_{dt_string}/run_{runs_num + 1}{self.node}"
        self.run_manager_metrics.generate_config(dir_name, tracker, ground_truth, metric_manager)
        self.current_run = runs_num

        # ground_truth = self.check_ground_truth(ground_truth)
        simulation_parameters = dict(
            tracker=tracker,
            ground_truth=ground_truth,
            metric_manager=metric_manager
        )
        self.run_simulation(simulation_parameters,
                            dir_name)

    def prepare_monte_carlo_simulation(self, combo_dict, nruns, nprocesses, config_path):
        """Prepares multiple trackers for simulation run and run a multi-processor or a single
        processor simulation

        Parameters
        ----------
        combo_dict : dict
            dictionary of all parameter combinations for monte-carlo
        nruns : int
            Number of monte-carlo runs
        nprocesses : int
            Number of processor to be used
        config_path : string
            configuration path
        """
        # Load the tracker from the config file
        config_data = self.set_components(config_path)
        tracker = config_data[self.TRACKER]
        ground_truth = config_data[self.GROUNDTRUTH]
        metric_manager = config_data[self.METRIC_MANAGER]
        # Generate all the trackers from the loaded tracker
        trackers, ground_truths, metric_managers = self.set_trackers(
            combo_dict, tracker, ground_truth, metric_manager)
        self.total_trackers = len(trackers)
        try:
            now = datetime.now()
            dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
            self.config_starttime = dt_string
            dt_string = dt_string + self.node
            if nprocesses > 1:
                # Run with multiprocess
                pool = Pool(nprocesses)
                for runs in range(nruns):
                    self.total_runs = nruns
                    self.current_run = runs
                    dt_string_ = [dt_string] * len(trackers)
                    combo_dict_ = [combo_dict] * len(trackers)
                    runs_num_ = [runs] * len(trackers)
                    pool.map(self.run_monte_carlo_simulation, trackers, ground_truths,
                             metric_managers, dt_string_, combo_dict_,
                             range(0, len(trackers)), runs_num_)

            else:
                for runs in range(nruns):
                    self.total_runs = nruns
                    self.current_run = runs
                    for idx in range(0, len(trackers)):
                        self.run_monte_carlo_simulation(trackers[idx], ground_truths[idx],
                                                        metric_managers[idx], dt_string,
                                                        combo_dict, idx, runs)
                        config_data = self.set_components(config_path)
                        tracker = config_data[self.TRACKER]
                        ground_truth = config_data[self.GROUNDTRUTH]
                        metric_manager = config_data[self.METRIC_MANAGER]
        except Exception as e:
            info_logger.error(f'Could not run simulation. error: {e}')

    def run_monte_carlo_simulation(self, tracker, ground_truth, metric_manager,
                                   dt_string, combo_dict, idx, runs_num):
        """Finallising setting current run parameters for montecarlo simulations and then
        executes the simulation. Is ran in its own process if multiprocessing is used.

        Parameters
        ----------
        tracker : list
            a single tracker used in the simulation run
        ground_truth:
            the ground truth for the simulation run
        metric_manager:
            the metric manager for the simulation run
        dt_string : str
            string of the datetime for the metrics directory name
        combo_dict : dict
            dictionary of all the possible combinations of values
        idx : int
            the index of current simulation in current run
        runs_num : int
            the index of the current run
        """
        self.current_trackers = idx
        path, config = os.path.split(self.config_path)
        dir_name = f"{self.slurm_dir}{config}_{dt_string}/" + \
            f"simulation_{idx}/run_{runs_num + 1}{self.node}"
        self.run_manager_metrics.parameters_to_csv(dir_name, combo_dict[idx])
        self.run_manager_metrics.generate_config(dir_name, tracker, ground_truth, metric_manager)
        simulation_parameters = dict(
            tracker=tracker,
            ground_truth=ground_truth,
            metric_manager=metric_manager
        )
        self.run_simulation(simulation_parameters, dir_name)

    def set_components(self, config_path):
        """Sets the tracker, ground truth and metric manager to the correct variables
        from the configuration file.

        Parameters
        ----------
        config_path : str
            path to configuration
        Returns
        -------
        Object:
            TRACKER: tracker,
            GROUNDTRUTH: ground_truth,
            METRIC_MANAGER: metric_manager
        """

        try:
            tracker, ground_truth, metric_manager = None, None, None
            with open(config_path, 'r') as file:
                config_data = self.read_config_file(file)

            tracker = config_data[self.TRACKER]
            ground_truth = config_data[self.GROUNDTRUTH]
            metric_manager = config_data[self.METRIC_MANAGER]
        except Exception as e:
            print(f'{datetime.now()} Could not read config file: {e}')
            info_logger.error(f'Could not read config file: {e}')

        return {self.TRACKER: tracker,
                self.GROUNDTRUTH: ground_truth,
                self.METRIC_MANAGER: metric_manager}

    def logging_starting(self, log_time):
        """Handles logging and output for messages regarding the start
        of a simulation.

        Parameters
        ----------
        log_time : str
            timestamp of log information
        """
        if self.total_trackers > 1:
            info_logger.info(f"Starting simulation {self.current_trackers + 1}"
                             f" / {self.total_trackers} and monte-carlo"
                             f" {self.current_run + 1} / {self.nruns}")
        else:
            info_logger.info(f"Starting simulation"
                             f" {self.current_run + 1} / {self.nruns}")

    def logging_success(self, log_time):
        """Handles logging and output for messages regarding successful
        simulation runs.

        Parameters
        ----------
        log_time : str
            timestamp of log information
        """
        if self.total_trackers > 1:
            info_logger.info(f"Successfully ran simulation {self.current_trackers + 1} /"
                             f" {self.total_trackers} and monte-carlo"
                             f" {self.current_run + 1} / {self.nruns}"
                             f" in {datetime.now() - log_time}")
        else:
            info_logger.info(f"Successfully ran simulation {self.current_run + 1} /"
                             f" {self.nruns} in {datetime.now() - log_time}")

    def logging_failed_simulation(self, log_time, e):
        """Handles logging and output for messages regarding failed simulation
        runs.

        Parameters
        ----------
        log_time : datetime
            timestamp of log information
        """
        if self.total_trackers > 1:
            info_logger.info(f"Failed to run Simulation {self.current_trackers + 1} /"
                             f" {self.total_trackers} and monte-carlo"
                             f" {self.current_run} / {self.nruns}"
                             f" in {datetime.now() - log_time}")
            info_logger.exception(f"{e}")

            print(f"Failed to run Simulation {self.current_trackers + 1} /"
                  f" {self.total_trackers} and monte-carlo"
                  f" {self.current_run} / {self.nruns}"
                  f" in {datetime.now() - log_time}")
            print(f"{e}")

        else:
            info_logger.error(f"Failed to run Simulation"
                              f" {self.current_run + 1} / {self.nruns}")
            info_logger.exception(f"{e}")
            print(f"{datetime.now()}: Failed to run Simulation"
                  f" {self.current_run + 1} / {self.nruns}: {e}")

    def logging_metric_manager_fail(self, e):
        info_logger.error(f'Metric manager error: {e}')
        print(f'{datetime.now()} Metric manager error: {e}')


def setup_logger(name, log_file, level=logging.INFO):
    """ Create new Logging files which work with multiprocessing.

    Parameters
    ----------
    log_file : str
        file name
    level : logging, optional
        Level of logging required, by default logging.INFO

    Returns
    -------
    logger : Object
    The logger object to log outputs to a file
    """
    # TODO: set normal logger for non multiprocessing
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


info_logger = setup_logger('info_logger', 'simulation_info.log')
