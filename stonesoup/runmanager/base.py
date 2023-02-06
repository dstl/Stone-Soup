import copy
import json
import logging
import time
import glob

import os
import multiprocessing

from pathos.multiprocessing import ProcessPool as Pool
from datetime import datetime
from stonesoup.serialise import YAML
from .inputmanager import InputManager
from .runmanagermetrics import RunmanagerMetrics


class RunManager:
    """
    Run Manager Core base class

    Core RunManager class that contains all functionality for;
    - loading configuration and parameter files,
    - generating and setting parameters,
    - running a number of montecarlo simulations,
    - with or without multiprocessing.
    """
    TRACKER = "tracker"
    GROUNDTRUTH = "ground_truth"
    METRIC_MANAGER = "metric_manager"
    args = dict(config=None, parameters=None, config_dir=None, nruns=None,
                montecarlo=False, n_processes=None, output_dir=None)

    def __init__(self, rm_args=args):
        """The init function for RunManagerCore, initiating the key settings to allow
        the running of simulations.

        Parameters
        ----------
        rm_args : dict
            A dictionary of run manager parameters including the following;
                config : str
                    The path to the configuration file containing the tracker
                parameters : str
                    The path to the parameters json file containing the parameters for a
                    configuration
                config_dir : str
                    The path to the directory containing configuration and parameter pairs
                nruns : int, optional
                    number of monte-carlo runs. (Default is 1)
                montecarlo : bool
                    Not implemented yet. A boolean to indicate if montecarlo simulations are to be
                    used. (Default is False)
                n_processes : int, optional
                    number of processing cores to use. (Default is 1)
                output_dir : str
                    The path to store all run manager output files
        """

        self.config_path = rm_args.get("config")
        self.parameters_path = rm_args.get("parameters")
        self.config_dir = rm_args.get("config_dir")
        self.nruns = rm_args.get("nruns")
        self.montecarlo = rm_args.get("montecarlo")  # Not implemented yet
        self.nprocesses = rm_args.get("n_processes")
        self.output_dir = rm_args.get("output_dir")

        if self.output_dir is None:
            self.output_dir = ""

        self.total_trackers = 0
        self.current_run = 0
        self.total_runs = 0
        self.current_trackers = 0

        self.config_starttime = ""
        self.parameter_details_log = dict()

        self.input_manager = InputManager()
        self.run_manager_metrics = RunmanagerMetrics()
        info_logger.info(f'RunManager started. {datetime.now()}')
        info_logger.info(f'RunManager Output located in: {self.output_dir}')

    @staticmethod
    def read_json(json_input):
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
            self.run_single_config()
            self.average_metrics()
        else:
            pairs = self.config_parameter_pairing()
            if not pairs and not self.config_path:
                info_logger.info(f"{datetime.now()} No files in " +
                                 f"current directory: {self.config_dir}")

            for path in pairs:
                # Read the param data
                if len(path) == 1:
                    self.config_path = path[0]
                    info_logger.info(f'Running {self.config_path}')
                    self.parameters_path = None
                    self.run_single_config()
                else:
                    self.config_path, param_path = path[0], path[1]
                    info_logger.info(f'Running {self.config_path}')
                    json_data = self.read_json(param_path)
                    self.nruns = self.set_runs_number(self.nruns, json_data)
                    nprocesses = self.set_processes_number(self.nprocesses, json_data)
                    combo_dict = self.prepare_monte_carlo(json_data)
                    self.prepare_monte_carlo_simulation(combo_dict, self.nruns,
                                                        nprocesses, self.config_path)
                self.average_metrics()
                self.total_trackers = 0

        # End timer
        end = time.time()
        info_logger.info(f"{datetime.now()} Finished all simulations in " +
                         f"--- {end - start} seconds ---")
        # Average all of the metrics at the end

    def run_single_config(self):
        """
        Prepares running a single simulation for a single config file.
        """
        if self.nruns is None:
            self.nruns = 1
        if self.nprocesses is None:
            self.nprocesses = 1
        self.prepare_single_simulation()

    def average_metrics(self, batch_size=200):
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

        start = time.time()
        path, config = os.path.split(self.config_path)

        try:
            info_logger.info(f"{datetime.now()} Averaging metrics for all Monte-Carlo Simulations")
            directory = glob.glob(f'./{self.output_dir}/{config}_{self.config_starttime}'
                                  f'*/simulation*',
                                  recursive=False)
            if directory:
                for simulation in directory:
                    summed_df, sim_amt = self.run_manager_metrics.sum_simulations(simulation,
                                                                                  batch_size)
                    df = self.run_manager_metrics.average_simulations(summed_df, sim_amt)
                    df.to_csv(f"./{simulation}/average.csv", index=False)
            else:
                if self.output_dir != "":
                    directory = glob.glob(f'{self.output_dir}/{config}_{self.config_starttime}*',
                                          recursive=False)
                    for node in directory:
                        summed_df, sim_amt = self.run_manager_metrics.sum_simulations(node,
                                                                                      batch_size)
                        df = self.run_manager_metrics.average_simulations(summed_df, sim_amt)
                        df.to_csv(f"./{node}/average.csv", index=False)
                else:
                    directory = glob.glob(f'{config}_{self.config_starttime}*', recursive=False)
                    summed_df, sim_amt = self.run_manager_metrics.sum_simulations(directory,
                                                                                  batch_size)
                    df = self.run_manager_metrics.average_simulations(summed_df, sim_amt)
                    df.to_csv(f"./{config}_{self.config_starttime}/average.csv", index=False)
            end = time.time()
            info_logger.info(f"{datetime.now()} Finished Averaging in " +
                             f"--- {end - start} seconds ---")

        except Exception as f:
            info_logger.error(f"{datetime.now()} No metrics exist for simulations. "
                              f"Failed to average simulations. {f}")
            print(f"{datetime.now()} No metrics exist for simulations. "
                  f"Failed to average simulations.")

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
            except Exception as f:
                info_logger.error(f, "runs_num value from json not found, defaulting to 1")
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
            except Exception as f:
                info_logger.error(f, "proc_num value from json not found, defaulting to 1")
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
        list
            list that contains Tuple pairs of config path and parameter path
            (config_path, parameter_path)
        """
        pairs = []
        if self.config_dir:
            pairs = self.get_config_and_param_lists()
            if self.config_path and self.parameters_path:
                if [self.config_path, self.parameters_path] not in pairs:
                    pairs.append((self.config_path, self.parameters_path))
        else:
            if self.config_path and self.parameters_path:
                pairs = [(self.config_path, self.parameters_path)]
        return pairs

    @staticmethod
    def check_ground_truth(ground_truth):
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
        except (ValueError, Exception):
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
        tracker_status = None
        metric_status = None
        fail_status = ""
        log_time = datetime.now()
        try:
            self.logging_starting(log_time)
            for _, ctracks in tracker:
                self.run_manager_metrics.tracks_to_csv(dir_name, ctracks)
                self.run_manager_metrics.detection_to_csv(dir_name, tracker.detector.detections)
                self.run_manager_metrics.groundtruth_to_csv(dir_name,
                                                            self.check_ground_truth(ground_truth))

                if metric_manager is not None:
                    # Generate the metrics
                    metric_manager.add_data(self.check_ground_truth(ground_truth), ctracks,
                                            tracker.detector.detections,
                                            overwrite=False)
            try:
                if metric_manager is not None:
                    metrics = metric_manager.generate_metrics()
                    self.run_manager_metrics.metrics_to_csv(dir_name, metrics)
                    metric_status = "Success"
                else:
                    metric_status = "Not Applicable"
                    info_logger.error("No Metric Manager provided in Config file.")
            except Exception as e:
                os.rename(dir_name, dir_name + "_!FAILED")
                fail_status = e
                metric_status = "Failed"
                self.logging_failed_simulation(log_time, e)

            self.logging_success(log_time)
            tracker_status = "Success"
        except Exception as f:
            os.rename(dir_name, dir_name + "_!FAILED")
            tracker_status = "Failed"
            metric_status = "Failed"
            fail_status = f
            self.logging_failed_simulation(log_time, f)

        finally:
            # Clear manager after run to stop subsequent runs slowing down
            del metric_manager
            del ground_truth
            del tracker
            print('--------------------------------')
            return tracker_status, metric_status, fail_status

    def set_trackers(self, combo_dict, tracker, ground_truth, metric_manager):
        """Set the trackers, groundtruths and metricmanagers list (stonesoup objects)

        Parameters
        ----------
        combo_dict : Union[dict, Sequence[dict]]
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
        """Sets the parameter value to the tracker.

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
        """Sets the parameter value to the attribute in the stone soup object

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

    def read_config_file(self):
        """
        Reads and loads configuration data from given config.yaml file.
        If user has added a groundtruth or metric manager in the config file,
        assume the second object is metric manager unless the user has set
        the groundtruth flag argument (-g True), in which case the second object
        is set as groundtruth. If there are three objects in the config file,
        always assume groundtruth is second object and metric manager is third.

        Returns
        -------
        object dictionary with the loaded tracker, groundtruth and metric_manager
        """
        try:
            with open(self.config_path, 'r') as file:
                config_data = YAML(typ='safe').load(file.read())
            file.close()
        except Exception as er:
            info_logger.error(f"{datetime.now()} Failed to load config data: {er}")
            config_data = {"tracker": None, "groundtruth": None, "metricmanager": None}
            exit()
        tracker = config_data.get("tracker")
        groundtruth = config_data.get("groundtruth")
        metric_manager = config_data.get("metricmanager")

        # Try to find groundtruth in tracker if not set
        if groundtruth is None:
            try:
                groundtruth = tracker.detector.groundtruth
            except Exception as err:
                info_logger.error(f"Ground truth not found, error: {err}")
                info_logger.error("Check groundtruth is stored in the tracker in config file.")

        return {self.TRACKER: tracker,
                self.GROUNDTRUTH: groundtruth,
                self.METRIC_MANAGER: metric_manager}

    @staticmethod
    def read_config_dir(config_dir):
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

    @staticmethod
    def get_filepaths(directory, excludesubfolders=True):
        """Returns the filepaths for a specific directory

        Parameters
        ----------
        directory : str
            Path to directory
        excludesubfolders : bool
            Flag to exclude filepaths from subfolders. Default is True.

        Returns
        -------
        list
            List of all file paths from specified directory
        """
        file_paths = []
        if os.path.exists(directory):
            for root, directories, files in os.walk(directory):
                if excludesubfolders:
                    directories.clear()
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        file_paths.append(filepath)
        return file_paths

    def get_config_and_param_lists(self):
        """Matches the config file and parameter file by name and pairs them together
        within a list

        Returns
        -------
        List
            List of file paths pair together
        """

        pairs = set()
        files = self.get_filepaths(self.config_dir)
        for search_file in files:
            split = search_file.split(".", 1)
            for file in files:
                if search_file is not file:
                    if split[1] == "json":
                        json_split = split[0].split("_parameters")[0]
                        if file == json_split + ".yaml":
                            pairs.add((file, search_file))
                    elif split[1] == "yaml":
                        if file == split[1] + ".json" or file == split[1] + "_parameters.json":
                            pairs.add((search_file, file))
                    else:
                        print("Error: File is not a configuration or parameter file.")

        return sorted(list(pairs))

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
            components = self.read_config_file()
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
                    components = self.read_config_file()
                    tracker = components[self.TRACKER]
                    ground_truth = components[self.GROUNDTRUTH]
                    metric_manager = components[self.METRIC_MANAGER]

        except Exception as e:
            print(f'{datetime.now()} Preparing simulation error: {e}')
            info_logger.error(f'Could not run simulation. error: {e}')

    def run_single_simulation(self, tracker, ground_truth, metric_manager, runs_num, dt_string):
        """Finalising setting current run parameters for a single simulation and then
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
        dir_name = f"{self.output_dir}/{config}_{dt_string}/run_{runs_num + 1}"
        self.run_manager_metrics.generate_config(dir_name, tracker, ground_truth, metric_manager)
        self.current_run = runs_num

        simulation_parameters = dict(
            tracker=tracker,
            ground_truth=ground_truth,
            metric_manager=metric_manager
        )
        tracker_status, metric_status, fail_status = self.run_simulation(simulation_parameters,
                                                                         dir_name)

        self.parameter_details_log["Monte-Carlo Run"] = runs_num + 1
        self.parameter_details_log["Tracking Status"] = tracker_status
        self.parameter_details_log["Metric Status"] = metric_status
        self.parameter_details_log["Fail Status"] = fail_status
        self.run_manager_metrics.create_summary_csv(f"{self.output_dir}{config}_{dt_string}",
                                                    self.parameter_details_log)

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
        config_data = self.read_config_file()
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
                        config_data = self.read_config_file()
                        tracker = config_data[self.TRACKER]
                        ground_truth = config_data[self.GROUNDTRUTH]
                        metric_manager = config_data[self.METRIC_MANAGER]

        except Exception as e:
            info_logger.error(f'Could not run simulation. error: {e}')

    def run_monte_carlo_simulation(self, tracker, ground_truth, metric_manager,
                                   dt_string, combo_dict, idx, runs_num):
        """Finalising setting current run parameters for montecarlo simulations and then
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
        dir_name = f"{self.output_dir}{config}_{dt_string}/" + \
            f"simulation_{idx}/run_{runs_num + 1}"
        self.run_manager_metrics.generate_config(dir_name, tracker, ground_truth, metric_manager)
        simulation_parameters = dict(
            tracker=tracker,
            ground_truth=ground_truth,
            metric_manager=metric_manager
        )
        tracker_status, metric_status, fail_status = self.run_simulation(simulation_parameters,
                                                                         dir_name)

        self.parameter_details_log = dict()
        self.parameter_details_log["Simulation ID"] = idx
        self.parameter_details_log["Monte-Carlo Run"] = runs_num + 1
        for key, value in combo_dict[idx].items():
            self.parameter_details_log[key] = value
        self.parameter_details_log["Tracking Status"] = tracker_status
        self.parameter_details_log["Metric Status"] = metric_status
        self.parameter_details_log["Fail Status"] = fail_status
        self.run_manager_metrics.create_summary_csv(f"{self.output_dir}{config}_{dt_string}",
                                                    self.parameter_details_log)

    def logging_starting(self, log_time):
        """Handles logging and output for messages regarding the start
        of a simulation.

        Parameters
        ----------
        log_time : datetime.datetime
            timestamp of log information
        """
        if self.total_trackers > 1:
            info_logger.info(f"{log_time}: Starting simulation {self.current_trackers + 1}"
                             f" / {self.total_trackers} and monte-carlo"
                             f" {self.current_run + 1} / {self.nruns}")
        else:
            info_logger.info(f"{log_time}: Starting monte-carlo run"
                             f" {self.current_run + 1} / {self.nruns}")

    def logging_success(self, log_time):
        """Handles logging and output for messages regarding successful
        simulation runs.

        Parameters
        ----------
        log_time : datetime.datetime
            timestamp of log information
        """
        if self.total_trackers > 1:
            info_logger.info(f"Successfully ran simulation {self.current_trackers + 1} /"
                             f" {self.total_trackers} and monte-carlo"
                             f" {self.current_run + 1} / {self.nruns}"
                             f" in {datetime.now() - log_time}")
        else:
            info_logger.info(f"Successfully ran monte-carlo {self.current_run + 1} /"
                             f" {self.nruns} in {datetime.now() - log_time}")

    def logging_failed_simulation(self, log_time, e):
        """Handles logging and output for messages regarding failed simulation
        runs.

        Parameters
        ----------
        log_time : datetime
            timestamp of log information
        e : error
            error generated from exception
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
            info_logger.error(f"Failed to run Monte-Carlo"
                              f" {self.current_run + 1} / {self.nruns}")
            info_logger.exception(f"{e}")
            print(f"{datetime.now()}: Failed to run Monte-Carlo"
                  f" {self.current_run + 1} / {self.nruns}: {e}")

    @staticmethod
    def logging_metric_manager_fail(error):
        info_logger.error(f'Metric manager error: {error}')
        print(f'{datetime.now()} Metric manager error: {error}')


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


if os.path.exists('./simulation_info.log'):
    try:
        os.remove('./simulation_info.log')
    except Exception as e:
        print(f"File not able to be removed. {e}")
else:
    print("Simulation log does not exist.")
info_logger = setup_logger('info_logger', 'simulation_info.log')
