import copy
import json
import logging
from datetime import datetime

import os

import numpy as np
import multiprocessing as mp


from stonesoup.serialise import YAML
from .inputmanager import InputManager
from .runmanagermetrics import RunmanagerMetrics
from .base import RunManager


class RunManagerCore(RunManager):
    TRACKER = "tracker"
    GROUNDTRUTH = "ground_truth"
    METRIC_MANAGER = "metric_manager"

    def __init__(self, config_path, parameters_path, groundtruth_setting, dir):

        self.config_path = config_path
        self.parameters_path = parameters_path
        self.groundtruth_setting = groundtruth_setting
        self.dir = dir

        self.input_manager = InputManager()
        self.run_manager_metrics = RunmanagerMetrics()

        logging.basicConfig(filename='simulation.log', encoding='utf-8', level=logging.INFO)
        logging.info(f'RunManagerCore started. {datetime.now()}')

    def read_json(self, json_input):
        """Read json file from directory

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

    def run(self, nruns=1, nprocesses=1):
        """Handles the running of multiple files, single files and defines the structure
        of the run.

        Parameters
        ----------
        nruns : int, optional
            number of monte-carlo runs, by default 1
        nprocesses : int, optional
            number of processing cores to use, by default 1
        """
        pairs = self.config_parameter_pairing()

        # Single simulation. No param file detected
        if self.config_path and self.parameters_path is None:
            if nruns is None:
                nruns = 1
            self.prepare_and_run_single_simulation(self.config_path,
                                                self.groundtruth_setting,
                                                nruns)
            logging.info(f'{datetime.now()} Ran single run successfully.')

        for path in pairs:
            # Read the param data
            config_path = path[1]
            param_path = path[0]
            json_data = self.read_json(param_path)

            nruns = self.set_runs_number(nruns, json_data)
            combo_dict = self.prepare_monte_carlo(json_data)
            
            self.run_monte_carlo_simulation(combo_dict, nruns,
                                            nprocesses, config_path)

    def set_runs_number(self, nruns, json_data):
        if nruns is None:
            if json_data['configuration']['runs_num']:
                nruns = json_data['configuration']['runs_num']
        else:
            nruns = 1
        return nruns

    def prepare_monte_carlo(self, json_data):
        # Generate all the parameters for the monte carlo run
        trackers_combination_dict = self.input_manager.generate_parameters_combinations(
            json_data["parameters"])
        # Generate all the the possible combinations with the parameters
        combo_dict = self.input_manager.generate_all_combos(trackers_combination_dict)
        return combo_dict

        # logging.info(f'All simulations completed. Time taken to run: {datetime.now() - now}')

    def config_parameter_pairing(self):
        pairs = []
        if self.dir:
            paths = self.get_filepaths(self.dir)

            pairs = self.get_config_and_param_lists(paths)

        elif self.config_path and self.parameters_path:
            pairs = [[self.parameters_path, self.config_path]]

        elif self.dir and self.config_path and self.parameters_path:
            paths = self.get_filepaths(self.dir)
            pairs = self.get_config_and_param_lists(paths)
            pairs.append([self.parameters_path, self.config_path])
            # logging.info(f'All simulations completed. Time taken to run: {datetime.now() - now}')

        return pairs

    def check_ground_truth(self, ground_truth):
        try:
            ground_truth = ground_truth.groundtruth_paths
        except Exception:
            ground_truth = ground_truth
        return ground_truth

    def run_simulation(self, simulation_parameters, dir_name):
        """Runs a single simulation

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

        log_time = datetime.now()
        try:
            timeFirst = datetime.now()
            for time, ctracks in tracker.tracks_gen():
                # Update groundtruth, tracks and detections
                self.run_manager_metrics.groundtruth_to_csv(dir_name,
                                                            self.check_ground_truth(ground_truth))
                self.run_manager_metrics.tracks_to_csv(dir_name, ctracks)
                self.run_manager_metrics.detection_to_csv(dir_name, tracker.detector.detections)
                if metric_manager is not None:
                    # Generate the metrics
                    metric_manager.add_data(self.check_ground_truth(ground_truth), ctracks,
                                            tracker.detector.detections,
                                            overwrite=False)

                    # Sometimes the metric manager generate metrics fails.
                    # We can't remove this try catch.
                    try:
                        metrics = metric_manager.generate_metrics()
                        self.run_manager_metrics.metrics_to_csv(dir_name, metrics)
                    except Exception as e:
                        print("Metric manager: {}".format(e))

            timeAfter = datetime.now()
            timeTotal = timeAfter-timeFirst
            print(timeTotal)
        except Exception as e:
            logging.error(f'{log_time}: Failed to run Simulation: {e}')

        else:
            logging.info(f'{log_time} Successfully ran simulation in {datetime.now() - log_time} ')
            print('Success!', flush=True)

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
        for k, v in parameter.items():
            split_path = k.split('.')
            if len(split_path) > 1:
                split_path = split_path[1::]
            self.set_param(split_path, tracker, v)

    def set_param(self, split_path, el, value):
        """Sets the paramater file value to the attribute in the stone soup object

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
        """[summary]

        Parameters
        ----------
        config_file : str
            file path to configuration file
        groundtruth_setting : bool
            checks if ground truth exists

        Returns
        -------
        object dictionary with tracker, groundtruth and metric_manager
        """
        config_string = config_file.read()
        tracker, groundtruth, metric_manager = None, None, None

        try:
            config_data = YAML('safe').load(config_string)
        except Exception as e:
            print("Failed to load config data: ", e)
            config_data = [None, None, None]

        tracker = config_data[0]

        # User has set a flag to use the groundtruth added in config file
        if self.groundtruth_setting is True:
            groundtruth = config_data[1]
            # Also set metric manager if it has been added in config file
            # (Change this to flag setting too?)
            if len(config_data) > 2:
                metric_manager = config_data[2]
        else:
            # Try to find groundtruth and metric manager if user has not flagged
            for x in config_data:
                if "GroundTruth" in str(type(x)) or "MovingPlatform" in str(type(x)):
                    groundtruth = x
                elif "metricgenerator" in str(type(x)):
                    metric_manager = x

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
        return pairs

    def prepare_and_run_single_simulation(self, nruns):

        """Prepares a single simulation for a run

        Parameters
        ----------
        config_path : str
            path to configuration
        groundtruth_setting : bool
            Defines if ground truth is present
        nruns : int
            Number of monte-carlo runs
        """
        try:
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            components = self.set_components(self.config_path)
            tracker = components[self.TRACKER]
            ground_truth = components[self.GROUNDTRUTH]
            metric_manager = components[self.METRIC_MANAGER]
            for runs in range(nruns):
                dir_name = f"metrics_{dt_string}/run_{runs}"
                print("RUN")
                # ground_truth = self.check_ground_truth(ground_truth)
                simulation_parameters = dict(
                    tracker=tracker,
                    ground_truth=ground_truth,
                    metric_manager=metric_manager
                )

                self.run_simulation(simulation_parameters,
                                    dir_name)

        except Exception as e:
            print(f'{datetime.now()} Preparing simulation error: {e}')
            logging.error(f'{datetime.now()} Could not run simulation. error: {e}')

    def run_monte_carlo_simulation(self, combo_dict, nruns, nprocesses, config_path):
        """Prepares multiple trackers for simulation runs

        Parameters
        ----------
        combo_dict : dict
            dictionary of all parameter combinations for monte-carlo
        nruns : int
            Number of monte-carlo runs
        """

        # Load the tracker from the config file
        config_data = self.set_components(config_path)

        tracker = config_data[self.TRACKER]
        ground_truth = config_data[self.GROUNDTRUTH]
        metric_manager = config_data[self.METRIC_MANAGER]

        # Generate all the trackers from the loaded tracker
        trackers, ground_truths, metric_managers = self.set_trackers(
            combo_dict, tracker, ground_truth, metric_manager)

        try:
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            for runs_num in range(nruns):
                if nprocesses > 1:
                    mp_args = [(trackers[idx], ground_truths[idx], metric_managers[idx],
                               dt_string) for idx in range(0, len(trackers))]
                    pool = mp.Pool(nprocesses)
                    pool.starmap(self.run_multi_process_simulation, mp_args)
                for idx in range(0, len(trackers)):
                    dir_name = f"metrics_{dt_string}/simulation_{idx}/run_{runs_num}"
                    self.run_manager_metrics.parameters_to_csv(dir_name, combo_dict[idx])
                    self.run_manager_metrics.generate_config(
                        dir_name, trackers[idx], ground_truths[idx], metric_managers[idx])
                    print("RUN")

                    simulation_parameters = dict(
                        tracker=trackers[idx],
                        ground_truth=ground_truths[idx],
                        metric_manager=metric_managers[idx]
                    )

                    self.run_simulation(simulation_parameters,
                                        dir_name)
        except Exception as e:
            print(f'{datetime.now()} Preparing simulation error: {e}')
            logging.error(f'{datetime.now()} Could not run simulation. error: {e}')

    def run_multi_process_simulation(self, combo_dict, nruns, config_path):
        pass

    def set_components(self, config_path):
        """Sets the tracker, ground truth and metric manager to the correct variables
        from the configuration file.

        Parameters
        ----------
        config_path : str
            path to configuration
        groundtruth_setting : bool
            Defines if ground truth is present

        Returns
        -------
        Tracker:
            Tracker stone soup object
        GroundTruth:
            Ground Truth stone soup object
        MetricManager:
            Metric manager stone soup object
        """
        tracker, ground_truth, metric_manager = None, None, None
        try:
            with open(config_path, 'r') as file:
                config_data = self.read_config_file(file)
            tracker = config_data[self.TRACKER]
            ground_truth = config_data[self.GROUNDTRUTH]
            metric_manager = config_data[self.METRIC_MANAGER]
        except Exception as e:
            print(f'{datetime.now()} Could not read config file: {e}')
            logging.error(f'{datetime.now()} Could not read config file: {e}')
        return {self.TRACKER: tracker,
                self.GROUNDTRUTH: ground_truth,
                self.METRIC_MANAGER: metric_manager}


