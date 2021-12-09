import copy
import json
import logging
from datetime import datetime
import numpy as np
import os

from stonesoup.serialise import YAML
from .inputmanager import InputManager
from .runmanagermetrics import RunmanagerMetrics
from .base import RunManager


class RunManagerCore(RunManager):
    def __init__(self):
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

    def run(self, config_path, parameters_path,
            groundtruth_setting, dir, nruns=1, nprocesses=1):
        """Handles the running of multiple files, single files and defines the structure
        of the run.

        Parameters
        ----------
        config_path : str
            path to configuration file
        parameters_path : str
            path to parameter file
        groundtruth_setting : bool
            define if ground truth exists
        dir : str
            directory of configuration files
        nruns : int, optional
            number of monte-carlo runs
        nprocesses : int, optional
            number of processing cores to use
        """
        pairs = []
        input_manager = InputManager()
        now = datetime.now()

        if dir:
            paths = self.get_filepaths(dir)
            pairs = self.get_config_and_param_lists(paths)

        elif config_path and parameters_path:
            pairs = [[parameters_path, config_path]]

        elif dir and config_path and parameters_path:
            paths = self.get_filepaths(dir)
            pairs = self.get_config_and_param_lists(paths)
            pairs.append([parameters_path, config_path])

        elif config_path and parameters_path is None:
            if nruns is None:
                nruns = 1
            self.prepare_and_run_single_sim(config_path, groundtruth_setting, nruns)
            logging.info(f'{datetime.now()} Ran single run successfully.')

        for path in pairs:
            # add check file type
            param_path = path[0]
            config_path = path[1]
            json_data = self.read_json(param_path)
            if nruns is None:
                if json_data['configuration']['runs_num']:
                    nruns = json_data['configuration']['runs_num']
                else:
                    nruns = 1
            trackers_combination_dict = input_manager.generate_parameters_combinations(
                json_data["parameters"])

            combo_dict = input_manager.generate_all_combos(trackers_combination_dict)
            self.prepare_and_run_multi_sim(config_path, combo_dict, groundtruth_setting, nruns)
            logging.info(f'All simulations completed. Time taken to run: {datetime.now() - now}')

    def run_simulation(self, tracker, ground_truth,
                       metric_manager, dir_name):
        """Runs a simulation

        Parameters
        ----------
        tracker : Tracker
            Stonesoup tracker object
        ground_truth : GroundTruth
            ground truth object, can be csv
        metric_manager : MetricManager
            Metric manager object
        dir_name : str
            output directory for metrics
        """

        log_time = datetime.now()
        try:
            timeFirst = datetime.now()
            for time, ctracks in tracker.tracks_gen():
                # Update groundtruth, tracks and detections

                try:
                    RunmanagerMetrics.groundtruth_to_csv(dir_name, ground_truth.groundtruth_paths)
                except Exception as e:
                    logging.error(e)
                    try:
                        RunmanagerMetrics.groundtruth_to_csv(dir_name, ground_truth)
                    except Exception as e:
                        logging.error(e)
                        pass
                # tracks.update(ctracks)
                # detections.update(tracker.detector.detections)

                RunmanagerMetrics.tracks_to_csv(dir_name, ctracks)
                RunmanagerMetrics.detection_to_csv(dir_name, tracker.detector.detections)

                try:
                    if metric_manager is not None:
                        # Generate the metrics
                        try:
                            metric_manager.add_data(ground_truth.groundtruth_paths,
                                                    ctracks, tracker.detector.detections,
                                                    overwrite=False)
                        except Exception:
                            metric_manager.add_data(ground_truth,
                                                    ctracks,
                                                    tracker.detector.detections,
                                                    overwrite=False)
                except Exception as e:
                    print(f"Error with metric manager. {e}")
                    logging.error(f"{datetime.now()}, Error: {e}")
            try:
                metrics = metric_manager.generate_metrics()
                RunmanagerMetrics.metrics_to_csv(dir_name, metrics)
            except Exception as e:
                print("Metric manager: {}".format(e))
            timeAfter = datetime.now()

            timeTotal = timeAfter-timeFirst
            print(timeTotal)
        except Exception as e:
            logging.error(f'{log_time}: Failed to run Simulation: {e}', flush=True)

        else:
            # logging.info(f'{log_time}: Simulation {index} / {len(combos)-1} ran '
            #              'successfully in {datetime.now() - log_time}.'
            #              'With Parameters: {combos[index]}')
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

            for k, v in parameter.items():
                split_path = k.split('.')
                if len(split_path) > 1:
                    split_path = split_path[1::]
                self.set_param(split_path, tracker_copy, v)
            trackers.append(tracker_copy)
            ground_truths.append(ground_truth_copy)
            metric_managers.append(metric_manager_copy)

        return trackers, ground_truths, metric_managers

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

    def read_config_file(self, config_file, groundtruth_setting):
        """[summary]

        Parameters
        ----------
        config_file : str
            file path to configuration file
        groundtruth_setting : bool
            checks if ground truth exists

        Returns
        -------
        str
            Tracker
        str
            Ground Truth
        str
            Metric manager
        """
        config_string = config_file.read()

        tracker, gt, mm, csv_data = None, None, None, None

        config_data = YAML('safe').load(config_string)

        # Set explicitly if user has included groundtruth in config and set flag
        if groundtruth_setting is True:
            print("MANUAL READ")
            tracker = config_data[0]
            gt = config_data[1]
            if len(config_data) > 2:
                mm = config_data[2]
        else:
            print("AUTOMATIC READ")
            for x in config_data:
                if "Tracker" in str(type(x)):
                    tracker = x
                elif "GroundTruth" in str(type(x)):
                    gt = x
                elif "metricgenerator" in str(type(x)):
                    mm = x
                elif type(x) is np.ndarray:
                    csv_data = x
                    print("CSV data found")

        return tracker, gt, mm

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
        file_paths =[]
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
    
    def prepare_and_run_single_sim(self, config_path, groundtruth_setting, nruns):
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
            tracker, ground_truth, metric_manager = self.set_components(config_path, groundtruth_setting)
            for runs in range(nruns):
                dir_name = f"metrics_{dt_string}/run_{runs}"
                print("RUN")
                self.run_simulation(tracker, ground_truth, metric_manager,
                                            dir_name, ground_truth)
        except Exception as e:
            print(f'{datetime.now()} Preparing simulation error: {e}')
            logging.error(f'{datetime.now()} Could not run simulation. error: {e}')
            
    def prepare_and_run_multi_sim(self, config_path, combo_dict, groundtruth_setting, nruns):
        """Prepares multiple trackers for simulation runs

        Parameters
        ----------
        config_path : str
            path to configuration
        combo_dict : dict
            dictionary of all parameter combinations for monte-carlo
        groundtruth_setting : bool
            Defines if ground truth is present
        nruns : int
            Number of monte-carlo runs
        """
        tracker, ground_truth, metric_manager = self.set_components(config_path, groundtruth_setting)
        trackers, ground_truths, metric_managers = self.set_trackers(
            combo_dict, tracker, ground_truth, metric_manager)
        try:
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            for idx in range(0, len(trackers)):
                for runs_num in range(nruns):
                    dir_name = f"metrics_{dt_string}/simulation_{idx}/run_{runs_num}"
                    RunmanagerMetrics.parameters_to_csv(dir_name, combo_dict[idx])
                    RunmanagerMetrics.generate_config(
                        dir_name, trackers[idx], ground_truths[idx], metric_managers[idx])
                    print("RUN")
                    self.run_simulation(trackers[idx], ground_truths[idx], metric_managers[idx],
                                        dir_name)
        except Exception as e:
            print(f'{datetime.now()} Preparing simulation error: {e}')
            logging.error(f'{datetime.now()} Could not run simulation. error: {e}')


    def set_components(self, config_path, groundtruth_setting):
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
        tracker, ground_truth, metric_manager= None, None, None
        try:
            with open(config_path, 'r') as file:
                tracker, ground_truth, metric_manager = self.read_config_file(file, groundtruth_setting)
        except Exception as e:
            print(f'{datetime.now()} Could not read config file: {e}')
            logging.error(f'{datetime.now()} Could not read config file: {e}')
        return tracker, ground_truth, metric_manager