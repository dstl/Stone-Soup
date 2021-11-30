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
        
    def read_json(self, json_input):
        """ Reads JSON Files and stores in dictionary

        Args:
            json_input : json file filled with parameters
        """
        with open(json_input) as json_file:
            json_data = json.load(json_file)
            return json_data
            
    def run(self, config_path=None, parameters_path=None, 
            ground_truth=None, dir=None, nruns=1, nprocesses=None, output_path=None):
        """Run the run manager

        Args:
            config_path : Path of the config file
            parameters_path : Path of the parameters file
            groundtruth : Checks if there is a ground truth available in the config file
        """
        
        pairs = []
        trackers = [] 
        ground_truths = [] 
        metric_managers = []
        tracker = None 
        metric_manager = None
        
        input_manager = InputManager()
                
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
           self.prepare_and_run_single_sim(config_path, nruns)

        for path in pairs:
            param_path = path[0]
            config_path = path[1]
            json_data = self.read_json(param_path)
            trackers_combination_dict = input_manager.generate_parameters_combinations(
                json_data["parameters"])
            combo_dict = input_manager.generate_all_combos(trackers_combination_dict)
            tracker, ground_truth, metric_manager, csv_data = self.set_components(config_path)

            if ground_truth is None:
                try:
                    ground_truth = tracker.detector.groundtruth
                except Exception as e:
                    logging.error(f'{datetime.now()} : {e}')
                    print(f'No groundtruth in tracker detector {e}', flush=True)

            trackers, ground_truths, metric_managers = self.set_trackers(
                combo_dict, tracker, ground_truth, metric_manager)
            
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            for idx in range(0, len(trackers)):
                for runs_num in range(0, json_data["configuration"]["runs_num"]):
                    dir_name = f"metrics_{dt_string}/simulation_{idx}/run_{runs_num}"
                    RunmanagerMetrics.parameters_to_csv(dir_name, combo_dict[idx])
                    RunmanagerMetrics.generate_config(
                        dir_name, trackers[idx], ground_truths[idx], metric_managers[idx])
                    if ground_truth == None:
                        groundtruth = trackers[idx].detector.groundtruth
                    else:
                        groundtruth = ground_truths[idx]
                    print("RUN")
                    self.run_simulation(trackers[idx], groundtruth, metric_managers[idx],
                                        dir_name, ground_truth, idx, combo_dict)
            
            logging.info(f'All simulations completed. Time taken to run: {datetime.now() - now}')

    def run_simulation(self, tracker, ground_truth,
                       metric_manager, dir_name,
                       groundtruth_setting, index=None, combos=None):
        """Start the simulation

        Args:
            tracker: Tracker
            groundtruth: GroundTruth
            metric_manager: Metric Manager
            dir_name: Directory name for saving the simulations
            groundtruth_setting: unsued
            index: Keeps a track of which simulation is being ran
            combos: List of combinations for logging the parameters for each simulation.
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
            # logging.error(f'{log_time}: Simulation {index} failed in {datetime.now() - log_time}.'
            #               f'error: {e}  . Parameters: {combos[index]}')
            print(f'Failed to run Simulation: {e}', flush=True)

        else:
            # logging.info(f'{log_time}: Simulation {index} / {len(combos)-1} ran '
            #              'successfully in {datetime.now() - log_time}.'
            #              'With Parameters: {combos[index]}')
            print('Success!', flush=True)

    def set_trackers(self, combo_dict, tracker, ground_truth, metric_manager):
        """Set the trackers, groundtruths and metricmanagers list (stonesoup objects)

        Args:
            combo_dict (dict): dictionary of all the possible combinations of values

            tracker (tracker): stonesoup tracker

            groundtruth (groundtruth): stonesoup groundtruth

            metric_manager (metricmanager): stonesoup metric_manager

        Returns:
            list: list of trackers
            list: list of groundtruths
            list: list of metric managers
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
        """[summary]

        Args:
            split_path ([type]): [description]
            el ([type]): [description]
            value ([type]): [description]
        """

        if len(split_path) > 1:
            newEl = getattr(el, split_path[0])
            self.set_param(split_path[1::], newEl, value)
        else:
            if len(split_path) > 0:
                setattr(el, split_path[0], value)

    def read_config_file(self, config_file):
        """Read the configuration file

        Args:
            config_file (file path): file path of the configuration file

        Returns:
            trackers,ground_truth,metric_manager: trackers, ground_truth and
            metric manager stonesoup structure
        """
        config_string = config_file.read()

        tracker, gt, mm, csv_data = None, None, None, None

        config_data = YAML('safe').load(config_string)
        for x in config_data:
            if "Tracker" in str(type(x)):
                tracker = x
                print("Tracker found")
            elif "GroundTruth" in str(type(x)):
                gt = x
                print("Groundtruth found")
            elif "metricgenerator" in str(type(x)):
                mm = x
                print("Metric manager found")
            elif type(x) is np.ndarray:
                csv_data = x
                print("CSV data found")

        return tracker, gt, mm, csv_data

    def read_config_dir(self, config_dir):
        if os.path.exists(config_dir):
            files = os.listdir(config_dir)
        else:  
            return None
        return files
    
    def get_filepaths(self, directory):
        file_paths =[]
        if os.path.exists(directory):
            for root, directories, files in os.walk(directory):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    file_paths.append(filepath)
        return file_paths

    def get_config_and_param_lists(self, files):
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
    
    def prepare_and_run_single_sim(self, config_path, nruns):
        try:
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            tracker, ground_truth, metric_manager, csv_data = self.set_components(config_path)
            for runs in range(nruns):
                dir_name = f"metrics_{dt_string}/run_{runs}"
                print("RUN")
                self.run_simulation(tracker, ground_truth, metric_manager,
                                            dir_name, ground_truth)
        except Exception as e:
            print(f'{datetime.now()} Preparing simulation error: {e}')
            logging.error(f'{datetime.now()} Could not run simulation. error: {e}')

    def set_components(self, config_path):
        try:
            with open(config_path, 'r') as file:
                tracker, ground_truth, metric_manager, csv_data = self.read_config_file(file)
        except Exception as e:
            print(f'{datetime.now()} Could not read config file: {e}')
            logging.error(f'{datetime.now()} Could not read config file: {e}')
        return tracker, ground_truth, metric_manager, csv_data