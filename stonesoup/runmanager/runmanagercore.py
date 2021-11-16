import copy
import json
import logging
import sys
from datetime import datetime
import numpy as np
import multiprocessing as mp

from stonesoup.serialise import YAML
from inputmanager import InputManager
from runmanagermetrics import RunmanagerMetrics
from base import RunManager


class RunManagerCore(RunManager):

    def read_json(self, json_input):
        """ Reads JSON Files and stores in dictionary

        Args:
            json_input : json file filled with parameters
        """
        with open(json_input) as json_file:
            json_data = json.load(json_file)
            return json_data

    def run_multiprocess(self, args_list):
        pool = mp.Pool(mp.cpu_count())
        result = pool.starmap(self.run_simulation, args_list)
        return result

    def run(self, config_path, parameters_path, groundtruth_setting, output_path=None):
        """Run the run manager

        Args:
            config_path : Path of the config file
            parameters_path : Path of the parameters file
            groundtruth_setting : Checks if there is a ground truth available in the config file
        """
        logging.basicConfig(filename='simulation.log', encoding='utf-8', level=logging.INFO)
        input_manager = InputManager()
        json_data = self.read_json(parameters_path)
        trackers_combination_dict = input_manager.generate_parameters_combinations(
            json_data["parameters"])
        combo_dict = input_manager.generate_all_combos(trackers_combination_dict)
        
        try:
            with open(config_path, 'r') as file:
                tracker, ground_truth, metric_manager, csv_data = self.read_config_file(file)
        except Exception as e:
            print(e)
            logging.error(f'{datetime.now()} : {e}')

        
        if ground_truth is None:
            try:
                ground_truth = tracker.detector.groundtruth
            except Exception as e:
                logging.error(f'{datetime.now()} : {e}')
                print(f'No groundtruth in tracker detector {e}', flush=True)

        trackers = []
        ground_truths = []
        metric_managers = []

        trackers, ground_truths, metric_managers = self.set_trackers(
            combo_dict, tracker, ground_truth, metric_manager)
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        multiprocess = True
        if multiprocess:
            dir_name = f"metrics_{dt_string}/simulation_/run_"
            all_args = [(trackers[idx], ground_truths[idx], metric_managers[idx],
                         dir_name, groundtruth_setting, idx, combo_dict) for idx in range(0, len(trackers))]
            print("RUN")
            self.run_multiprocess(all_args)
        else:
            for idx in range(0, len(trackers)):
                for runs_num in range(0, json_data["configuration"]["runs_num"]):
                    dir_name = f"metrics_{dt_string}/simulation_{idx}/run_{runs_num}"
                    RunmanagerMetrics.parameters_to_csv(dir_name, combo_dict[idx])
                    RunmanagerMetrics.generate_config(
                        dir_name, trackers[idx], ground_truths[idx], metric_managers[idx])
                    if groundtruth_setting == 0:
                        groundtruth = trackers[idx].detector.groundtruth
                    else:
                        groundtruth = ground_truths[idx]
                    print("RUN")
                    self.run_simulation(trackers[idx], groundtruth, metric_managers[idx],
                                        dir_name, groundtruth_setting, idx, combo_dict)
        # Final line of the log show total time taken to run.
        logging.info(f'All simulations completed. Time taken to run: {datetime.now() - now}')
        print(f'All simulations completed. Time taken to run: {datetime.now() - now}')

    def run_simulation(self, tracker, ground_truth,
                       metric_manager, dir_name,
                       groundtruth_setting, index, combos):
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
            logging.error(f'{log_time}: Simulation {index} failed in {datetime.now() - log_time}.'
                          f'error: {e}  . Parameters: {combos[index]}')
            print(f'Failed to run Simulation: {e}', flush=True)

        else:
            logging.info(f'{log_time}: Simulation {index} / {len(combos)-1} ran '
                         'successfully in {datetime.now() - log_time}.'
                         'With Parameters: {combos[index]}')
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
                    # path_param = '.'.join(split_path[1::])
                    split_path = split_path[1::]

                # setattr(tracker_copy.initiator, split_path[-1], v)
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
            # print(split_path[0])
            newEl = getattr(el, split_path[0])
            self.set_param(split_path[1::], newEl, value)
        else:
            # print(value)
            # print(getattr(el,split_path[0]))
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
        
        if not hasattr(config_data, "__len__"):
            config_data = [config_data]
        
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


if __name__ == "__main__":
    args = sys.argv[1:]

    try:
        configInput = args[0]

    except Exception as e:
        # configInput = "C:\\Users\\Davidb1\\Documents\\Python\\data\\testConfigs\\\
        #                testConfigs\\metrics_config_v5.yaml"
        # configInput = "C:\\Users\\gbellant.LIVAD\\Documents\\Projects\\serapis\\\
        #     Serapis C38 LOT 1\\config.yaml"
        configInput = "C:\\Users\\gbellant\\Downloads\\2021_Nov_16_11_23_13_039262.yaml"
        logging.error(e)

    try:
        parametersInput = args[1]
    except Exception as e:
        # parametersInput = "C:\\Users\\gbellant.LIVAD\\Documents\\Projects\\serapis\\\
        #     Serapis C38 LOT 1\\parameters.json"
        logging.error(e)
        # parametersInput= "C:\\Users\\gbellant\\Documents\\Projects\\Serapis\\dummy3.json"
        parametersInput = "C:\\Users\\gbellant\\Downloads\\2021_Nov_16_11_23_13_039262_parameters.json"

    try:
        groundtruthSettings = args[2]
    except Exception as e:
        groundtruthSettings = 1
        logging.error(e)

    rmc = RunManagerCore()

    rmc.run(configInput, parametersInput, groundtruthSettings)
