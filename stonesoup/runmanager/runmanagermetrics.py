import numpy as np
import os
import csv
from itertools import chain
import json
from stonesoup.types.array import CovarianceMatrix, StateVector
from datetime import timedelta
from stonesoup.serialise import YAML
from .base import RunManager


class RunmanagerMetrics(RunManager):
    """Class for generating

    Args:
        Runmanager : Run manager base class
    """
    def tracks_to_csv(dir_name, tracks, overwrite=False):
        """Create a csv file for the track. It will contain the following columns:
            time    id    state    mean    covar

        Args:
            dir_name: name of the directory where to create the config file
            tracks: tracks data
            overwrite: overwrite the file.

        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if not os.path.isfile(os.path.join(dir_name, 'tracks.csv')) or overwrite:
            with open(os.path.join(dir_name, 'tracks.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time', 'id', 'state', 'mean', 'covar'])
                csvfile.close()

        with open(os.path.join(dir_name, 'tracks.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for t in tracks:
                # Export the track state as a single space-delimited string
                # The visualisation GUI will automatically expand this data when loading
                c = ' '.join([str(i) for i in list(chain.from_iterable(zip(*t.covar)))])
                writer.writerow([t.state.timestamp, t.id,
                                 ' '.join([str(n) for n in t.state.state_vector]),
                                 ' '.join([str(n) for n in t.state.mean]),
                                 c])

    def metrics_to_csv(dir_name, metrics, overwrite=False):
        """Create a csv file for the metrics. It will contain the following columns:
            title    value    generator    timestamp

        Args:
            dir_name: name of the directory where to create the config file
            tracks: tracks data
            overwrite: overwrite the file.

        """
        filename = "metrics.csv"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        metricDictionary = {}
        for metric in metrics:
            if(isinstance(metric.value, list)):
                metricDictionary[metric.title] = []
                metricDictionary["timestamp"] = []
                for metric_line in metric.value:
                    metricDictionary[metric.title].append(metric_line.value)
                    metricDictionary["timestamp"].append(metric_line.timestamp)

        keys = sorted(metricDictionary.keys())
        if not os.path.isfile(os.path.join(dir_name, filename)) or overwrite:
            with open(os.path.join(dir_name, filename), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(keys)
                csvfile.close()

        with open(os.path.join(dir_name, filename), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(zip(*[metricDictionary[key] for key in keys]))

    def detection_to_csv(dir_name, detections, overwrite=False):
        """Create a csv file for the detections. It will contain the following columns:
            time    x  y

        Args:
            dir_name: name of the directory where to create the config file
            detections: detections data
            overwrite: overwrite the file.
        """
        filename = "detections.csv"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if not os.path.isfile(os.path.join(dir_name, filename)) or overwrite:
            with open(os.path.join(dir_name, filename), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time', 'x', 'y'])
                csvfile.close()

        with open(os.path.join(dir_name, filename), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for d_set in detections:
                if d_set:
                    writer.writerow([d_set.timestamp,
                                    str(d_set.state_vector[0]),
                                    str(d_set.state_vector[1])])

    def groundtruth_to_csv(dir_name, groundtruths, overwrite=False):
        """Create a csv file for the grountruth. It will contain the following columns:

        Args:
            dir_name: name of the directory where to create the config file
            groundtruths: groundtruths data
            overwrite: overwrite the file.
        """
        filename = "groundtruth.csv"

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if not os.path.isfile(os.path.join(dir_name, filename)) or overwrite:
            with open(os.path.join(dir_name, filename), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time', 'id','state'])
                csvfile.close()

        with open(os.path.join(dir_name, filename), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for gt in groundtruths:
                writer.writerow([gt.state.timestamp, gt.id,
                                ' '.join([str(n) for n in gt.state.state_vector])])

    def parameters_to_csv(dir_name, parameters, overwrite=False):
        """Create a csv file for the parameters. It will contain the parameter name for each simulation.

        Args:
            dir_name: name of the directory where to create the config file
            parameters: dictionary of the parameter details for the simulation runs.
            overwrite: overwrite the file.
        """
        filename = "parameters.json"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        try:
            for k, v in parameters.items():
                if isinstance(v, StateVector) or isinstance(v, CovarianceMatrix):              
                    parameters[k] = list(v)
                    if type(parameters[k][0]) is CovarianceMatrix:
                        for i in range(len(parameters[k])):
                            parameters[k][i] = list(parameters[k][i])
                elif isinstance(v, timedelta):
                    parameters[k] = str(v)

            with open(os.path.join(dir_name, filename), 'a', newline='') as paramfile:
                json.dump(parameters, paramfile)
        except Exception as e:
            print(e)

    def generate_config(dir_name, tracker=None, groundtruth=None, metrics=None, overwrite=False):
        """Creates a config.yaml file using the parameters you specificed in the model.

        Args:
            dir_name: name of the directory where to create the config file
            parameters: dictionary of the parameter details for the simulation runs.
            overwrite: overwrite the file.
        """
        data = [tracker, groundtruth, metrics]
        filename = "config.yaml"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        f = open(os.path.join(dir_name, filename), "w")
        yaml = YAML()
        yaml.dump(data, f)
        f.close()
