
import os
import csv
from itertools import chain
import json
from stonesoup.types.array import CovarianceMatrix, StateVector
from datetime import timedelta
from stonesoup.serialise import YAML
from .base import RunManager
import datetime

class RunmanagerMetrics(RunManager):
    """Class for generating

    Parameters
    ----------
    Runmanager : Class
        Run manager base class
    """
    def tracks_to_csv(dir_name, tracks, overwrite=False):
        """Create a csv file for the track. It will contain the following columns:
            time    id    state    mean    covar

        Parameters
        ----------
        dir_name : str
            name of the directory where to create the config file
        tracks : Track
            Stonesoup track data
        overwrite : bool, optional
            overwrite the file, by default False
        """
        try:
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
        except Exception as e:
            print(f'{datetime.now()}: Failed to write to {dir_name}')
    def metrics_to_csv(dir_name, metrics, overwrite=False):
        """Create a csv file for the metrics. It will contain the following columns:
            title    value    generator    timestamp

        Parameters
        ----------
        dir_name : [type]
            name of the directory where to create the config file
        metrics : Metric
            Metrics object
        overwrite : bool, optional
            overwrite the file, by default False
        """
        filename = "metrics.csv"
        try:
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
        except Exception as e:
            print(f'{datetime.now()}: Failed to write to {filename}')

    def detection_to_csv(dir_name, detections, overwrite=False):
        """Create a csv file for the detections. It will contain the following columns:
            time    x  y

        Parameters
        ----------
        dir_name : str
            name of the directory where to create the config file
        detections : Detections
            Detections Stonesoup
        overwrite : bool, optional
            overwrite the file., by default False
        """
        filename = "detections.csv"
        try:
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
        except Exception as e:
            print(f'{datetime.now()}: Failed to write to {filename}')
            
            
    def groundtruth_to_csv(dir_name, groundtruths, overwrite=False):
        """Create a csv file for the grountruth.

        Parameters
        ----------
        dir_name : str
            name of the directory where to create the config file
        groundtruths : GrouthTruth
            GrouthTruth Stonesoup
        overwrite : bool, optional
            overwrite the file., by default False
        """
        filename = "groundtruth.csv"
        try:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            if not os.path.isfile(os.path.join(dir_name, filename)) or overwrite:
                with open(os.path.join(dir_name, filename), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['time', 'id', 'state'])
                    csvfile.close()

            with open(os.path.join(dir_name, filename), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for gt in groundtruths:
                    writer.writerow([gt.state.timestamp, gt.id,
                                    ' '.join([str(n) for n in gt.state.state_vector])])
        except Exception as e:
            print(f'{datetime.now()}: Failed to write to {filename}')

    def parameters_to_csv(dir_name, parameters, overwrite=False):
        """Create a csv file for the parameters. It will contain the parameter name for each simulation.

        Parameters
        ----------
        dir_name : str
            name of the directory where to create the config file
        parameters : dict
            Dictionary of paramater details for the simulation run
        overwrite : bool, optional
            overwrite the file, by default False
        """
        filename = "parameters.json"
        
        try:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

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
            print(f'{datetime.now()}: Failed to write to {filename}')

    def generate_config(dir_name, tracker=None, groundtruth=None, metrics=None, overwrite=False):
        """Creates a config.yaml file using the parameters you specificed in the model.

        Parameters
        ----------
        dir_name : str
            name of the directory where to create the config file
        tracker : Tracker, optional
            Stonesoup tracker object, by default None
        groundtruth : GrouthTruth, optional
            Stonesoup tracker object, by default None
        metrics : Metrics, optional
            Stone soup metrics object, by default None
        overwrite : bool, optional
            overwrite the file, by default False
        """
        try:
            data = [tracker, groundtruth, metrics]
            filename = "config.yaml"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            f = open(os.path.join(dir_name, filename), "w")
            yaml = YAML()
            yaml.dump(data, f)
            f.close()
        except Exception as e:
            print(f'{datetime.now()}: Failed to write to {filename}')