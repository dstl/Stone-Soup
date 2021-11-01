from base import RunManager
import os
import csv
from itertools import chain
import json
from stonesoup.types.array import CovarianceMatrix, StateVector
from datetime import timedelta
from stonesoup.serialise import YAML

class RunmanagerMetrics(RunManager):
    """Class for generating

    Args:
        Runmanager : Run manager base class
    """
    def tracks_to_csv(dir_name, tracks, overwrite=False):
        """Create a csv file for the track. It will contain the following columns:
            time    id  state   mean    covar

        Args:
            dir_name: name of the directory where to create the config file
            tracks: tracks data
            overwrite: overwrite the file.

        """
        if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        if not os.path.isfile(os.path.join(dir_name, 'tracks.csv')) or overwrite:
            with open(os.path.join(dir_name, 'tracks.csv'), 'w',newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time', 'id', 'state', 'mean', 'covar'])
                csvfile.close()

        with open(os.path.join(dir_name, 'tracks.csv'), 'a',newline='') as csvfile:
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

        filename = "metrics.csv"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if not os.path.isfile(os.path.join(dir_name, filename)) or overwrite:
            with open(os.path.join(dir_name, filename), 'w',newline='') as csvfile:
                writer = csv.writer(csvfile)

                for metric in metrics:
                    for metric_line in metric.value:
                        title = []
                        for property in type(metric_line).properties:
                            title.append(property)
                        writer.writerow(title)
                        break
                csvfile.close()

        with open(os.path.join(dir_name, filename), 'a',newline='') as csvfile:
            writer = csv.writer(csvfile)
            for metric in metrics:
                for metric_line in metric.value:
                    row = []
                    for property in type(metric_line).properties:
                        row.append(getattr(metric_line,property))
                    writer.writerow(row)



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
            with open(os.path.join(dir_name, filename), 'w',newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time', 'x', 'y'])
                csvfile.close()

        with open(os.path.join(dir_name, filename), 'a',newline='') as csvfile:
            writer = csv.writer(csvfile)
            for d_set in detections:
                if d_set:
                    writer.writerow([d_set.timestamp, str(d_set.state_vector[0]), str(d_set.state_vector[1])])


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
            with open(os.path.join(dir_name, filename), 'w',newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time', 'state'])
                csvfile.close()


        with open(os.path.join(dir_name, filename), 'a',newline='') as csvfile:
            writer = csv.writer(csvfile)
            for gt in groundtruths:
                writer.writerow([gt.state.timestamp,
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

        for k, v in parameters.items():
            if isinstance(v, StateVector) or isinstance(v, CovarianceMatrix):
                parameters[k] = list(v)
            elif isinstance(v, timedelta):
                #may change this in the future, unsure on the datatype for saving to json.
                parameters[k] = str(v)

        with open(os.path.join(dir_name, filename), 'a', newline='') as paramfile:
            json.dump(parameters, paramfile)

    def generate_config(dir_name, tracker, groundtruth, metrics, overwrite=False):
        data = [tracker, groundtruth, metrics]
        filename = "config.yaml"
        if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        f = open(os.path.join(dir_name, filename), "w")
        yaml = YAML()
        yaml.dump(data, f)
        f.close()
