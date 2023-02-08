import csv
import glob
import json
import os
from datetime import datetime, timedelta
from itertools import chain

import pandas as pd

from ..serialise import YAML
from ..types.array import CovarianceMatrix, StateVector


class RunManagerMetrics:
    """Class for generating metrics and storing simulation results and output data
    into csv files.

    """
    @staticmethod
    def tracks_to_csv(dir_name, tracks, overwrite=False):
        """Create a csv file for the tracks. It will contain the following columns:
            time  |  id  |  state  |  mean  |  covar

        Parameters
        ----------
        dir_name : str
            name of the directory where to create the config file
        tracks : Track
            Stonesoup track data
        overwrite : bool, optional
            Overwrite the file. Default is False
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
            print(f'{datetime.now()}: Failed to write to {dir_name}, {e}')

    def metrics_to_csv(self, dir_name, metrics, overwrite=False):
        """Create a CSV file for the metrics. It will contain the following columns:
            title  |  value  |  generator  |  timestamp

        Parameters
        ----------
        dir_name : str
            name of the directory where to create the config file
        metrics : Metric
            Metrics object
        overwrite : bool, optional
            overwrite the file. Default is False
        """
        filename = "metrics.csv"
        try:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            metricDictionary = self.create_metric_dict(metrics)
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
            print(f'{datetime.now()}: Failed to write to {filename}, {e}')

    @staticmethod
    def create_metric_dict(metrics):
        """Creates a dictionary of the metric values

        Parameters
        ----------
        metrics : MetricManager
            stonesoup metric object

        Returns
        -------
        dict
            dictionary version of metric manager to be used for printing CSV files
        """
        metricDictionary = {}
        for metric in metrics:
            if isinstance(metric.value, list):
                metricDictionary[metric.title] = []
                metricDictionary["timestamp"] = []
                for metric_line in metric.value:
                    metricDictionary[metric.title].append(metric_line.value)
                    metricDictionary["timestamp"].append(metric_line.timestamp)
        return metricDictionary

    @staticmethod
    def detection_to_csv(dir_name, detections, overwrite=False):
        """Create a CSV file for the detections. It will contain the following columns:
            time  |  x  |  y

        Parameters
        ----------
        dir_name : str
            name of the directory where to create the config file
        detections : Detections
            Detections Stonesoup values
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
            print(f'{datetime.now()}: Failed to write to {filename}, {e}')

    @staticmethod
    def groundtruth_to_csv(dir_name, groundtruths, overwrite=False):
        """Create a CSV file for the groundtruth. It will contain the following columns:
            time  |  id  |  state

        Parameters
        ----------
        dir_name : str
            name of the directory where to create the config file
        groundtruths : GroundTruth
            GroundTruth Stonesoup values
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
                    try:
                        state = gt.state
                    except Exception as e:
                        state = gt
                        print(f"gt.state not found: {e}")
                    try:
                        id_ = gt.id
                    except Exception as e:
                        id_ = ""
                        print(f"gt.id not found: {e}")
                    writer.writerow([state.timestamp, id_,
                                    ' '.join([str(n) for n in state.state_vector])])

        except Exception as e:
            print(f'{datetime.now()}: Failed to write to {filename}, {e}')

    def parameters_to_csv(self, dir_name, parameters):
        """Create a CSV file for the parameters. It will contain the parameter name for each
        simulation.

        Parameters
        ----------
        dir_name : Union[str, Path]
            name of the directory where to create the config file
        parameters : dict
            Dictionary of parameter details for the simulation run
        """
        filename = "parameters.json"
        try:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            parameters = self.write_params(parameters)
            with open(os.path.join(dir_name, filename), 'a', newline='') as paramfile:
                json.dump(parameters, paramfile)
        except Exception as e:
            print(f'{datetime.now()}: Failed to write to {filename}, {e}')

    @staticmethod
    def generate_config(dir_name, tracker=None,
                        groundtruth=None, metrics=None):
        """Creates a config.yaml file using the parameters you specified in the model.

        Parameters
        ----------
        dir_name : Union[str, Path]
            name of the directory where to create the config file
        tracker : Tracker, optional
            Stonesoup tracker object, by default None
        groundtruth : GroundTruth, optional
            Stonesoup tracker object, by default None
        metrics : Metrics, optional
            Stone soup metrics object, by default None
        """

        data = [tracker, groundtruth, metrics]
        filename = "config.yaml"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        f = open(os.path.join(dir_name, filename), "w")
        yaml = YAML()
        yaml.dump(data, f)
        f.close()

    @staticmethod
    def write_params(parameters):
        """Finds the correct parameters to print out to file for quick access
        to which parameters have been changed in the config.

        Parameters
        ----------
        parameters : dict
            list of parameters used in the configuration

        Returns
        -------
        dict
            dictionary of parameters in JSON format
        """
        try:
            for k, v in parameters.items():
                if isinstance(v, StateVector) or isinstance(v, CovarianceMatrix):
                    parameters[k] = list(v)
                    if type(parameters[k][0]) is CovarianceMatrix:
                        for i in range(len(parameters[k])):
                            parameters[k][i] = list(parameters[k][i])
                elif isinstance(v, timedelta):
                    parameters[k] = str(v)
        except Exception as e:
            print(f'{datetime.now()}: failed to write parameters correctly. {e}')
        return parameters

    @staticmethod
    def create_summary_csv(dir_name, run_info):
        filename = "tracking_log.csv"
        try:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            if not os.path.isfile(os.path.join(dir_name, filename)):
                with open(os.path.join(dir_name, filename), 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, run_info.keys())
                    writer.writeheader()
                    csvfile.close()

            with open(os.path.join(dir_name, filename), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, run_info.keys())
                writer.writerow(run_info)

        except Exception as e:
            print(e)

    @staticmethod
    def average_simulations(dataframes, length_files):
        """Takes list of dataframes and averages them on cell level

        Parameters
        ----------
        dataframes : Sequence[DataFrame]
            A set of dataframes to be averaged
        length_files : int
            Length of dataframe set

        Returns
        -------
        DataFrame
            Returns pandas DataFrame
        """
        timestamp = dataframes.iloc[:, -1]
        df = dataframes.iloc[:, :-1].div(length_files)
        df["timestamp"] = timestamp
        return df

    def sum_simulations(self, directory, chunk_size: int):
        """Sums metrics.csv files and processes them in batches to reserve memory space.

        Parameters
        ----------
        directory : str
            directory path where metrics.csv is located
        chunk_size : int
            size of batches

        Returns
        -------
        DataFrame
            Returns the sum of dataframes loaded from CSV files.
        """
        all_files = glob.glob(f'./{directory}*/run*[!!]/metrics.csv', recursive=True)
        batch = self.batch_list(all_files, chunk_size)
        summed_dataframe = pd.DataFrame()
        for files in batch:
            dfs = [pd.read_csv(file, infer_datetime_format=True) for file in files]
            averagedf = pd.concat(dfs, ignore_index=False).groupby(level=0).sum()
            averagedf["timestamp"] = dfs[0].iloc[:, -1]
            if not summed_dataframe.empty:
                summed_dataframe = summed_dataframe + averagedf
            if summed_dataframe.empty:
                summed_dataframe = averagedf

        return summed_dataframe, len(all_files)

    @staticmethod
    def batch_list(lst, n):
        """ Splits list into batches/chunks to be used when memory issues arise with
        averaging large datasets.

        Parameters
        ----------
        lst : list
            List object
        n : int
            number of items per batch

        Returns
        -------
        List
            Returns a list containing n number of batches
        """
        if n == 0:
            n = len(lst)
        chunked_lists = []
        for i in range(0, len(lst), n):
            chunked_lists.append(lst[i:i + n])
        return chunked_lists
