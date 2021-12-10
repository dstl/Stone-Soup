
import csv
import json
from datetime import datetime, timedelta
import time
import os

from stonesoup.serialise import YAML
from stonesoup.types import metric, array
from ..runmanagermetrics import RunmanagerMetrics as runmanager_metrics


# Run from stonesoup working directory
# def setup_module():
#     while os.getcwd().split('\\')[-1] != 'Stone-Soup':
#         os.chdir(os.path.dirname(os.getcwd()))

# test_dir_name = "stonesoup\\runmanager\\tests\\test_csvs"
test_dir_name = "stonesoup/runmanager/tests/test_csvs"
runmanager_metrics = runmanager_metrics()

class DummyTrack:
    def __init__(self, state, id, covar):
        self.state = state
        self.id = id
        self.covar = covar


class DummyState:
    def __init__(self, timestamp, state_vector, mean):
        self.timestamp = timestamp
        self.state_vector = state_vector
        self.mean = mean


def test_cwd_path():
    assert os.path.isdir(test_dir_name) is True


def test_tracks_to_csv():

    if os.path.exists(test_dir_name + "/tracks.csv"):
        os.remove(test_dir_name + "/tracks.csv")

    test_track1 = DummyTrack(DummyState(datetime.now(), [1], [1]), 1, [[1.0, 1.0]])
    test_track2 = DummyTrack(DummyState(datetime.now(), [2], [2]), 2, [[2.0, 2.0]])
    test_track3 = DummyTrack(DummyState(datetime.now(), [3], [3]), 3, [[3.0, 3.0]])

    test_tracks = [test_track1, test_track2, test_track3]

    assert test_tracks[0].state is not None
    assert test_tracks[0].id is not None
    assert test_tracks[0].covar is not None

    runmanager_metrics.tracks_to_csv(test_dir_name, test_tracks)

    with open(test_dir_name+"/tracks.csv") as csvfile:
        test_tracks_loaded = csv.DictReader(csvfile, delimiter=",")
        i = 0
        for row in test_tracks_loaded:
            assert row["time"] == str(test_tracks[i].state.timestamp)
            assert int(row["id"]) == test_tracks[i].id
            assert [int(row["state"])] == test_tracks[i].state.state_vector
            assert [int(row["mean"])] == test_tracks[i].state.mean
            assert [[float(x) for x in row["covar"].split(" ")]] == test_tracks[i].covar
            i += 1


def test_metrics_to_csv():

    if os.path.exists(test_dir_name + "/metrics.csv"):
        os.remove(test_dir_name + "/metrics.csv")

    date_time = datetime.now()
    time.sleep(1)
    date_time2 = datetime.now()

    test_metric = metric.Metric("a",
                                [metric.SingleTimeMetric("x", 1, "gen1.1", date_time),
                                 metric.SingleTimeMetric("y", 2, "gen1.2", date_time2)],
                                "gen1")

    test_metric2 = metric.Metric("b",
                                 [metric.SingleTimeMetric("m", 3, "gen2.1", date_time),
                                  metric.SingleTimeMetric("n", 4, "gen2.2", date_time2)],
                                 "gen2")

    test_metrics = [test_metric, test_metric2]

    runmanager_metrics.metrics_to_csv(test_dir_name, test_metrics)

    with open(test_dir_name+"/metrics.csv") as csvfile:
        test_metrics_loaded = csv.DictReader(csvfile, delimiter=",")
        i = 0
        for row in test_metrics_loaded:
            assert int(row["a"]) == test_metric.value[i].value
            assert row["timestamp"] == str(test_metric.value[i].timestamp)
            assert int(row["b"]) == test_metric2.value[i].value
            assert row["timestamp"] == str(test_metric2.value[i].timestamp)
            i += 1


def test_detection_to_csv():

    if os.path.exists(test_dir_name + "/detections.csv"):
        os.remove(test_dir_name + "/detections.csv")

    test_detection1 = DummyState(datetime.now(), [1, 2], None)
    test_detection2 = DummyState(datetime.now(), [3, 4], None)

    test_detections = [test_detection1, test_detection2]

    runmanager_metrics.detection_to_csv(test_dir_name, test_detections)

    with open(test_dir_name+"/detections.csv") as csvfile:
        test_detections_loaded = csv.DictReader(csvfile, delimiter=",")
        i = 0
        for row in test_detections_loaded:
            assert row["time"] == str(test_detections[i].timestamp)
            assert int(row["x"]) == test_detections[i].state_vector[0]
            assert int(row["y"]) == test_detections[i].state_vector[1]
            i += 1


def test_groundtruth_to_csv():

    if os.path.exists(test_dir_name + "/groundtruth.csv"):
        os.remove(test_dir_name + "/groundtruth.csv")

    test_gt1 = DummyTrack(DummyState(datetime.now(), [1], None), None, None)
    test_gt2 = DummyTrack(DummyState(datetime.now(), [2], None), None, None)
    test_gt3 = DummyTrack(DummyState(datetime.now(), [3], None), None, None)

    test_groundtruths = [test_gt1, test_gt2, test_gt3]

    runmanager_metrics.groundtruth_to_csv(test_dir_name, test_groundtruths)

    with open(test_dir_name+"/groundtruth.csv") as csvfile:
        test_groundtruths_loaded = csv.DictReader(csvfile, delimiter=",")
        i = 0
        for row in test_groundtruths_loaded:
            assert row["time"] == str(test_groundtruths[i].state.timestamp)
            assert [int(row["state"])] == test_groundtruths[i].state.state_vector
            i += 1


def test_parameters_to_csv():

    if os.path.exists(test_dir_name + "/parameters.json"):
        os.remove(test_dir_name + "/parameters.json")

    test_parameters = {"a": array.StateVector([1.0, 2.0, 3.0, 4.0]),
                       "b": array.CovarianceMatrix([[5.0, 6.0], [7.0, 8.0]]),
                       "c": timedelta(1),
                       "d": 1,
                       "e": "string",
                       "f": None}

    runmanager_metrics.parameters_to_csv(test_dir_name, test_parameters)

    with open(test_dir_name + "/parameters.json") as json_file:
        test_parameters_loaded = json.load(json_file)

    assert test_parameters_loaded["a"] == list(test_parameters["a"])
    assert test_parameters_loaded["b"] == list(test_parameters["b"])
    assert test_parameters_loaded["c"] == str(test_parameters["c"])
    assert test_parameters_loaded["d"] == test_parameters["d"]
    assert test_parameters_loaded["e"] == test_parameters["e"]
    assert test_parameters_loaded["f"] == test_parameters["f"]


def test_generate_config():
    test_tracker = {"tracker": 246}
    test_gt = {"groundtruth": 567}
    test_metrics = {"metrics": 987}

    runmanager_metrics.generate_config(test_dir_name, test_tracker, test_gt, test_metrics)

    with open(test_dir_name+"/config.yaml", 'r') as file:
        tracker, gt, mm = YAML('safe').load(file)

    assert tracker == test_tracker
    assert gt == test_gt
    assert mm == test_metrics

    file.close()
