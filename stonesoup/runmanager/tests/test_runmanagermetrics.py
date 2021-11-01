import pytest

import pandas as pd
import json
from datetime import datetime, timedelta
import typing
import os

from stonesoup.serialise import YAML
from stonesoup.types import metric, array
from stonesoup.tracker.simple import SingleTargetTracker
from ..runmanagermetrics import RunmanagerMetrics as rmm


test_dir_name = "tests\\test_csvs"

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


def test_tracks_to_csv():

    if os.path.exists(test_dir_name + "\\tracks.csv"):
        os.remove(test_dir_name + "\\tracks.csv")

    test_track1 = DummyTrack(DummyState(datetime.now(), [1], [1]), 1, [[1.0, 1.0]])
    test_track2 = DummyTrack(DummyState(datetime.now(), [2], [2]), 2, [[2.0, 2.0]])
    test_track3 = DummyTrack(DummyState(datetime.now(), [3], [3]), 3, [[3.0, 3.0]])

    test_tracks = [test_track1, test_track2, test_track3]

    assert test_tracks[0].state is not None
    assert test_tracks[0].id is not None
    assert test_tracks[0].covar is not None

    rmm.tracks_to_csv(test_dir_name, test_tracks)

    test_tracks_loaded = pd.read_csv("tests\\test_csvs\\tracks.csv", delimiter=",")

    for i in range(len(test_tracks)):
        assert test_tracks_loaded["time"][i] == str(test_tracks[i].state.timestamp)
        assert test_tracks_loaded["id"][i] == test_tracks[i].id
        assert test_tracks_loaded["state"][i] == test_tracks[i].state.state_vector
        assert test_tracks_loaded["mean"][i] == test_tracks[i].state.mean
        assert [[float(x) for x in test_tracks_loaded["covar"][i].split(" ")]] == test_tracks[i].covar

def test_metrics_to_csv():

    if os.path.exists(test_dir_name + "\\metrics.csv"):
        os.remove(test_dir_name + "\\metrics.csv")

    test_metric = metric.Metric("a", [metric.Metric("x", 1, "gen1.1"),
                                      metric.Metric("y", 2, "gen1.2")], "gen1")

    test_metrics = [test_metric]

    rmm.metrics_to_csv(test_dir_name, test_metrics)

    test_metrics_loaded = pd.read_csv("tests\\test_csvs\\metrics.csv", delimiter=",")

    for i in range(len(test_metric.value)):
        assert test_metrics_loaded["title"][i] == test_metric.value[i].title
        assert test_metrics_loaded["value"][i] == test_metric.value[i].value
        assert test_metrics_loaded["generator"][i] == test_metric.value[i].generator

def test_detection_to_csv():

    if os.path.exists(test_dir_name + "\\detections.csv"):
        os.remove(test_dir_name + "\\detections.csv")

    test_detection1 = DummyState(datetime.now(), [1, 2], None)
    test_detection2 = DummyState(datetime.now(), [3, 4], None)

    test_detections = [test_detection1, test_detection2]

    rmm.detection_to_csv(test_dir_name, test_detections)

    test_detections_loaded = pd.read_csv("tests\\test_csvs\\detections.csv", delimiter=",")

    for i in range(len(test_detections)):
        assert test_detections_loaded["time"][i] == str(test_detections[i].timestamp)
        assert test_detections_loaded["x"][i] == test_detections[i].state_vector[0]
        assert test_detections_loaded["y"][i] == test_detections[i].state_vector[1]

def test_groundtruth_to_csv():

    if os.path.exists(test_dir_name + "\\groundtruth.csv"):
        os.remove(test_dir_name + "\\groundtruth.csv")

    test_gt1 = DummyTrack(DummyState(datetime.now(), [1], None), None, None)
    test_gt2 = DummyTrack(DummyState(datetime.now(), [2], None), None, None)
    test_gt3 = DummyTrack(DummyState(datetime.now(), [3], None), None, None)

    test_groundtruths = [test_gt1, test_gt2, test_gt3]

    rmm.groundtruth_to_csv(test_dir_name, test_groundtruths)

    test_groundtruths_loaded = pd.read_csv("tests\\test_csvs\\groundtruth.csv", delimiter=",")

    for i in range(len(test_groundtruths)):
        assert test_groundtruths_loaded["time"][i] == str(test_groundtruths[i].state.timestamp)
        assert test_groundtruths_loaded["state"][i] == test_groundtruths[i].state.state_vector

def test_parameters_to_csv():
    # Fails because Object of type int32 is not JSON serializable
    # - Apparently this exception is raised when it is of numpy int type
    if os.path.exists(test_dir_name + "\\parameters.json"):
        os.remove(test_dir_name + "\\parameters.json")

    test_parameters = {"a": array.StateVector([1, 2, 3, 4]),
                       "b": array.CovarianceMatrix([[5, 6], [7, 8]]),
                       "c": timedelta(1)}

    rmm.parameters_to_csv(test_dir_name, test_parameters)

    with open(test_dir_name + "\\parameters.json") as json_file:
        test_parameters_loaded = json.load(json_file)

    assert test_parameters_loaded["a"] == list(test_parameters["a"])
    assert test_parameters_loaded["b"] == list(test_parameters["b"])
    assert test_parameters_loaded["c"] == list(test_parameters["c"])

def test_generate_config():
    test_tracker = {"tracker": 246}
    test_gt = {"groundtruth": 567}
    test_metrics = {"metrics": 987}

    rmm.generate_config(test_dir_name, test_tracker, test_gt, test_metrics)

    with open(test_dir_name+"\\config.yaml", 'r') as file:
        tracker, gt, mm = YAML('safe').load(file)

    assert tracker == test_tracker
    assert gt == test_gt
    assert mm == test_metrics

    file.close()
