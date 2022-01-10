# -*- coding: utf-8 -*-

import numpy as np

from ..tracktotruthmetrics import SIAPMetrics, IDSIAPMetrics
from ...measures import Euclidean
from ...types.groundtruth import GroundTruthPath
from ...types.metric import SingleTimeMetric, TimeRangeMetric
from ...types.track import Track


def test_siap(trial_manager, trial_truths, trial_tracks, trial_associations):
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(position_measure=position_measure,
                                 velocity_measure=velocity_measure)

    trial_manager.generators = [siap_generator]

    timestamps = trial_manager.list_timestamps()

    # Test num_tracks_at_time
    for timestamp in timestamps:
        assert siap_generator.num_tracks_at_time(trial_manager, timestamp) == 3

    # Test num_associated_tracks_at_time
    assert siap_generator.num_associated_tracks_at_time(trial_manager, timestamps[0]) == 2
    assert siap_generator.num_associated_tracks_at_time(trial_manager, timestamps[1]) == 3
    assert siap_generator.num_associated_tracks_at_time(trial_manager, timestamps[2]) == 3
    assert siap_generator.num_associated_tracks_at_time(trial_manager, timestamps[3]) == 2

    # Test accuracy_at_time
    assoc0_pos_accuracy = np.sqrt(0.1 ** 2 + 0.1 ** 2)
    assoc1_pos_accuracy = np.sqrt(0.5 ** 2 + 0.5 ** 2)
    assoc0_vel_accuracy = np.sqrt(0.2 ** 2 + 0.2 ** 2)
    assoc1_vel_accuracy = np.sqrt(0.6 ** 2 + 0.6 ** 2)
    exp_pos_accuracy = assoc0_pos_accuracy + assoc1_pos_accuracy
    exp_vel_accuracy = assoc0_vel_accuracy + assoc1_vel_accuracy

    pos_accuracy = siap_generator.accuracy_at_time(trial_manager, timestamps[0], position_measure)
    assert pos_accuracy == exp_pos_accuracy
    vel_accuracy = siap_generator.accuracy_at_time(trial_manager, timestamps[0], velocity_measure)
    assert vel_accuracy == exp_vel_accuracy

    # Test truth_track_from_association
    for association in trial_associations:
        truth, track = siap_generator.truth_track_from_association(association)
        assert isinstance(truth, GroundTruthPath)
        assert isinstance(track, Track)

    # Test total_time_tracked
    assert siap_generator.total_time_tracked(trial_manager, trial_truths[0]) == 3  # seconds
    assert siap_generator.total_time_tracked(trial_manager, trial_truths[1]) == 2
    assert siap_generator.total_time_tracked(trial_manager, trial_truths[2]) == 1
    assert siap_generator.total_time_tracked(trial_manager, GroundTruthPath()) == 0

    # Test min_num_tracks_needed_to_track
    assert siap_generator.min_num_tracks_needed_to_track(trial_manager, trial_truths[0]) == 2
    assert siap_generator.min_num_tracks_needed_to_track(trial_manager, trial_truths[1]) == 2
    assert siap_generator.min_num_tracks_needed_to_track(trial_manager, trial_truths[2]) == 1
    assert siap_generator.min_num_tracks_needed_to_track(trial_manager, GroundTruthPath()) == 0

    # Test rate_of_track_number_changes
    exp_rate = (2 - 1 + 2 - 1 + 1 - 1) / (3 + 2 + 1)
    assert siap_generator.rate_of_track_number_changes(trial_manager) == exp_rate

    # Test truth_lifetime
    for truth in trial_truths:
        assert siap_generator.truth_lifetime(truth) == 3

    # Test longest_track_time_on_truth
    assert siap_generator.longest_track_time_on_truth(trial_manager, trial_truths[0]) == 2
    assert siap_generator.longest_track_time_on_truth(trial_manager, trial_truths[1]) == 1
    assert siap_generator.longest_track_time_on_truth(trial_manager, trial_truths[2]) == 1

    # Test compute_metric
    metrics = siap_generator.compute_metric(trial_manager)
    expected_titles = ["SIAP Completeness", "SIAP Ambiguity", "SIAP Spuriousness",
                       "SIAP Position Accuracy", "SIAP Velocity Accuracy",
                       "SIAP Rate of Track Number Change", "SIAP Longest Track Segment",
                       "SIAP Completeness at times", "SIAP Ambiguity at times",
                       "SIAP Spuriousness at times", "SIAP Position Accuracy at times",
                       "SIAP Velocity Accuracy at times"]

    for expected_title in expected_titles:
        assert len({metric for metric in metrics if metric.title == expected_title}) == 1
    assert len({metric for metric in metrics if metric.title not in expected_titles}) == 0

    for metric in metrics:
        assert isinstance(metric, TimeRangeMetric)
        assert metric.time_range.start_timestamp == timestamps[0]
        assert metric.time_range.end_timestamp == timestamps[3]
        assert metric.generator == siap_generator

        if metric.title.endswith(" at times"):
            assert isinstance(metric.value, list)
            assert len(metric.value) == 4  # number of timestamps

            for thing in metric.value:
                assert isinstance(thing, SingleTimeMetric)
                assert isinstance(thing.value, (float, int))
                assert thing.generator == siap_generator
        else:
            assert isinstance(metric.value, (float, int))


def test_id_siap(trial_manager, trial_truths, trial_tracks, trial_associations):
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    truth_id = track_id = "colour"
    siap_generator = IDSIAPMetrics(position_measure=position_measure,
                                   velocity_measure=velocity_measure,
                                   truth_id=truth_id,
                                   track_id=track_id)

    trial_manager.generators = [siap_generator]

    timestamps = trial_manager.list_timestamps()

    # Test find_track_id
    assert siap_generator.find_track_id(trial_tracks[0], timestamps[0]) == "red"
    assert siap_generator.find_track_id(trial_tracks[0], timestamps[1]) == "blue"
    assert siap_generator.find_track_id(trial_tracks[0], timestamps[2]) == "red"
    assert siap_generator.find_track_id(trial_tracks[0], timestamps[3]) == "red"

    assert siap_generator.find_track_id(trial_tracks[1], timestamps[0]) == "red"
    assert siap_generator.find_track_id(trial_tracks[1], timestamps[1]) == "red"
    assert siap_generator.find_track_id(trial_tracks[1], timestamps[2]) == "green"
    assert siap_generator.find_track_id(trial_tracks[1], timestamps[3]) == "green"

    assert siap_generator.find_track_id(trial_tracks[2], timestamps[0]) is None
    assert siap_generator.find_track_id(trial_tracks[2], timestamps[1]) is None
    assert siap_generator.find_track_id(trial_tracks[2], timestamps[2]) == "blue"
    assert siap_generator.find_track_id(trial_tracks[2], timestamps[3]) == "green"

    # Test num_id_truths_at_time
    u, c, i = siap_generator.num_id_truths_at_time(trial_manager, timestamps[0])
    assert u == 0
    assert c == 1
    assert i == 1

    u, c, i = siap_generator.num_id_truths_at_time(trial_manager, timestamps[1])
    assert u == 1
    assert c == 0
    assert i == 1

    u, c, i = siap_generator.num_id_truths_at_time(trial_manager, timestamps[2])
    assert u == 0
    assert c == 2
    assert i == 0

    u, c, i = siap_generator.num_id_truths_at_time(trial_manager, timestamps[3])
    assert u == 0
    assert c == 1
    assert i == 1

    # Test compute_metric
    metrics = siap_generator.compute_metric(trial_manager)
    expected_titles = ["SIAP Completeness", "SIAP Ambiguity", "SIAP Spuriousness",
                       "SIAP Position Accuracy", "SIAP Velocity Accuracy",
                       "SIAP Rate of Track Number Change", "SIAP Longest Track Segment",
                       "SIAP Completeness at times", "SIAP Ambiguity at times",
                       "SIAP Spuriousness at times", "SIAP Position Accuracy at times",
                       "SIAP Velocity Accuracy at times",
                       "SIAP ID Completeness", "SIAP ID Correctness", "SIAP ID Ambiguity",
                       "SIAP ID Completeness at times", "SIAP ID Correctness at times",
                       "SIAP ID Ambiguity at times"]

    for expected_title in expected_titles:
        assert len({metric for metric in metrics if metric.title == expected_title}) == 1
    assert len({metric for metric in metrics if metric.title not in expected_titles}) == 0

    for metric in metrics:
        assert isinstance(metric, TimeRangeMetric)
        assert metric.time_range.start_timestamp == timestamps[0]
        assert metric.time_range.end_timestamp == timestamps[3]
        assert metric.generator == siap_generator

        if metric.title.endswith(" at times"):
            assert isinstance(metric.value, list)
            assert len(metric.value) == 4  # number of timestamps

            for thing in metric.value:
                assert isinstance(thing, SingleTimeMetric)
                assert isinstance(thing.value, (float, int))
                assert thing.generator == siap_generator
        else:
            assert isinstance(metric.value, (float, int))
