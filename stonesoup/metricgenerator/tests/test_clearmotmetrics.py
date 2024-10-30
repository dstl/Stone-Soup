import numpy as np
import pytest

from ...measures import Euclidean
from ...metricgenerator.manager import MultiManager
from ...types.association import AssociationSet, TimeRangeAssociation
from ...types.metric import TimeRangeMetric
from ...types.time import TimeRange
from ...types.track import Track
from ..clearmotmetrics import AssociationSetNotValid, ClearMotMetrics


def test_clearmot_simple(trial_truths, trial_tracks, trial_timestamps):
    """We test the most obvious scenario, where we have a single truth track and
    a single estimated track. They are both associated for the full time extent of the experiment.
    """
    trial_manager = MultiManager()
    trial_manager.add_data({'groundtruth_paths': trial_truths[:1],
                            'tracks': trial_tracks[:1]})

    trial_associations = AssociationSet({
        # association for full time range (4 timesteps)
        TimeRangeAssociation(objects={trial_truths[0], trial_tracks[0]},
                             time_range=TimeRange(trial_timestamps[0], trial_timestamps[-1])),
    })
    trial_manager.association_set = trial_associations

    position_measure = Euclidean((0, 2))
    clearmot_generator = ClearMotMetrics(distance_measure=position_measure)

    trial_manager.generators = [clearmot_generator]
    metrics = clearmot_generator.compute_metric(trial_manager)

    _check_metric_interface(trial_manager, clearmot_generator, metrics)

    dx = dy = 0.1
    expected_avg_pos_accuracy = np.sqrt(dx ** 2 + dy ** 2)

    motp = metrics[0].value
    assert motp == pytest.approx(expected_avg_pos_accuracy)

    mota = metrics[1].value

    # because the track is associated with the complete extent of the truth,
    # i.e. there are no false positves or misses
    expected_mota = 1.0
    assert mota == pytest.approx(expected_mota)


def _check_metric_interface(trial_manager, clearmot_generator, metrics):
    expected_titles = ["MOTP", "MOTA"]

    # make sure that the titles are correct
    returned_metric_titles = [metric.title for metric in metrics]
    assert len(expected_titles) == len(returned_metric_titles)
    assert set(expected_titles) == set(returned_metric_titles)

    timestamps = trial_manager.list_timestamps(clearmot_generator)

    for metric in metrics:
        assert isinstance(metric, TimeRangeMetric)
        assert metric.time_range.start == timestamps[0]
        assert metric.time_range.end == timestamps[-1]
        assert metric.generator == clearmot_generator

        assert isinstance(metric.value, (float, float))


def test_clearmot_with_false_positives(trial_truths, trial_tracks, trial_timestamps):
    """Test with a single truth track and two hypothesis tracks, where the second track is
    not assigned, i.e. causing false positives over its lifetime
    """
    trial_manager = MultiManager()
    trial_manager.add_data({'groundtruth_paths': trial_truths[:1],
                            'tracks': trial_tracks[:2]})

    # we test the mo
    trial_associations = AssociationSet({
        # association for full time range (4 timesteps)
        TimeRangeAssociation(objects={trial_truths[0], trial_tracks[0]},
                             time_range=TimeRange(trial_timestamps[0], trial_timestamps[-1])),
    })
    trial_manager.association_set = trial_associations

    position_measure = Euclidean((0, 2))
    clearmot_generator = ClearMotMetrics(distance_measure=position_measure)

    trial_manager.generators = [clearmot_generator]

    metrics = clearmot_generator.compute_metric(trial_manager)

    dx = dy = 0.1
    expected_avg_pos_accuracy = np.sqrt(dx ** 2 + dy ** 2)

    motp = metrics[0].value
    assert motp == pytest.approx(expected_avg_pos_accuracy)

    mota = metrics[1].value

    num_gt_samples = len(trial_truths[0])
    num_false_positives = len(trial_tracks[1])
    expected_mota = 1.0 - (num_false_positives)/num_gt_samples

    assert mota == pytest.approx(expected_mota)


def test_clearmot_with_false_positives_and_miss_matches(trial_truths, trial_tracks,
                                                        trial_timestamps, time_period):
    """Test with a single truth track and 3 hypothesis tracks, where:
    - the first and second track are assigned to the truth track, but have different IDs over
        different periods of time, caussing an ID-mismatch
    - the third track is track is not assigned, i.e. causing false positives over its lifetime
    """
    trial_manager = MultiManager()

    cut_timestamp = trial_timestamps[2]
    track_part_a = Track(states=trial_tracks[0][:cut_timestamp], id=trial_tracks[0].id + "-a")
    track_part_b = Track(states=trial_tracks[0][cut_timestamp:], id=trial_tracks[0].id + "-b")

    trial_manager.add_data({'groundtruth_paths': trial_truths[:2],
                            'tracks': {track_part_a, track_part_b, trial_tracks[1]}})

    trial_associations = AssociationSet({
        TimeRangeAssociation(objects={trial_truths[0], track_part_a},
                             time_range=TimeRange(trial_timestamps[0], cut_timestamp - time_period)
                             ),
        TimeRangeAssociation(objects={trial_truths[0], track_part_b},
                             time_range=TimeRange(cut_timestamp, trial_timestamps[-1])),
    })
    trial_manager.association_set = trial_associations

    position_measure = Euclidean((0, 2))
    clearmot_generator = ClearMotMetrics(distance_measure=position_measure)

    trial_manager.generators = [clearmot_generator]

    metrics = clearmot_generator.compute_metric(trial_manager)

    dx = dy = 0.1
    expected_avg_pos_accuracy = np.sqrt(dx ** 2 + dy ** 2)

    motp = metrics[0].value
    assert motp == pytest.approx(expected_avg_pos_accuracy)

    mota = metrics[1].value

    num_gt_samples = len(trial_truths[0]) + len(trial_truths[1])
    num_false_positives = len(trial_tracks[1])
    num_miss_matches = 1  # ID switch at the cut timestamp
    num_misses = len(trial_truths[1])  # GT-1 was not associated at all

    expected_mota = 1.0 - (num_false_positives + num_miss_matches + num_misses)/num_gt_samples
    assert mota == pytest.approx(expected_mota)


def test_clearmot_match_check(trial_manager):
    """Since there are multiple tracks assigned with the truth at the second timestep,
    and CLEAR MOT does not support that, we raise an exception. This test checks that.
    """
    position_measure = Euclidean((0, 2))
    clearmot_generator = ClearMotMetrics(distance_measure=position_measure)

    trial_manager.generators = [clearmot_generator]

    with pytest.raises(AssociationSetNotValid):
        _ = clearmot_generator.compute_metric(trial_manager)
