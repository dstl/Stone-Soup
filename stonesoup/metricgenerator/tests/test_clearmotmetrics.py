
import numpy as np
import pytest

from ...measures import Euclidean
from ...metricgenerator.manager import MultiManager
from ...types.association import AssociationSet, TimeRangeAssociation
from ...types.metric import TimeRangeMetric
from ...types.time import TimeRange
from ..clearmotmetrics import ClearMotMetrics


def test_clearmot_simple(trial_truths, trial_tracks, trial_timestamps):
    """We test the most obvious scenario, where we have a single truth track and 
    a single estimated track. They are both associated for the full time extent of the experiment.
    """
    trial_manager = MultiManager()
    trial_manager.add_data({'groundtruth_paths': trial_truths[:1],
                            'tracks': trial_tracks[:1]})
    
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

    dx = dy = 0.1
    expected_avg_pos_accuracy = np.sqrt(dx ** 2 + dy ** 2)

    metrics = clearmot_generator.compute_metric(trial_manager)

    motp = metrics[0].value
    assert motp == pytest.approx(expected_avg_pos_accuracy)

    mota = metrics[1].value
    assert mota == pytest.approx(1.0)


def test_clearmot(trial_manager, trial_truths, trial_tracks, trial_associations):
    position_measure = Euclidean((0, 2))
    clearmot_generator = ClearMotMetrics(distance_measure=position_measure)

    trial_manager.generators = [clearmot_generator]

    # Test compute_metric
    metrics = clearmot_generator.compute_metric(trial_manager)
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

        assert isinstance(metric.value, (float, int))
