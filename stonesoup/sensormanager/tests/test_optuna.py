import copy
from collections import defaultdict
import pytest
from ordered_set import OrderedSet
import numpy as np

try:
    from ..optuna import OptunaSensorManager
except ImportError:
    # Catch optional dependencies import error
    pytest.skip(
        "Skipping due to missing optional dependencies. Usage of Optuna Sensor Manager requires "
        "that the optional package `optuna`is installed.",
        allow_module_level=True
    )

from ..reward import UncertaintyRewardFunction
from ...hypothesiser.distance import DistanceHypothesiser
from ...measures import Mahalanobis
from ...dataassociator.neighbour import GNNWith2DAssignment
from ...sensor.radar.radar import RadarRotatingBearingRange
from ...sensor.action.dwell_action import ChangeDwellAction


def test_optuna_manager(params):
    predictor = params['predictor']
    updater = params['updater']
    sensor_set = params['sensor_set']
    timesteps = params['timesteps']
    tracks = params['tracks']
    truths = params['truths']

    reward_function = UncertaintyRewardFunction(predictor, updater)
    optunasensormanager = OptunaSensorManager(sensor_set, reward_function=reward_function,
                                              timeout=0.1)

    hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(),
                                        missed_distance=5)
    data_associator = GNNWith2DAssignment(hypothesiser)

    sensor_history = defaultdict(dict)
    dwell_centres = dict()

    for timestep in timesteps[1:]:
        chosen_actions = optunasensormanager.choose_actions(tracks, timestep)
        measurements = set()
        for chosen_action in chosen_actions:
            for sensor, actions in chosen_action.items():
                sensor.add_actions(actions)
        for sensor in sensor_set:
            sensor.act(timestep)
            sensor_history[timestep][sensor] = copy.copy(sensor)
            dwell_centres[timestep] = sensor.dwell_centre[0][0]
            measurements |= sensor.measure(OrderedSet(truth[timestep] for truth in truths),
                                           noise=False)
        hypotheses = data_associator.associate(tracks,
                                               measurements,
                                               timestep)
        for track in tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = updater.update(hypothesis)
                track.append(post)
            else:
                track.append(hypothesis.prediction)

    # Double check choose_actions method types are as expected
    assert isinstance(chosen_actions, list)

    for chosen_actions in chosen_actions:
        for sensor, actions in chosen_action.items():
            assert isinstance(sensor, RadarRotatingBearingRange)
            assert isinstance(actions[0], ChangeDwellAction)

    # Check sensor following track as expected
    assert dwell_centres[timesteps[5]] - np.radians(135) < 1e-3
    assert dwell_centres[timesteps[15]] - np.radians(45) < 1e-3
    assert dwell_centres[timesteps[25]] - np.radians(-45) < 1e-3
    assert dwell_centres[timesteps[35]] - np.radians(-135) < 1e-3
