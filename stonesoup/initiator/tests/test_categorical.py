# -*- coding: utf-8 -*-

from datetime import datetime

import numpy as np

from ..categorical import SimpleCategoricalMeasurementInitiator
from ...models.measurement.categorical import MarkovianMeasurementModel
from ...types.array import StateVector
from ...types.detection import CategoricalDetection
from ...types.state import CategoricalState
from ...types.update import CategoricalStateUpdate
from ...updater.categorical import HMMUpdater


def test_categorical_initiator():
    E = np.array([[30, 25, 5],
                  [20, 25, 10],
                  [10, 25, 80],
                  [40, 25, 5]])
    measurement_model = MarkovianMeasurementModel(E)

    updater = HMMUpdater(measurement_model)

    now = datetime.now()

    # Prior state information
    prior_state = CategoricalState([80, 10, 10], timestamp=now)

    measurement_categories = ['red', 'green', 'blue', 'yellow']
    detection1 = CategoricalDetection(StateVector([10, 20, 30, 40]),
                                      timestamp=now,
                                      categories=measurement_categories)
    detection2 = CategoricalDetection(StateVector([40, 30, 20, 10]),
                                      timestamp=now,
                                      categories=measurement_categories)
    detection3 = CategoricalDetection(StateVector([10, 40, 40, 10]),
                                      timestamp=now,
                                      categories=measurement_categories)
    detections = {detection1, detection2, detection3}

    initiator = SimpleCategoricalMeasurementInitiator(prior_state=prior_state, updater=updater)

    tracks = initiator.initiate(detections)

    assert len(tracks) == 3
    for track in tracks:
        assert len(track) == 1
        assert isinstance(track.state, CategoricalStateUpdate)
        assert track.state.hypothesis.measurement in detections
