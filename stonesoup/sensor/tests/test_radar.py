# -*- coding: utf-8 -*-

import numpy as np
import datetime

from stonesoup.functions import cart2pol
from stonesoup.types.state import State
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.sensor.radar import SimpleRadar


def test_simple_radar():

    # Input arguments
    # TODO: pytest parametarization
    noise_covar = CovarianceMatrix(np.array([[0.015, 0],
                                             [0, 0.1]]))
    radar_position = StateVector(
        np.array(([[1], [1]])))
    target_state = State(radar_position +
                         np.array([[1], [1]]),
                         timestamp=datetime.datetime.now())
    measurement_mapping = np.array([0, 2])

    # Create a radar object
    radar = SimpleRadar(
        position=radar_position,
        ndims=2,
        mapping=measurement_mapping,
        noise_covar=noise_covar)

    # Assert that the object has been correctly initialised
    assert(np.equal(radar.position, radar_position).all())
    assert(np.equal(radar.measurement_model.origin_offset,
                    radar_position).all())

    # Generate a noiseless measurement for the given target
    measurement = radar.gen_measurement(target_state, noise=0)
    rho, phi = cart2pol(target_state.state_vector[0][0]
                        - radar_position[0][0],
                        target_state.state_vector[1][0]
                        - radar_position[1][0])

    # Assert correction of generated measurement
    assert(measurement.timestamp == target_state.timestamp)
    assert(np.equal(measurement.state_vector,
                    StateVector(np.array([[phi], [rho]]))).all())
