# -*- coding: utf-8 -*-
"""Test for updater.particle module"""
import numpy as np
import datetime

from ...models.measurement.linear import LinearGaussian
from ...resampler.particle import SystematicResampler
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.particle import Particle
from ...types.prediction import (
    ParticleStatePrediction, ParticleMeasurementPrediction)
from ...updater.particle import ParticleUpdater


def test_particle():
    # Measurement model
    lg = LinearGaussian(ndim_state=2, mapping=[0],
                        noise_covar=np.array([[0.04]]))
    timestamp = datetime.datetime.now()
    particles = [Particle(np.array([[10], [10]]),
                          1 / 9),
                 Particle(np.array([[10], [20]]),
                          1 / 9),
                 Particle(np.array([[10], [30]]),
                          1 / 9),
                 Particle(np.array([[20], [10]]),
                          1 / 9),
                 Particle(np.array([[20], [20]]),
                          1 / 9),
                 Particle(np.array([[20], [30]]),
                          1 / 9),
                 Particle(np.array([[30], [10]]),
                          1 / 9),
                 Particle(np.array([[30], [20]]),
                          1 / 9),
                 Particle(np.array([[30], [30]]),
                          1 / 9),
                 ]

    prediction = ParticleStatePrediction(particles,
                                         timestamp=timestamp)
    measurement = Detection(np.array([[20]]), timestamp=timestamp)
    resampler = SystematicResampler()
    updater = ParticleUpdater(lg, resampler)
    eval_measurement_prediction = ParticleMeasurementPrediction([
                                            Particle(i.state_vector[0], 1 / 9)
                                            for i in particles],
                                            timestamp=timestamp)

    measurement_prediction = updater.get_measurement_prediction(prediction)

    assert np.all([eval_measurement_prediction.particles[i].state_vector ==
                   measurement_prediction.particles[i].state_vector
                   for i in range(9)])
    assert measurement_prediction.timestamp == timestamp

    updated_state = updater.update(SingleHypothesis(
        prediction, measurement, measurement_prediction))

    # Don't know what the particles will exactly be due to randomness so check
    # some obvious properties

    assert np.all(particle.weight == 1 / 9
                  for particle in updated_state.particles)
    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.measurement_prediction \
        == measurement_prediction
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert np.all(
        np.isclose(updated_state.state_vector, np.array([[20], [20]])))
