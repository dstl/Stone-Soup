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

from ...updater.particle import RaoBlackwellisedParticleUpdater
from ...types.particle import RaoBlackwellisedParticle
from ...resampler.particle import RaoBlackwellisedSystematicResampler
from ...models.transition.linear import ConstantVelocity, ConstantAcceleration
from ...models.transition.linear import CombinedLinearGaussianTransitionModel
from ...predictor.multi_model import RaoBlackwellisedMultiModelPredictor
from ...types.prediction import ParticleState


def test_particle():
    # Measurement model
    lg = LinearGaussian(ndim_state=2, mapping=[0],
                        noise_covar=np.array([[0.04]]))
    timestamp = datetime.datetime.now()
    particles = [Particle(np.array([[10], [10]]),
                          1 / 9, dynamic_model=0),
                 Particle(np.array([[10], [20]]),
                          1 / 9, dynamic_model=0),
                 Particle(np.array([[10], [30]]),
                          1 / 9, dynamic_model=0),
                 Particle(np.array([[20], [10]]),
                          1 / 9, dynamic_model=0),
                 Particle(np.array([[20], [20]]),
                          1 / 9, dynamic_model=0),
                 Particle(np.array([[20], [30]]),
                          1 / 9, dynamic_model=0),
                 Particle(np.array([[30], [10]]),
                          1 / 9, dynamic_model=0),
                 Particle(np.array([[30], [20]]),
                          1 / 9, dynamic_model=0),
                 Particle(np.array([[30], [30]]),
                          1 / 9, dynamic_model=0),
                 ]

    prediction = ParticleStatePrediction(particles,
                                         timestamp=timestamp)
    measurement = Detection(np.array([[20]]), timestamp=timestamp)
    resampler = SystematicResampler()
    updater = ParticleUpdater(lg, resampler)
    eval_measurement_prediction = ParticleMeasurementPrediction([
                                            Particle(i.state_vector[0], 1 / 9, dynamic_model=0)
                                            for i in particles],
                                            timestamp=timestamp)

    measurement_prediction = updater.predict_measurement(prediction)

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


def test_rao_blackwellised_updater():

    measurement_model = LinearGaussian(
        ndim_state=9,
        mapping=(0, 3, 6),
        noise_covar=np.diag([0.75, 0.75, 0.75]))

    start_time = datetime.datetime.now()

    particle1 = RaoBlackwellisedParticle(state_vector=np.reshape([1, 1, 1, 1, 1, 1, 1, 1, 1], (-1, 1)), weight=0.5,
                                          model_probabilities=[0.5, 0.5], dynamic_model=0)
    particle2 = RaoBlackwellisedParticle(state_vector=np.reshape([1, 1, 1, 1, 1, 1, 1, 1, 1], (-1, 1)), weight=0.5,
                                          model_probabilities=[0.5, 0.5],  dynamic_model=1)
    particles = [particle1, particle2]

    prior_state = ParticleState(particles, timestamp=start_time - datetime.timedelta(seconds=1))

    dynamic_model_list = [CombinedLinearGaussianTransitionModel((ConstantVelocity(0.01),
                                                                 ConstantVelocity(0.01),
                                                                 ConstantVelocity(0.01))),
                          CombinedLinearGaussianTransitionModel((ConstantAcceleration(0.1),
                                                                 ConstantAcceleration(0.1),
                                                                 ConstantAcceleration(0.1)))]

    transition = [[0.95, 0.05],
                  [0.05, 0.95]]

    position_mapping = [[0, 1, 3, 4, 6, 7],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8]]

    resampler = RaoBlackwellisedSystematicResampler()
    updater = RaoBlackwellisedParticleUpdater(measurement_model=measurement_model, resampler=resampler)

    predictor = RaoBlackwellisedMultiModelPredictor(position_mapping=position_mapping, transition_matrix=transition,
                                                    transition_model=dynamic_model_list)

    measurement = Detection(np.array([1, 1, 1]), timestamp=start_time)

    prediction = predictor.predict(prior_state, timestamp=measurement.timestamp)

    hypothesis = SingleHypothesis(prediction, measurement)

    post, n_eff = updater.update(hypothesis, predictor=predictor,
                                 prior_timestamp=start_time - datetime.timedelta(seconds=1), transition=transition)

    print(post.hypothesis.prediction.particles)
    # Check to see that the probabilities sum to 1
    assert [sum(particle.model_probabilities) == 1 for particle in post.particles]
