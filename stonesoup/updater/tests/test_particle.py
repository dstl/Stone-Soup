"""Test for updater.particle module"""
import datetime
from functools import partial

import numpy as np
import pytest

from ...models.measurement.linear import LinearGaussian
from ...resampler.particle import SystematicResampler
from ...types.array import StateVectors
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.multihypothesis import MultipleHypothesis
from ...types.particle import Particle
from ...types.prediction import (
    ParticleStatePrediction, ParticleMeasurementPrediction)
from ...updater.particle import (
    ParticleUpdater, GromovFlowParticleUpdater,
    GromovFlowKalmanParticleUpdater, BernoulliParticleUpdater)
from ...predictor.particle import BernoulliParticlePredictor
from ...models.transition.linear import ConstantVelocity
from ...types.update import BernoulliParticleStateUpdate
from ...regulariser.particle import MCMCRegulariser
from ...sampler.particle import ParticleSampler, GaussianDetectionParticleSampler
from ...sampler.detection import DetectionSampler


@pytest.fixture(params=(
        ParticleUpdater,
        partial(ParticleUpdater, resampler=SystematicResampler()),
        GromovFlowParticleUpdater,
        GromovFlowKalmanParticleUpdater))
def updater(request):
    updater_class = request.param
    measurement_model = LinearGaussian(
        ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    return updater_class(measurement_model)


def test_particle(updater):
    # Measurement model
    timestamp = datetime.datetime.now()
    particles = [Particle([[10], [10]], 1 / 9),
                 Particle([[10], [20]], 1 / 9),
                 Particle([[10], [30]], 1 / 9),
                 Particle([[20], [10]], 1 / 9),
                 Particle([[20], [20]], 1 / 9),
                 Particle([[20], [30]], 1 / 9),
                 Particle([[30], [10]], 1 / 9),
                 Particle([[30], [20]], 1 / 9),
                 Particle([[30], [30]], 1 / 9),
                 ]

    prediction = ParticleStatePrediction(None, particle_list=particles,
                                         timestamp=timestamp)
    measurement = Detection([[20.0]], timestamp=timestamp)
    eval_measurement_prediction = ParticleMeasurementPrediction(None, particle_list=[
                                            Particle(i.state_vector[0, :], 1 / 9)
                                            for i in particles],
                                            timestamp=timestamp)

    measurement_prediction = updater.predict_measurement(prediction)

    assert np.all(eval_measurement_prediction.state_vector == measurement_prediction.state_vector)
    assert measurement_prediction.timestamp == timestamp

    updated_state = updater.update(SingleHypothesis(
        prediction, measurement, measurement_prediction))

    # Don't know what the particles will exactly be due to randomness so check
    # some obvious properties

    assert np.all(weight == 1 / 9 for weight in updated_state.weight)
    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.measurement_prediction == measurement_prediction
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert np.allclose(updated_state.mean, StateVectors([[20.0], [20.0]]), rtol=2e-2)


def test_bernoulli_particle():
    timestamp = datetime.datetime.now()
    timediff = 2
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    lg = LinearGaussian(ndim_state=2,
                        mapping=(0,),
                        noise_covar=np.array([[1]]))

    cv = ConstantVelocity(noise_diff_coeff=0)

    detection_probability = 0.9
    prior_particles = [Particle([[10], [10]], 1 / 9),
                       Particle([[10], [20]], 1 / 9),
                       Particle([[10], [30]], 1 / 9),
                       Particle([[20], [10]], 1 / 9),
                       Particle([[20], [20]], 1 / 9),
                       Particle([[20], [30]], 1 / 9),
                       Particle([[30], [10]], 1 / 9),
                       Particle([[30], [20]], 1 / 9),
                       Particle([[30], [30]], 1 / 9)]

    prior_detections = [Detection(np.array([15]), timestamp, measurement_model=lg),
                        Detection(np.array([40]), timestamp, measurement_model=lg),
                        Detection(np.array([5]), timestamp, measurement_model=lg)]

    prior_hypotheses = MultipleHypothesis([SingleHypothesis(None, detection)
                                          for detection in prior_detections])

    detections = [Detection(np.array([35]), new_timestamp, measurement_model=lg),
                  Detection(np.array([10]), new_timestamp, measurement_model=lg),
                  Detection(np.array([20]), new_timestamp, measurement_model=lg),
                  Detection(np.array([50]), new_timestamp, measurement_model=lg)]

    existence_prob = 0.5
    birth_prob = 0.01
    survival_prob = 0.98
    nbirth_parts = 9

    prior = BernoulliParticleStateUpdate(None,
                                         particle_list=prior_particles,
                                         existence_probability=existence_prob,
                                         timestamp=timestamp,
                                         hypothesis=prior_hypotheses)

    detection_sampler = GaussianDetectionParticleSampler(nbirth_parts)
    backup_sampler = ParticleSampler(distribution_func=np.random.uniform,
                                     params={'low': np.array([0, 10]),
                                             'high': np.array([30, 30]),
                                             'size': (nbirth_parts, 2)},
                                     ndim_state=2)
    sampler = DetectionSampler(detection_sampler=detection_sampler, backup_sampler=backup_sampler)

    predictor = BernoulliParticlePredictor(transition_model=cv,
                                           birth_sampler=sampler,
                                           birth_probability=birth_prob,
                                           survival_probability=survival_prob)

    prediction = predictor.predict(prior, timestamp=new_timestamp)
    resampler = SystematicResampler()
    regulariser = MCMCRegulariser()

    updater = BernoulliParticleUpdater(measurement_model=None,
                                       resampler=resampler,
                                       regulariser=regulariser,
                                       birth_probability=birth_prob,
                                       survival_probability=survival_prob,
                                       clutter_rate=2,
                                       clutter_distribution=1/10,
                                       nsurv_particles=9,
                                       detection_probability=detection_probability)

    hypotheses = MultipleHypothesis(
        [SingleHypothesis(prediction, detection) for detection in detections])

    update = updater.update(hypotheses)

    # Can't check the exact particles due to regularisation and resampling but can check the
    # updater is returning the correct information.

    # Check that the correct number of particles are returned.
    assert len(update) == 9
    # Check the timestamp.
    assert update.timestamp == new_timestamp
    # Check the weights
    assert np.around(float(np.sum(update.weight)), decimals=1) == 1.0
    # Check that the detections are output in the update state
    assert update.hypothesis is not None
    assert update.hypothesis == hypotheses
    # Check that the existence probability is returned
    assert update.existence_probability is not None
