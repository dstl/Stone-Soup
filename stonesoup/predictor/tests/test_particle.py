import itertools

import datetime
import copy

import numpy as np
import pytest

from ...models.transition.linear import ConstantVelocity
from ...predictor.particle import (
    ParticlePredictor, ParticleFlowKalmanPredictor, BernoulliParticlePredictor, SMCPHDPredictor)
from ...types.array import StateVector
from ...types.numeric import Probability
from ...types.particle import Particle
from ...types.prediction import ParticleStatePrediction, BernoulliParticleStatePrediction
from ...types.update import BernoulliParticleStateUpdate
from ...types.state import ParticleState, BernoulliParticleState
from ...models.measurement.linear import LinearGaussian
from ...types.detection import Detection
from ...sampler.particle import ParticleSampler
from ...sampler.detection import SwitchingDetectionSampler, GaussianDetectionParticleSampler
from ...functions import gm_sample
from ...types.hypothesis import SingleHypothesis
from ...types.multihypothesis import MultipleHypothesis


@pytest.mark.parametrize(
    "predictor_class",
    (ParticlePredictor, ParticleFlowKalmanPredictor))
def test_particle(predictor_class):
    # Initialise a transition model
    cv = ConstantVelocity(noise_diff_coeff=0)

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    # Define prior state
    prior_particles = [Particle(np.array([[10], [10]]),
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
    prior = ParticleState(None, particle_list=prior_particles, timestamp=timestamp)

    eval_particles = [Particle(cv.matrix(timestamp=new_timestamp,
                                         time_interval=time_interval)
                               @ particle.state_vector,
                               1 / 9)
                      for particle in prior_particles]
    eval_mean = np.mean(np.hstack([i.state_vector for i in eval_particles]),
                        axis=1).reshape(2, 1)

    eval_prediction = ParticleStatePrediction(None, new_timestamp, particle_list=eval_particles)

    predictor = predictor_class(transition_model=cv)

    prediction = predictor.predict(prior, timestamp=new_timestamp)

    assert np.allclose(prediction.mean, eval_mean)
    assert prediction.timestamp == new_timestamp
    assert np.all([eval_prediction.state_vector[:, i] ==
                   prediction.state_vector[:, i] for i in range(9)])
    assert np.all([prediction.weight[i] == 1 / 9 for i in range(9)])


def test_bernoulli_particle_no_detection():
    # Initialise transition model
    cv = ConstantVelocity(noise_diff_coeff=0)

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2
    new_timestamp = timestamp+datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    random_seed = 1990
    nbirth_parts = 9
    prior_particles = [Particle(np.array([[10], [10]]), 1 / 9),
                       Particle(np.array([[10], [20]]), 1 / 9),
                       Particle(np.array([[10], [30]]), 1 / 9),
                       Particle(np.array([[20], [10]]), 1 / 9),
                       Particle(np.array([[20], [20]]), 1 / 9),
                       Particle(np.array([[20], [30]]), 1 / 9),
                       Particle(np.array([[30], [10]]), 1 / 9),
                       Particle(np.array([[30], [20]]), 1 / 9),
                       Particle(np.array([[30], [30]]), 1 / 9)]

    existence_prob = 0.5
    birth_prob = 0.01
    survival_prob = 0.98

    prior = BernoulliParticleState(None,
                                   particle_list=prior_particles,
                                   existence_probability=existence_prob,
                                   timestamp=timestamp)

    backup_sampler = ParticleSampler(distribution_func=np.random.uniform,
                                     params={'low': np.array([0, 10]),
                                             'high': np.array([30, 30]),
                                             'size': (nbirth_parts, 2)},
                                     ndim_state=2)
    detection_sampler = GaussianDetectionParticleSampler(nbirth_parts)
    sampler = SwitchingDetectionSampler(detection_sampler=detection_sampler,
                                        backup_sampler=backup_sampler)

    np.random.seed(random_seed)
    birth_samples = np.random.uniform(np.array([0, 10]), np.array([30, 30]), (nbirth_parts, 2))

    eval_prior = copy.copy(prior)
    eval_prior.state_vector = np.concatenate((eval_prior.state_vector,
                                             birth_samples.T),
                                             axis=1)
    eval_prior.weight = np.array([1/18]*18)

    eval_prediction = [Particle(cv.matrix(
        timestamp=new_timestamp,
        time_interval=time_interval) @ particle.state_vector,
        weight=1/18) for particle in eval_prior]

    eval_prediction = BernoulliParticleStatePrediction(None,
                                                       timestamp=new_timestamp,
                                                       particle_list=eval_prediction)

    eval_existence = birth_prob * (1 - existence_prob) + survival_prob * existence_prob

    eval_weight = eval_prior.weight

    eval_weight[:9] = survival_prob*existence_prob/eval_existence * eval_weight[:9]
    eval_weight[9:] = birth_prob*(1-existence_prob)/eval_existence * eval_weight[9:]
    eval_weight = eval_weight/np.sum(eval_weight)

    np.random.seed(random_seed)
    predictor = BernoulliParticlePredictor(transition_model=cv,
                                           birth_sampler=sampler,
                                           birth_probability=birth_prob,
                                           survival_probability=survival_prob)

    prediction = predictor.predict(prior, timestamp=new_timestamp)

    # check that the correct number of particles are output
    assert len(prediction) == len(eval_prediction)
    # check that the prior is correct
    assert np.allclose(eval_prior.state_vector, prediction.parent.state_vector)
    # check that the prediction is correct
    assert np.allclose(eval_prediction.state_vector, prediction.state_vector)
    # check timestamp
    assert prediction.timestamp == new_timestamp
    # check that the existence estimate is correct
    assert prediction.existence_probability == eval_existence
    # check that the weight prediction is correct
    assert np.allclose(eval_weight, prediction.weight.astype(np.float64))
    # check that the weights are normalised
    assert np.around(float(np.sum(prediction.weight)), decimals=1) == 1


def test_bernoulli_particle_detection():
    # Initialise transition model
    cv = ConstantVelocity(noise_diff_coeff=0)
    lg = LinearGaussian(ndim_state=2,
                        mapping=(0,),
                        noise_covar=np.array([[1]]))

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2
    new_timestamp = timestamp+datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    random_seed = 1990
    nbirth_parts = 9
    prior_particles = [Particle(np.array([[10], [10]]), 1 / 9),
                       Particle(np.array([[10], [20]]), 1 / 9),
                       Particle(np.array([[10], [30]]), 1 / 9),
                       Particle(np.array([[20], [10]]), 1 / 9),
                       Particle(np.array([[20], [20]]), 1 / 9),
                       Particle(np.array([[20], [30]]), 1 / 9),
                       Particle(np.array([[30], [10]]), 1 / 9),
                       Particle(np.array([[30], [20]]), 1 / 9),
                       Particle(np.array([[30], [30]]), 1 / 9)]

    detections = [Detection(np.array([5]), timestamp, measurement_model=lg),
                  Detection(np.array([7]), timestamp, measurement_model=lg),
                  Detection(np.array([15]), timestamp, measurement_model=lg)]

    existence_prob = 0.5
    birth_prob = 0.01
    survival_prob = 0.98

    hypotheses = MultipleHypothesis([SingleHypothesis(None, detection)
                                     for detection in detections])
    prior = BernoulliParticleStateUpdate(None,
                                         particle_list=prior_particles,
                                         existence_probability=existence_prob,
                                         timestamp=timestamp,
                                         hypothesis=hypotheses)

    detection_sampler = GaussianDetectionParticleSampler(nbirth_parts)
    backup_sampler = ParticleSampler(distribution_func=np.random.uniform,
                                     params={'low': np.array([0, 10]),
                                             'high': np.array([30, 30]),
                                             'size': (nbirth_parts, 2)},
                                     ndim_state=2)
    sampler = SwitchingDetectionSampler(detection_sampler=detection_sampler,
                                        backup_sampler=backup_sampler)

    np.random.seed(random_seed)
    birth_samples = gm_sample([np.array([detections[0].state_vector[0], 0]),
                              np.array([detections[1].state_vector[0], 0]),
                              np.array([detections[2].state_vector[0], 0])],
                              [np.diag([detections[0].measurement_model.noise_covar[0, 0], 0]),
                              np.diag([detections[1].measurement_model.noise_covar[0, 0], 0]),
                              np.diag([detections[2].measurement_model.noise_covar[0, 0], 0])],
                              nbirth_parts,
                              np.array([1/3]*3))

    eval_prior = copy.copy(prior)
    eval_prior.state_vector = np.concatenate((eval_prior.state_vector,
                                             birth_samples),
                                             axis=1)
    eval_prior.weight = np.array([1/18]*18)

    eval_prediction = [Particle(cv.matrix(
        timestamp=new_timestamp,
        time_interval=time_interval) @ particle.state_vector,
        weight=1/18) for particle in eval_prior]

    eval_prediction = BernoulliParticleStatePrediction(None,
                                                       timestamp=new_timestamp,
                                                       particle_list=eval_prediction)

    eval_existence = birth_prob * (1 - existence_prob) + survival_prob * existence_prob

    eval_weight = eval_prior.weight

    eval_weight[:9] = survival_prob*existence_prob/eval_existence * eval_weight[:9]
    eval_weight[9:] = birth_prob*(1-existence_prob)/eval_existence * eval_weight[9:]
    eval_weight = eval_weight/np.sum(eval_weight)

    predictor = BernoulliParticlePredictor(transition_model=cv,
                                           birth_sampler=sampler,
                                           birth_probability=birth_prob,
                                           survival_probability=survival_prob)

    np.random.seed(random_seed)
    prediction = predictor.predict(prior, timestamp=new_timestamp)

    # check that the correct number of particles are output
    assert len(prediction) == len(eval_prediction)
    # check that the prior is correct
    assert np.allclose(eval_prior.state_vector, prediction.parent.state_vector)
    # check that the prediction is correct
    assert np.allclose(eval_prediction.state_vector, prediction.state_vector)
    # check timestamp
    assert prediction.timestamp == new_timestamp
    # check that the existence estimate is correct
    assert prediction.existence_probability == eval_existence
    # check that the weight prediction is correct
    assert np.allclose(eval_weight, prediction.weight.astype(np.float64))
    # check that the weights are normalised
    assert np.around(float(np.sum(prediction.weight)), decimals=1) == 1


@pytest.mark.parametrize(
    "birth_scheme",
    ('mixture', 'expansion', 'some_other_scheme'))
def test_smcphd(birth_scheme):

    # Initialise a transition model
    cv = ConstantVelocity(noise_diff_coeff=0)

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    # Parameters for SMC-PHD
    death_probability = Probability(0.01)  # Probability of death
    birth_probability = Probability(0.1)  # Probability of birth
    birth_rate = 0.05  # Birth-rate (Mean number of new targets per scan)
    birth_sampler = ParticleSampler(distribution_func=np.random.multivariate_normal,
                                    params={'mean': np.array([20., 0.0]),
                                            'cov': np.diag([10. ** 2, 1. ** 2])},
                                    ndim_state=2)
    num_particles = 9  # Number of particles

    # Define prior state
    prior_particles = [Particle(np.array([[i], [j]]), 1/num_particles)
                       for i, j in itertools.product([10, 20, 30], [10, 20, 30])]
    prior = ParticleState(None, particle_list=prior_particles, timestamp=timestamp)

    if birth_scheme == 'some_other_scheme':
        with pytest.raises(ValueError):
            SMCPHDPredictor(transition_model=cv,
                            death_probability=death_probability,
                            birth_probability=birth_probability,
                            birth_rate=birth_rate,
                            birth_sampler=birth_sampler,
                            birth_func_num_samples_field='size',
                            birth_scheme=birth_scheme)
        return

    predictor = SMCPHDPredictor(transition_model=cv,
                                death_probability=death_probability,
                                birth_probability=birth_probability,
                                birth_rate=birth_rate,
                                birth_sampler=birth_sampler,
                                birth_func_num_samples_field='size',
                                birth_scheme=birth_scheme)

    # Ensure same random numbers are generated
    np.random.seed(16549)

    # Perform the prediction
    prediction = predictor.predict(prior, timestamp=new_timestamp)

    # Compute the expected survival probability
    prob_survive = np.exp(-float(death_probability) * time_interval.total_seconds())

    # Compute the expected surviving particles
    # NOTE: The line below is valid since process noise is zero
    eval_particles = [Particle(cv.matrix(timestamp=new_timestamp,
                                         time_interval=time_interval)
                               @ particle.state_vector,
                               prob_survive * particle.weight)
                      for particle in prior_particles]
    if birth_scheme == 'mixture':
        # NOTE: In the lines below, we utilise the knowledge that the above configuration results
        #       in a single birth particle, that replaces the first particle in the prior, whose
        #       state vector is known. This might not be the case for other configurations, or if
        #       the random seed is changed.
        birth_weight = birth_rate / num_particles
        new_weight = (prob_survive + birth_weight) * 1 / num_particles
        eval_particles[0].state_vector = StateVector([[11.31091636],
                                                      [-1.39374536]])
        for particle in eval_particles:
            particle.weight = new_weight

    else:
        # NOTE: Once again, we utilise the knowledge that the above configuration results in a
        #       single birth particle, that is appended to the prior (since birth_scheme is
        #       'expansion'), and whose state vector is known. This might not be the case for
        #       other configurations, or if the random seed is changed.
        num_birth = round(float(birth_probability) * num_particles)
        birth_weight = birth_rate / num_birth
        eval_particles.append(Particle(state_vector=StateVector([[18.3918058],
                                                                 [0.31072265]]),
                                       weight=birth_weight,
                                       parent=None))

    # Construct the expected prediction
    eval_prediction = ParticleStatePrediction(None, new_timestamp, particle_list=eval_particles)

    # Check that the computed mean is close to the expected mean
    assert np.allclose(prediction.mean, eval_prediction.mean)
    # Check that the timestamp is correct
    assert prediction.timestamp == new_timestamp
    # Check that the state vectors are correct
    assert np.allclose(eval_prediction.state_vector, prediction.state_vector)
    # Check that the weights are correct
    assert np.allclose(prediction.log_weight, eval_prediction.log_weight)
