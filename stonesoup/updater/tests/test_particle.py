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

    # Initialise two particles
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

    transition = [[0.50, 0.50],
                  [0.50, 0.50]]

    position_mapping = [[0, 1, 3, 4, 6, 7],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8]]

    resampler = RaoBlackwellisedSystematicResampler()
    updater = RaoBlackwellisedParticleUpdater(measurement_model=measurement_model, resampler=resampler)

    predictor = RaoBlackwellisedMultiModelPredictor(position_mapping=position_mapping, transition_matrix=transition,
                                                    transition_model=dynamic_model_list)
    # Set a measurement to be a position of (1, 1, 1)
    measurement = Detection(np.array([1, 1, 1]), timestamp=start_time)

    prediction = predictor.predict(prior_state, timestamp=measurement.timestamp)

    # Set the state vectors so that there is no randomness to the output.
    # Modelled by ConstantVelocity
    prediction.particles[0].state_vector = np.reshape([2, 1, 0, 2, 1, 0, 2, 1, 0], (-1, 1))
    # Modelled by ConstantAcceleration
    prediction.particles[1].state_vector = np.reshape([3, 2, 1, 3, 2, 1, 3, 2, 1], (-1, 1))

    def determine_particle_1_probs():

        """Forms the new Rao-Blackwellised probabilities associated with particle 1."""

        # find the noiseless application of the ConstantVelocity model on the previous value.
        particle_1_cv_mean = dynamic_model_list[0].function(particle1.state_vector[position_mapping[0]],
                                                            time_interval=datetime.timedelta(seconds=1), noise=False)
        # find the noiseless application of the ConstantAcceleration model on the previous value.
        particle_1_ca_mean = dynamic_model_list[1].function(particle1.state_vector[position_mapping[1]],
                                                            time_interval=datetime.timedelta(seconds=1), noise=False)
        # Put back the values that were previously taken out to apply the dynamic models.
        for j in range(len(particle1.state_vector)):
            if j not in predictor.position_mapping[0]:
                particle_1_cv_mean = np.insert(particle_1_cv_mean, j, particle1.state_vector[j])

        for j in range(len(particle1.state_vector)):
            if j not in predictor.position_mapping[1]:
                particle_1_ca_mean = np.insert(particle_1_ca_mean, j, particle1.state_vector[j])
        # Find the Probability of the position given ConstantVelocity and the previous position.
        prob_particle_1_given_cv = measurement_model.pdf(
            measurement_model.matrix() @ prediction.particles[0].state_vector,
            particle_1_cv_mean
        )
        # Find the Probability of the position given ConstantAcceleration and the previous position.
        prob_particle_1_given_ca = measurement_model.pdf(
            measurement_model.matrix() @ prediction.particles[0].state_vector,
            particle_1_ca_mean
        )
        # Get the transition probabilities from the transition matrix above.
        prob_particle_1_cv_transition = transition[prediction.particles[0].parent.dynamic_model][0]
        prob_particle_1_ca_transition = transition[prediction.particles[0].parent.dynamic_model][1]
        # Get the previous model probabilities set in the particle above.
        previous_probs_particle_1 = prediction.particles[0].parent.model_probabilities
        # Take the sum of these values for each model to get the denominator in the Rao-Blackwellised summation.
        denominator_particle_1 = [previous_probs_particle_1[0] *
                                  prob_particle_1_cv_transition *
                                  prob_particle_1_given_cv,
                                  previous_probs_particle_1[1] *
                                  prob_particle_1_ca_transition *
                                  prob_particle_1_given_ca]
        # These are the new model probabilities associated with each model.
        new_probs_particle_1 = [denominator_particle_1[0] / sum(denominator_particle_1),
                                denominator_particle_1[1] / sum(denominator_particle_1)]
        # Return these new model probabilities.
        return new_probs_particle_1

    def determine_particle_2_probs():

        """Forms the new Rao-Blackwellised probabilities associated with particle 2."""

        particle_2_cv_mean = dynamic_model_list[0].function(particle2.state_vector[position_mapping[0]],
                                                            time_interval=datetime.timedelta(seconds=1), noise=False)
        particle_2_ca_mean = dynamic_model_list[1].function(particle2.state_vector[position_mapping[1]],
                                                            time_interval=datetime.timedelta(seconds=1), noise=False)
        for j in range(len(particle2.state_vector)):
            if j not in predictor.position_mapping[0]:
                particle_2_cv_mean = np.insert(particle_2_cv_mean, j, particle1.state_vector[j])

        for j in range(len(particle2.state_vector)):
            if j not in predictor.position_mapping[1]:
                particle_2_ca_mean = np.insert(particle_2_ca_mean, j, particle1.state_vector[j])

        prob_particle_2_given_cv = measurement_model.pdf(
            measurement_model.matrix() @ prediction.particles[1].state_vector,
            particle_2_cv_mean
        )
        prob_particle_2_given_ca = measurement_model.pdf(
            measurement_model.matrix() @ prediction.particles[1].state_vector,
            particle_2_ca_mean
        )
        prob_particle_2_cv_transition = transition[prediction.particles[1].parent.dynamic_model][0]
        prob_particle_2_ca_transition = transition[prediction.particles[1].parent.dynamic_model][1]

        previous_probs_particle_2 = prediction.particles[1].parent.model_probabilities

        denominator_particle_2 = [previous_probs_particle_2[0] *
                                  prob_particle_2_cv_transition *
                                  prob_particle_2_given_cv,
                                  previous_probs_particle_2[1] *
                                  prob_particle_2_ca_transition *
                                  prob_particle_2_given_ca]

        new_probs_particle_2 = [denominator_particle_2[0] / sum(denominator_particle_2),
                                denominator_particle_2[1] / sum(denominator_particle_2)]
        return new_probs_particle_2
    # Call the above functions.
    new_probs_for_particle_1 = determine_particle_1_probs()
    new_probs_for_particle_2 = determine_particle_2_probs()

    hypothesis = SingleHypothesis(prediction, measurement)
    # Run the actual updater on these two particles.
    post, n_eff = updater.update(hypothesis, predictor=predictor,
                                 prior_timestamp=start_time - datetime.timedelta(seconds=1), transition=transition)

    # Check to see that the probabilities sum to 1.
    assert [sum(particle.model_probabilities) == 1 for particle in post.particles]
    # Run assertions to see if the updater algorithm matches that of the manual Rao-Blackwellised calculation.
    assert np.allclose(new_probs_for_particle_1, post.particles[0].model_probabilities)
    assert np.allclose(new_probs_for_particle_2, post.particles[1].model_probabilities)
