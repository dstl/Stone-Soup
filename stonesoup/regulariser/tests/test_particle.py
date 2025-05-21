import datetime

import numpy as np
import pytest

from ..particle import MCMCRegulariser, MultiModelMCMCRegulariser
from ...models.measurement.linear import LinearGaussian
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from ...predictor.particle import MultiModelPredictor
from ...resampler.particle import SystematicResampler
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.particle import Particle, MultiModelParticle
from ...types.prediction import ParticleStatePrediction, ParticleMeasurementPrediction
from ...types.state import ParticleState, MultiModelParticleState
from ...types.update import Update, ParticleStateUpdate
from ...updater.particle import MultiModelParticleUpdater
from ...updater.tests.test_multi_model_particle import (  # noqa: F401
    dynamic_model_list, position_mappings, transition_matrix, resampler)


def dummy_constraint_function(particles):
    part_indx = particles.state_vector[1, :] > 30
    return part_indx


@pytest.mark.parametrize(
    "transition_model, model_flag, constraint_func",
    [
        (
            CombinedLinearGaussianTransitionModel([ConstantVelocity([0.05])]),  # transition_model
            False,  # model_flag
            None  # constraint_function
        ),
        (
            CombinedLinearGaussianTransitionModel([ConstantVelocity([0.05])]),  # transition_model
            True,  # model_flag
            None  # constraint_function
        ),
        (
            None,  # transition_model
            False,  # model_flag
            None  # constraint_function
        ),
        (
            CombinedLinearGaussianTransitionModel([ConstantVelocity([0.05])]),  # transition_model
            False,  # model_flag
            dummy_constraint_function  # constraint_function
        )
    ],
    ids=["with_transition_model_init", "without_transition_model_init", "no_transition_model",
         "with_constraint_function"]
)
def test_regulariser(transition_model, model_flag, constraint_func):
    timestamp = datetime.datetime.now()
    particles = ParticleState(state_vector=None, particle_list=[Particle(np.array([[10], [10]]),
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
                                                                ], timestamp=timestamp)
    if transition_model is not None:
        new_state_vector = transition_model.function(particles,
                                                     noise=True,
                                                     time_interval=datetime.timedelta(seconds=1))
    else:
        new_state_vector = particles.state_vector

    prediction = ParticleStatePrediction(new_state_vector,
                                         timestamp=timestamp,
                                         transition_model=transition_model)

    measurement_model = LinearGaussian(ndim_state=2, mapping=(0, 1), noise_covar=np.eye(2))
    measurement = Detection(state_vector=np.array([[5], [7]]),
                            timestamp=timestamp, measurement_model=measurement_model)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=None)

    state_update = Update.from_state(state=prediction,
                                     hypothesis=hypothesis,
                                     timestamp=timestamp+datetime.timedelta(seconds=1))
    if constraint_func:
        indx = constraint_func(state_update)
        state_update.state_vector[:, indx] = particles.state_vector[:, indx]
    # A PredictedParticleState is used here as the point at which the regulariser is implemented
    # in the updater is before the updated state has taken the updated state type.
    state_update.weight = np.array([1/6, 5/48, 5/48, 5/48, 5/48, 5/48, 5/48, 5/48, 5/48])

    if model_flag:
        regulariser = MCMCRegulariser(constraint_func=constraint_func)
    else:
        regulariser = MCMCRegulariser(transition_model=transition_model,
                                      constraint_func=constraint_func)

    # state check
    new_particles = regulariser.regularise(particles, state_update)
    # Check the shape of the new state vector
    assert new_particles.state_vector.shape == state_update.state_vector.shape
    # Check weights are unchanged
    assert any(new_particles.weight == state_update.weight)
    # Check that the timestamp is the same
    assert new_particles.timestamp == state_update.timestamp
    # Check that moved particles have been reverted back to original states if constrained
    if constraint_func is not None:
        indx = constraint_func(prediction)  # likely unconstrained particles
        assert np.all(new_particles.state_vector[:, indx] == state_update.state_vector[:, indx])

    # list check3
    with pytest.raises(TypeError) as e:
        new_particles = regulariser.regularise(particles.particle_list,
                                               state_update)
    assert "Only ParticleState type is supported!" in str(e.value)
    with pytest.raises(Exception) as e:
        new_particles = regulariser.regularise(particles,
                                               state_update.particle_list)
    assert "Only ParticleState type is supported!" in str(e.value)


def test_no_measurement():
    particles = ParticleState(state_vector=None, particle_list=[Particle(np.array([[10], [10]]),
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
                                                                ])
    timestamp = datetime.datetime.now()
    prediction = ParticleStatePrediction(None, particle_list=particles.particle_list,
                                         timestamp=timestamp)
    meas_pred = ParticleMeasurementPrediction(None, particle_list=particles, timestamp=timestamp)
    state_update = ParticleStateUpdate(None, SingleHypothesis(prediction=prediction,
                                                              measurement=None,
                                                              measurement_prediction=meas_pred),
                                       particle_list=particles.particle_list, timestamp=timestamp)
    regulariser = MCMCRegulariser()

    new_particles = regulariser.regularise(particles, state_update)

    # Check the shape of the new state vector
    assert new_particles.state_vector.shape == state_update.state_vector.shape
    # Check weights are unchanged
    assert any(new_particles.weight == state_update.weight)
    # Check that the timestamp is the same
    assert new_particles.timestamp == state_update.timestamp


def test_multi_model_regulariser(
        dynamic_model_list, position_mappings, transition_matrix):  # noqa: F811

    # Initialise particles
    particle1 = MultiModelParticle(
        state_vector=[1, 1, 0.5, 1, 1, 0.5],
        weight=1/3000,
        dynamic_model=0)
    particle2 = MultiModelParticle(
        state_vector=[1, 1, 0.5, 1, 1, 0.5],
        weight=1/3000,
        dynamic_model=1)
    particle3 = MultiModelParticle(
        state_vector=[1, 1, 0.5, 1, 1, 0.5],
        weight=1/3000,
        dynamic_model=2)

    particles = [particle1, particle2, particle3] * 1000
    timestamp = datetime.datetime.now()

    particle_state = MultiModelParticleState(
        None, particle_list=particles, timestamp=timestamp)

    predictor = MultiModelPredictor(
        dynamic_model_list, transition_matrix, position_mappings
    )

    timestamp += datetime.timedelta(seconds=5)
    prediction = predictor.predict(particle_state, timestamp)

    measurement_model = LinearGaussian(6, [0, 3], np.diag([2, 2]))
    updater = MultiModelParticleUpdater(measurement_model, predictor, SystematicResampler())

    # Detection close to where known turn rate model would place particles
    detection = Detection([[6., 7.]], timestamp, measurement_model)

    update = updater.update(hypothesis=SingleHypothesis(prediction, detection))

    regulariser = MultiModelMCMCRegulariser(
        dynamic_model_list,
        position_mappings)

    new_particles = regulariser.regularise(particle_state, update)

    assert not np.array_equal(update.state_vector, new_particles.state_vector)
    assert np.array_equal(update.log_weight, new_particles.log_weight)
