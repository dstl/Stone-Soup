import numpy as np
import datetime
import pytest

from ...types.state import ParticleState
from ...types.particle import Particle
from ...types.hypothesis import SingleHypothesis
from ...types.prediction import ParticleStatePrediction, ParticleMeasurementPrediction
from ...models.measurement.linear import LinearGaussian
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from ...types.detection import Detection
from ...types.update import Update, ParticleStateUpdate
from ..particle import MCMCRegulariser


@pytest.mark.parametrize(
    "transition_model, model_flag",
    [
        (
            CombinedLinearGaussianTransitionModel([ConstantVelocity([0.05])]),  # transition_model
            False,  # model_flag
        ),
        (
            CombinedLinearGaussianTransitionModel([ConstantVelocity([0.05])]),  # transition_model
            True,  # model_flag
        ),
        (
            None,  # transition_model
            False,  # model_flag
        )
    ],
    ids=["with_transition_model_init", "without_transition_model_init", "no_transition_model"]
)
def test_regulariser(transition_model, model_flag):
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
    # A PredictedParticleState is used here as the point at which the regulariser is implemented
    # in the updater is before the updated state has taken the updated state type.
    state_update.weight = np.array([1/6, 5/48, 5/48, 5/48, 5/48, 5/48, 5/48, 5/48, 5/48])

    if model_flag:
        regulariser = MCMCRegulariser()
    else:
        regulariser = MCMCRegulariser(transition_model=transition_model)

    # state check
    new_particles = regulariser.regularise(prediction, state_update)
    # Check the shape of the new state vector
    assert new_particles.state_vector.shape == state_update.state_vector.shape
    # Check weights are unchanged
    assert any(new_particles.weight == state_update.weight)
    # Check that the timestamp is the same
    assert new_particles.timestamp == state_update.timestamp

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
