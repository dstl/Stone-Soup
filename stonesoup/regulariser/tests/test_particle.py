import numpy as np
import datetime

from ...types.state import ParticleState
from ...types.particle import Particle
from ...types.hypothesis import SingleHypothesis
from ...types.prediction import ParticleStatePrediction, ParticleMeasurementPrediction
from ...models.measurement.linear import LinearGaussian
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from ...types.detection import Detection
from ...types.update import ParticleStateUpdate
from ..particle import MCMCRegulariser


def test_regulariser():
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity([0.05])])

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
    new_state_vector = transition_model.function(particles,
                                                 noise=True,
                                                 time_interval=datetime.timedelta(seconds=1))
    prediction = ParticleStatePrediction(new_state_vector,
                                         timestamp=timestamp,
                                         transition_model=transition_model)
    meas_pred = ParticleMeasurementPrediction(prediction, timestamp=timestamp)
    measurement_model = LinearGaussian(ndim_state=2, mapping=(0, 1), noise_covar=np.eye(2))
    measurement = [Detection(state_vector=np.array([[5], [7]]),
                             timestamp=timestamp, measurement_model=measurement_model)]
    state_update = ParticleStateUpdate(None, SingleHypothesis(prediction=prediction,
                                                              measurement=measurement,
                                                              measurement_prediction=meas_pred),
                                       particle_list=particles.particle_list,
                                       timestamp=timestamp+datetime.timedelta(seconds=1))
    regulariser = MCMCRegulariser()

    # state check
    new_particles = regulariser.regularise(prediction, state_update, measurement, transition_model)
    # Check the shape of the new state vector
    assert new_particles.state_vector.shape == state_update.state_vector.shape
    # Check weights are unchanged
    assert any(new_particles.weight == state_update.weight)
    # Check that the timestamp is the same
    assert new_particles.timestamp == state_update.timestamp

    # list check
    new_particles = regulariser.regularise(particles.particle_list, state_update.particle_list,
                                           measurement)
    # Check the shape of the new state vector
    assert new_particles.state_vector.shape == state_update.state_vector.shape
    # Check weights are unchanged
    assert any(new_particles.weight == state_update.weight)


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

    new_particles = regulariser.regularise(particles, state_update, detections=None)

    # Check the shape of the new state vector
    assert new_particles.state_vector.shape == state_update.state_vector.shape
    # Check weights are unchanged
    assert any(new_particles.weight == state_update.weight)
    # Check that the timestamp is the same
    assert new_particles.timestamp == state_update.timestamp
