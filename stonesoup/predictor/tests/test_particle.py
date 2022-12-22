import datetime

import numpy as np
import pytest

from ...models.transition.linear import ConstantVelocity
from ...predictor.particle import (
    ParticlePredictor, ParticleFlowKalmanPredictor)
from ...types.particle import Particle
from ...types.prediction import ParticleStatePrediction
from ...types.state import ParticleState


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
