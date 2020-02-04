# coding: utf-8
import datetime

import numpy as np

from ...models.transition.linear import ConstantVelocity, CombinedLinearGaussianTransitionModel, ConstantAcceleration
from ...predictor.multi_model import MultiModelPredictor
from ...types.particle import Particle
from ...types.state import ParticleState


def test_multi_model():

    # Define time related variables.
    timestamp = datetime.datetime.now()
    time_diff = 2  # 2sec.
    new_timestamp = timestamp + datetime.timedelta(seconds=time_diff)

    # Define prior state.
    prior_particles = [Particle(np.array([[10], [10], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[10], [20], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[10], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[20], [10], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[20], [20], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[20], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [10], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [20], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       Particle(np.array([[30], [30], [10], [10], [10], [10], [10], [10], [10]]),
                                1 / 9, dynamic_model=0),
                       ]

    # Declare the model list.
    model_list = [
                  CombinedLinearGaussianTransitionModel((ConstantVelocity(0.1),
                                                         ConstantVelocity(0.1),
                                                         ConstantVelocity(0.1))),
                  CombinedLinearGaussianTransitionModel((ConstantAcceleration(0.01),
                                                         ConstantAcceleration(0.01),
                                                         ConstantAcceleration(0.01))),
                  CombinedLinearGaussianTransitionModel((ConstantVelocity(0.01),
                                                         ConstantVelocity(0.01),
                                                         ConstantVelocity(0.01))),
                  CombinedLinearGaussianTransitionModel((ConstantAcceleration(0.01),
                                                         ConstantAcceleration(0.01),
                                                         ConstantAcceleration(0.01))),
                  CombinedLinearGaussianTransitionModel((ConstantVelocity(0.01),
                                                         ConstantVelocity(0.01),
                                                         ConstantAcceleration(0.01))),
                  CombinedLinearGaussianTransitionModel((ConstantAcceleration(0.01),
                                                         ConstantVelocity(0.01),
                                                         ConstantVelocity(0.01))),
                  CombinedLinearGaussianTransitionModel((ConstantVelocity(0.01),
                                                         ConstantAcceleration(0.01),
                                                         ConstantVelocity(0.01))),
                  CombinedLinearGaussianTransitionModel((ConstantAcceleration(0.01),
                                                         ConstantAcceleration(0.01),
                                                         ConstantVelocity(0.01))),
                  ]
    # Give the respective position mapping.
    position_mapping = [
                        [0, 1, 3, 4, 6, 7],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8],
                        [0, 1, 3, 4, 6, 7],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8],
                        [0, 1, 3, 4, 6, 7, 8],
                        [0, 1, 2, 3, 4, 6, 7],
                        [0, 1, 3, 4, 5, 6, 7],
                        [0, 1, 2, 3, 4, 5, 6, 7],
                      ]

    # Provide the required transition matrix.
    transition = [
                  [0.65, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                  [0.05, 0.65, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                  [0.05, 0.05, 0.65, 0.05, 0.05, 0.05, 0.05, 0.05],
                  [0.05, 0.05, 0.05, 0.65, 0.05, 0.05, 0.05, 0.05],
                  [0.05, 0.05, 0.05, 0.05, 0.65, 0.05, 0.05, 0.05],
                  [0.05, 0.05, 0.05, 0.05, 0.05, 0.65, 0.05, 0.05],
                  [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.65, 0.05],
                  [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.65],
                 ]

    prior = ParticleState(prior_particles, timestamp=timestamp)

    predictor = MultiModelPredictor(model_list=model_list,
                                    position_mapping=position_mapping,
                                    transition_matrix=transition,
                                    transition_model=model_list[0])

    prediction = predictor.predict(prior, timestamp=new_timestamp, multi_craft=True)[0]

    dynamic_model_list = [p.dynamic_model for p in prediction.particles]
    dynamic_model_proportions = [dynamic_model_list.count(i) for i in range(len(transition))]
    dynamic_model_proportions = np.array(dynamic_model_proportions)
    print(dynamic_model_proportions[dynamic_model_proportions > 0])

    assert prediction.timestamp == new_timestamp
    assert np.all([prediction.particles[i].weight == 1 / 9 for i in range(9)])
    assert len(dynamic_model_proportions[dynamic_model_proportions > 0]) == len(transition)
