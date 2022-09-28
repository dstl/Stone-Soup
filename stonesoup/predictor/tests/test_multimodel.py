import datetime

import numpy as np

from ...models.transition.linear import (
    ConstantVelocity, CombinedLinearGaussianTransitionModel, ConstantAcceleration)
from ...predictor.particle import MultiModelPredictor
from ...types.array import StateVectors
from ...types.state import MultiModelParticleState


def test_multi_model():

    # Define time related variables.
    timestamp = datetime.datetime.now()
    time_diff = 2  # 2sec.
    new_timestamp = timestamp + datetime.timedelta(seconds=time_diff)

    # Define prior state.
    prior_vectors = np.full((9, 10), 10.).view(StateVectors)
    # Change particles starting x position, and velocity
    prior_vectors[0, :] = [10, 10, 10, 20, 20, 20, 30, 30, 30, 10]
    prior_vectors[1, :] = [10, 20, 30, 10, 20, 30, 10, 20, 30, 30]
    weight = np.full((prior_vectors.shape[1],), 1/prior_vectors.shape[1])
    model = np.full((prior_vectors.shape[1], ), 0)
    prior = MultiModelParticleState(
        prior_vectors, weight=weight, dynamic_model=model, timestamp=timestamp)

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
    position_mappings = [
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

    predictor = MultiModelPredictor(position_mappings=position_mappings,
                                    transition_matrix=transition,
                                    transition_models=model_list)

    prediction = predictor.predict(prior, timestamp=new_timestamp)

    dynamic_model_list = [p.dynamic_model for p in prediction.particles]
    dynamic_model_proportions = [dynamic_model_list.count(i) for i in range(len(transition))]
    dynamic_model_proportions = np.array(dynamic_model_proportions)

    assert prediction.timestamp == new_timestamp
    assert np.all([prediction.particles[i].weight == 1/10 for i in range(9)])
    assert len(dynamic_model_proportions) == len(transition)
