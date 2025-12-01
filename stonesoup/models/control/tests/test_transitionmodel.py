import pytest
import numpy as np
from datetime import timedelta

from ..linear import TransitionBasedLinearControlModel
from ...transition.linear import (ConstantAcceleration,
                                  CombinedLinearGaussianTransitionModel,
                                  ConstantVelocity, ConstantNthDerivative)


@pytest.mark.parametrize(
    "transition_model, mapping, control_model_output",
    [(ConstantVelocity(1),
      [1],
      np.array([[10.]])),
     (ConstantAcceleration(1),
      [2],
      np.array([[50.], [10.]])),
     (ConstantNthDerivative(constant_derivative=3, noise_diff_coeff=1),
      [3],
      np.array([[500/3], [50], [10.]])),
     (CombinedLinearGaussianTransitionModel([ConstantVelocity(1),
                                             ConstantVelocity(1)]),
      [1, 3],
      np.array([[10., 0.], [0., 10.]])),
     (CombinedLinearGaussianTransitionModel([ConstantAcceleration(1),
                                             ConstantAcceleration(1)]),
      [2, 5],
      np.array([[50., 0.], [10., 0.], [0., 50.], [0., 10.]])),
     (CombinedLinearGaussianTransitionModel([
         ConstantNthDerivative(constant_derivative=3, noise_diff_coeff=1),
         ConstantNthDerivative(constant_derivative=3, noise_diff_coeff=1)]),
      [3, 7],
      np.array([[500/3, 0.], [50., 0.], [10., 0.], [0., 500/3], [0., 50.], [0., 10.]])),
     (CombinedLinearGaussianTransitionModel([ConstantAcceleration(1),
                                             ConstantVelocity(1)]),
      [2, 4],
      np.array([[50., 0.], [10., 0.], [0., 10.]]))],
    ids=["CV1d", "CA1d", "CJ1d", "CV2d", "CA2d", "CJ2d", "CACV"]
)
def test_transition_based_control_model(transition_model, mapping, control_model_output):

    control_model = TransitionBasedLinearControlModel(
        control_matrix=np.array([[1., 0],
                                 [1., 0],
                                 [0, 1.],
                                 [0, 1.]]),
        control_noise=np.diag([0.005, 0.005]),
        transition_model=transition_model,
        mapping=mapping
    )

    matrix = control_model.matrix(time_interval=timedelta(seconds=10))

    assert np.allclose(matrix, control_model_output)
    assert matrix.shape[1] == len(mapping)
