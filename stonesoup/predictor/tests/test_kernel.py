import copy
import datetime

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from ..kernel import AdaptiveKernelKalmanPredictor
from ...kernel import QuadraticKernel, QuarticKernel
from ...models.transition.linear import ConstantVelocity, CombinedLinearGaussianTransitionModel
from ...types.array import StateVectors
from ...types.state import KernelParticleState, State
from ...types.update import KernelParticleStateUpdate

number_particles = 4
timestamp = datetime.datetime.now()
samples = multivariate_normal.rvs([0, 0, 0, 0],
                                  np.diag([0.01, 0.005, 0.1, 0.5])**2,
                                  size=number_particles)
prior = KernelParticleState(state_vector=StateVectors(samples.T),
                            weight=np.array([1/number_particles]*number_particles),
                            timestamp=timestamp)
samples = multivariate_normal.rvs([0, 0, 0, 0],
                                  np.diag([0.01, 0.005, 0.1, 0.5])**2,
                                  size=number_particles)
proposal = KernelParticleState(state_vector=StateVectors(samples.T),
                               weight=np.array([1/number_particles]*number_particles),
                               timestamp=timestamp)
prior_update = KernelParticleStateUpdate(state_vector=prior.state_vector,
                                         weight=prior.weight,
                                         proposal=prior.state_vector,
                                         hypothesis=None,
                                         timestamp=timestamp)


@pytest.mark.parametrize(
    "predictor_class, kernel, transition_model, prior_state, proposal_state",
    [
        (   # Standard Adaptive Kernel Kalman Predictor
            AdaptiveKernelKalmanPredictor,
            QuadraticKernel(),
            CombinedLinearGaussianTransitionModel(
                [ConstantVelocity(noise_diff_coeff=0.1), ConstantVelocity(noise_diff_coeff=0.1)]),
            prior,
            None
        ),
        (   # Standard Adaptive Kernel Kalman Predictor with proposal
            AdaptiveKernelKalmanPredictor,
            QuadraticKernel(),
            CombinedLinearGaussianTransitionModel(
                [ConstantVelocity(noise_diff_coeff=0.1), ConstantVelocity(noise_diff_coeff=0.1)]),
            prior,
            proposal
        ),
        (   # Standard Adaptive Kernel Kalman Predictor without kernel
            AdaptiveKernelKalmanPredictor,
            None,
            CombinedLinearGaussianTransitionModel(
                [ConstantVelocity(noise_diff_coeff=0.1), ConstantVelocity(noise_diff_coeff=0.1)]),
            prior,
            proposal
        ),
        (   # Standard Adaptive Kernel Kalman Predictor with KernelStateUpdate and no proposal
            AdaptiveKernelKalmanPredictor,
            None,
            CombinedLinearGaussianTransitionModel(
                [ConstantVelocity(noise_diff_coeff=0.1), ConstantVelocity(noise_diff_coeff=0.1)]),
            prior_update,
            None
        ),
        (   # Standard AKK Predictor with KernelStateUpdate, no proposal and kernel
            AdaptiveKernelKalmanPredictor,
            QuarticKernel(),
            CombinedLinearGaussianTransitionModel(
                [ConstantVelocity(noise_diff_coeff=0.1), ConstantVelocity(noise_diff_coeff=0.1)]),
            prior_update,
            None
        ),
    ],
    ids=["standard", "with_proposal", "no_kernel", "prior_update", "kernel"
         ]
)
def test_kernel(predictor_class, kernel, transition_model, prior_state, proposal_state):

    time_diff = datetime.timedelta(seconds=2)
    new_timestamp = prior.timestamp + time_diff
    predictor = predictor_class(transition_model=transition_model, kernel=kernel)
    prediction = predictor.predict(prior=prior_state, proposal=proposal_state,
                                   timestamp=new_timestamp, noise=False)
    if kernel is None:
        kernel = QuadraticKernel()
    if proposal_state is None:
        if isinstance(prior, KernelParticleStateUpdate):
            proposal_state = State(state_vector=prior.proposal)
        else:
            proposal_state = copy.deepcopy(prior)
    new_state_vector = transition_model.function(
        proposal_state,
        noise=False,
        time_interval=time_diff)
    k_tilde_tilde = kernel(proposal_state)
    k_tilde_nontilde = kernel(proposal_state, prior)

    kernel_t = np.linalg.pinv(
        k_tilde_tilde + predictor.lambda_predictor * np.identity(len(prior))) @ k_tilde_nontilde
    prediction_weights = kernel_t @ prior.weight
    v = (np.linalg.pinv(k_tilde_tilde + predictor.lambda_predictor * np.identity(
        len(prior))) @ k_tilde_tilde - np.identity(len(prior))) @ \
        (np.linalg.inv(k_tilde_tilde + predictor.lambda_predictor * np.identity(
            len(prior))) @ k_tilde_tilde - np.identity(len(prior))).T / len(prior)

    prediction_covariance = kernel_t @ prior.kernel_covar @ kernel_t.T + v
    prior_state.timestamp = timestamp

    assert hasattr(prediction, 'transition_model')
    assert prediction.timestamp == new_timestamp
    assert np.array_equal(prediction_weights, prediction.weight)
    assert np.allclose(prediction_covariance, prediction.kernel_covar)
    assert np.array_equal(new_state_vector, prediction.state_vector)
