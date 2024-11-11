from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from stonesoup.models.base_driver import NoiseCase
from stonesoup.models.driver import AlphaStableNSMDriver, GaussianDriver
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.levy_linear import (
    CombinedLinearLevyTransitionModel, LevyLangevin, LevyRandomWalk)
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.types.array import (CovarianceMatrices, CovarianceMatrix,
                                   StateVector, StateVectors)
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.prediction import (MarginalisedParticleState,
                                        MarginalisedParticleStatePrediction)
from stonesoup.types.update import MarginalisedParticleStateUpdate
from stonesoup.updater.particle import MarginalisedParticleUpdater


@pytest.fixture
def seed() -> int:
    return 0


@pytest.fixture(scope="function")
def alpha_stable_langevin_model_2d(seed):
    mu_W = 1
    sigma_W2 = 25
    alpha = 1.4
    c = 10
    noise_case = NoiseCase.GAUSSIAN_APPROX
    driver = AlphaStableNSMDriver(
        mu_W=mu_W, sigma_W2=sigma_W2, seed=seed, alpha=alpha, c=c, noise_case=noise_case
    )
    theta = 0.2
    transition_model_x = LevyLangevin(damping_coeff=theta, driver=driver)
    transition_model_y = LevyLangevin(damping_coeff=theta, driver=driver)
    return CombinedLinearLevyTransitionModel(model_list=[transition_model_x, transition_model_y])


@pytest.fixture(scope="function")
def gaussian_random_walk_model_1d(seed):
    mu_W = 1
    sigma_W2 = 25
    driver = GaussianDriver(mu_W=mu_W, sigma_W2=sigma_W2, seed=seed)
    transition_model = LevyRandomWalk(driver=driver)
    return CombinedLinearLevyTransitionModel(model_list=[transition_model])


@pytest.mark.parametrize(
    "model_type, hypothesis, expected_state, expected_covariance",
    [
        (
            "alpha_stable_langevin_model_2d",
            SingleHypothesis(
                prediction=MarginalisedParticleStatePrediction(
                    state_vector=StateVectors(
                        [
                            [0.85798578, 0.12636091, -0.0676282, 0.03570828, -0.42422919],
                            [-3.02711197, -2.48877242, -0.72049429, -2.42812059, -2.70124928],
                            [-0.40956703, -1.00294167, 0.70939368, -1.58729029, -1.17903268],
                            [-3.01473378, -4.1861108, 0.13154929, -3.38822607, -2.68388992],
                        ]
                    ),
                    covariance=CovarianceMatrices(
                        [
                            [
                                [
                                    192.29655805,
                                    191.51680052,
                                    199.54518545,
                                    190.1203207,
                                    192.02971017,
                                ],
                                [90.9730478, 89.65483754, 110.50326036, 90.29556339, 92.33711585],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ],
                            [
                                [90.9730478, 89.65483754, 110.50326036, 90.29556339, 92.33711585],
                                [
                                    98.55203117,
                                    98.78987551,
                                    152.07003477,
                                    106.72740959,
                                    106.23263394,
                                ],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ],
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [
                                    192.29655805,
                                    191.51680052,
                                    199.54518545,
                                    190.1203207,
                                    192.02971017,
                                ],
                                [90.9730478, 89.65483754, 110.50326036, 90.29556339, 92.33711585],
                            ],
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [90.9730478, 89.65483754, 110.50326036, 90.29556339, 92.33711585],
                                [
                                    98.55203117,
                                    98.78987551,
                                    152.07003477,
                                    106.72740959,
                                    106.23263394,
                                ],
                            ],
                        ]
                    ),
                    weight=np.array(
                        [
                            Probability(0.2),
                            Probability(0.2),
                            Probability(0.2),
                            Probability(0.2),
                            Probability(0.2),
                        ],
                    ),
                ),
                measurement=Detection(
                    state_vector=StateVector([[-5.43254207], [-2.04992048]]),
                    measurement_model=LinearGaussian(
                        ndim_state=4,
                        mapping=(0, 2),
                        noise_covar=CovarianceMatrix([[25.0, 0.0], [0.0, 25.0]]),
                    ),
                ),
            ),
            StateVectors(
                [
                    [-4.70881594, -4.83523313, -4.79068622, -4.79705464, -4.85562649],
                    [-5.66069483, -3.3606777, -4.79059169, -4.72338839, -4.83207812],
                    [-1.86119758, -1.74270902, -1.92903162, -1.99615636, -1.94960152],
                    [-3.70148167, -1.22636542, -4.61964166, -3.58241253, -3.05441646],
                ]
            ),
            CovarianceMatrices(
                [
                    [
                        [22.12374643, 22.11338797, 22.21659585, 22.09464918, 22.12020995],
                        [10.46646213, 10.35194929, 12.30300932, 10.49361156, 10.6364603],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [10.46646213, 10.35194929, 12.30300932, 10.49361156, 10.6364603],
                        [60.4653928, 61.66578225, 97.68912909, 68.82634687, 66.94703126],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [22.12374643, 22.11338797, 22.21659585, 22.09464918, 22.12020995],
                        [10.46646213, 10.35194929, 12.30300932, 10.49361156, 10.6364603],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [10.46646213, 10.35194929, 12.30300932, 10.49361156, 10.6364603],
                        [60.4653928, 61.66578225, 97.68912909, 68.82634687, 66.94703126],
                    ],
                ]
            ),
        ),
        (
            "gaussian_random_walk_model_1d",
            SingleHypothesis(
                prediction=MarginalisedParticleStatePrediction(
                    state_vector=StateVectors(
                        [[1.48184682, 1.96629969, 1.58480781, 1.07186231, 1.81230203]]
                    ),
                    covariance=CovarianceMatrices([[[125.0, 125.0, 125.0, 125.0, 125.0]]]),
                    weight=np.array(
                        [
                            Probability(0.2),
                            Probability(0.2),
                            Probability(0.2),
                            Probability(0.2),
                            Probability(0.2),
                        ],
                    ),
                ),
                measurement=Detection(
                    state_vector=StateVector([[0.35420189]]),
                    measurement_model=LinearGaussian(
                        ndim_state=1,
                        mapping=(0,),
                        noise_covar=CovarianceMatrix([[25.0]]),
                    ),
                ),
            ),
            StateVectors([[0.62288486, 0.59721858, 0.55930288, 0.54214271, 0.47381196]]),
            CovarianceMatrices(
                [[[20.83333333, 20.83333333, 20.83333333, 20.83333333, 20.83333333]]]
            ),
        ),
    ],
)
def test_marginalised_particle_filter_update(
    request: pytest.FixtureRequest,
    model_type: str,
    hypothesis: SingleHypothesis,
    expected_state: StateVectors,
    expected_covariance: CovarianceMatrices,
):
    start_time = datetime.now(tz=UTC).replace(microsecond=0)
    hypothesis.prediction.timestamp = start_time
    hypothesis.measurement.timestamp = start_time

    model = request.getfixturevalue(model_type)
    resampler = SystematicResampler()
    updater = MarginalisedParticleUpdater(model, resampler)
    posterior = updater.update(hypothesis)
    assert isinstance(posterior, MarginalisedParticleStateUpdate)
    assert posterior.hypothesis == hypothesis
    assert np.allclose(posterior.state_vector, expected_state)
    assert np.allclose(posterior.covariance, expected_covariance)
