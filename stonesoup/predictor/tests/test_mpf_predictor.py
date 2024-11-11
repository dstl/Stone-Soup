from datetime import timezone, datetime, timedelta

import numpy as np
import pytest

from stonesoup.models.base_driver import NoiseCase
from stonesoup.models.driver import AlphaStableNSMDriver, GaussianDriver
from stonesoup.models.transition.levy_linear import (
    CombinedLinearLevyTransitionModel,
    LevyLangevin,
    LevyRandomWalk,
)
from stonesoup.predictor.particle import MarginalisedParticlePredictor
from stonesoup.types.array import CovarianceMatrices, StateVectors
from stonesoup.types.numeric import Probability
from stonesoup.types.prediction import (
    MarginalisedParticleState,
    MarginalisedParticleStatePrediction,
)


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
    return CombinedLinearLevyTransitionModel(
        model_list=[transition_model_x, transition_model_y]
    )


@pytest.fixture(scope="function")
def gaussian_random_walk_model_1d(seed):
    mu_W = 1
    sigma_W2 = 25
    driver = GaussianDriver(mu_W=mu_W, sigma_W2=sigma_W2, seed=seed)
    transition_model = LevyRandomWalk(driver=driver)
    return CombinedLinearLevyTransitionModel(model_list=[transition_model])


@pytest.mark.parametrize(
    "model_type, prior, expected_state, expected_covariance",
    [
        (
            "alpha_stable_langevin_model_2d",
            MarginalisedParticleState(
                state_vector=StateVectors(
                    [
                        [-0.62665397, -0.88888401, 0.26237158, 0.77061426, -0.22988814],
                        [-0.04712397, 1.50397703, 0.67290205, 0.83930833, 2.23985039],
                        [1.89306702, 1.4276931, 0.76814597, 0.53354089, 0.60444506],
                        [1.36980183, 0.10191372, -0.03253117, 1.40181662, -0.82502972],
                    ]
                ),
                covariance=np.array(
                    [
                        [
                            [100.0, 100.0, 100.0, 100.0, 100.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [100.0, 100.0, 100.0, 100.0, 100.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [100.0, 100.0, 100.0, 100.0, 100.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [100.0, 100.0, 100.0, 100.0, 100.0],
                        ],
                    ]
                ),
                timestamp=None,
                weight=np.array(
                    [
                        Probability(0.2),
                        Probability(0.2),
                        Probability(0.2),
                        Probability(0.2),
                        Probability(0.2),
                    ],
                    dtype=object,
                ),
                parent=None,
                particle_list=None,
                fixed_covar=None,
            ),
            StateVectors(
                [
                    [-2.86082137, -0.60835831, 10.62216111, 33.9105639, 0.21075109],
                    [-2.58249062, -1.17297794, 11.94039787, 67.75687784, -1.38847372],
                    [0.94312498, 0.43746399, 10.48856876, 34.1833178, -1.73275825],
                    [-1.42240989, -2.32089029, 11.362838, 68.21742068, -3.89778532],
                ]
            ),
            CovarianceMatrices(
                [
                    [
                        [
                            1.91274417e02,
                            2.08692049e02,
                            3.35442281e03,
                            2.92355908e04,
                            2.05968615e02,
                        ],
                        [
                            9.61730879e01,
                            1.07339161e02,
                            4.09684328e03,
                            6.02232891e04,
                            1.14768907e02,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                    ],
                    [
                        [
                            9.61730879e01,
                            1.07339161e02,
                            4.09684328e03,
                            6.02232891e04,
                            1.14768907e02,
                        ],
                        [
                            1.43071683e02,
                            1.35040268e02,
                            5.18703675e03,
                            1.24597076e05,
                            1.39323271e02,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                    ],
                    [
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            1.91274417e02,
                            2.08692049e02,
                            3.35442281e03,
                            2.92355908e04,
                            2.05968615e02,
                        ],
                        [
                            9.61730879e01,
                            1.07339161e02,
                            4.09684328e03,
                            6.02232891e04,
                            1.14768907e02,
                        ],
                    ],
                    [
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            9.61730879e01,
                            1.07339161e02,
                            4.09684328e03,
                            6.02232891e04,
                            1.14768907e02,
                        ],
                        [
                            1.43071683e02,
                            1.35040268e02,
                            5.18703675e03,
                            1.24597076e05,
                            1.39323271e02,
                        ],
                    ],
                ]
            ),
        ),
        (
            "gaussian_random_walk_model_1d",
            MarginalisedParticleState(
                state_vector=StateVectors(
                    StateVectors(
                        [[0.85778795, 0.81499289, -1.55124413, -1.34084576, 0.3580332]]
                    )
                ),
                covariance=np.array([[[100.0, 100.0, 100.0, 100.0, 100.0]]]),
                timestamp=None,
                weight=np.array(
                    [
                        Probability(0.2),
                        Probability(0.2),
                        Probability(0.2),
                        Probability(0.2),
                        Probability(0.2),
                    ],
                    dtype=object,
                ),
                parent=None,
                particle_list=None,
                fixed_covar=None,
            ),
            StateVectors([[1.85778795, 1.81499289, -0.55124413, -0.34084576, 1.3580332]]),
            CovarianceMatrices([[[125.0, 125.0, 125.0, 125.0, 125.0]]]),
        ),
    ],
)
def test_marginalised_particle_filter_predict(
    request: pytest.FixtureRequest,
    model_type: str,
    prior: MarginalisedParticleState,
    expected_state: StateVectors,
    expected_covariance: CovarianceMatrices,
):
    start_time = datetime.now(tz=timezone.UTC).replace(microsecond=0)
    prior.timestamp = start_time
    model = request.getfixturevalue(model_type)
    predictor = MarginalisedParticlePredictor(transition_model=model)
    prediction = predictor.predict(prior, timestamp=start_time + timedelta(seconds=1))
    assert isinstance(prediction, MarginalisedParticleStatePrediction)
    assert prediction.parent == prior
    assert prediction.particle_list is None
    assert prediction.particle_list == prior.particle_list
    assert prediction.fixed_covar is None
    assert prediction.fixed_covar == prior.fixed_covar
    assert np.all(prediction.weight == prior.weight)
    assert np.allclose(prediction.state_vector, expected_state)
    assert np.allclose(prediction.covariance, expected_covariance)
