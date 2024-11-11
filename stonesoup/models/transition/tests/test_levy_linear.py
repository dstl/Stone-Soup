import pytest
from stonesoup.models.transition.levy_linear import (
    LevyLangevin,
    LevyRandomWalk,
    LevyConstantAcceleration,
    LevyConstantVelocity,
)
from stonesoup.models.driver import GaussianDriver, AlphaStableNSMDriver
from stonesoup.models.base_driver import NoiseCase, ConditionallyGaussianDriver
import numpy as np
from datetime import datetime, timedelta, UTC
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.array import StateVector
from typing import Union
from stonesoup.models.base import LevyModel


@pytest.fixture
def seed() -> int:
    return 0


@pytest.fixture(scope="function")
def gaussian_driver(seed):
    mu_W = 1
    sigma_W2 = 1
    return GaussianDriver(mu_W=mu_W, sigma_W2=sigma_W2, seed=seed)


@pytest.fixture
def conditionally_gaussian_driver(seed):
    mu_W = 1
    sigma_W2 = 1
    alpha = 1.5
    c = 10
    noise_case = NoiseCase.GAUSSIAN_APPROX
    return AlphaStableNSMDriver(
        mu_W=mu_W, sigma_W2=sigma_W2, seed=seed, alpha=alpha, c=c, noise_case=noise_case
    )


def compare_transition(model: LevyModel, expected_state_vectors: list[StateVector]):
    start_time = datetime.now(tz=UTC).replace(microsecond=0)
    timesteps = [start_time]
    truth = GroundTruthPath([GroundTruthState(expected_state_vectors[0], timestamp=timesteps[0])])

    num_steps = 3
    for k in range(1, num_steps + 1):
        timesteps.append(
            start_time + timedelta(seconds=k)
        )  # add next timestep to list of timesteps
        truth.append(
            GroundTruthState(
                model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
                timestamp=timesteps[k],
            )
        )

    for result, expected in zip(truth, expected_state_vectors):
        assert np.allclose(result.state_vector, expected)


@pytest.mark.parametrize(
    "driver_type, expected_state_vectors",
    [
        (
            "gaussian_driver",
            [
                StateVector([[0]]),
                StateVector([[1.12573022]]),
                StateVector([[1.99362536]]),
                StateVector([[3.63404801]]),
            ],
        ),
        (
            "conditionally_gaussian_driver",
            [
                StateVector([[0]]),
                StateVector([[-3.60737453]]),
                StateVector([[-7.18177547]]),
                StateVector([[-5.37123095]]),
            ],
        ),
    ],
)
def test_levy_random_walk(
    request: pytest.FixtureRequest,
    driver_type: str,
    expected_state_vectors: np.ndarray,
):
    driver = request.getfixturevalue(driver_type)
    transition_model = LevyRandomWalk(driver=driver)
    compare_transition(transition_model, expected_state_vectors)


@pytest.mark.parametrize(
    "driver_type, expected_state_vectors",
    [
        (
            "gaussian_driver",
            [
                StateVector([[0], [1]]),
                StateVector([[1.31573952], [1.61112188]]),
                StateVector([[2.94435263], [1.6449766]]),
                StateVector([[5.15437707], [2.73864109]]),
            ],
        ),
        (
            "conditionally_gaussian_driver",
            [
                StateVector([[0], [1]]),
                StateVector([[2.23700493], [2.13560881]]),
                StateVector([[2.52813741], [-4.1708489]]),
                StateVector([[-0.4501127], [-1.11469671]]),
            ],
        ),
    ],
)
def test_levy_langevin(
    request: pytest.FixtureRequest,
    driver_type: Union[GaussianDriver, ConditionallyGaussianDriver],
    expected_state_vectors: np.ndarray,
):
    driver = request.getfixturevalue(driver_type)
    theta = 0.2
    transition_model = LevyLangevin(damping_coeff=theta, driver=driver)
    compare_transition(transition_model, expected_state_vectors)


@pytest.mark.parametrize(
    "driver_type, expected_state_vectors",
    [
        (
            "gaussian_driver",
            [
                StateVector([[0], [1]]),
                StateVector([[1.43713489], [1.87426978]]),
                StateVector([[3.49119334], [2.23384713]]),
                StateVector([[6.49287515], [3.7695165]]),
            ],
        ),
        (
            "conditionally_gaussian_driver",
            [
                StateVector([[0], [1]]),
                StateVector([[2.47577405], [2.5539618]]),
                StateVector([[3.21401715], [-3.7669634]]),
                StateVector([[0.25101422], [-1.38010576]]),
            ],
        ),
    ],
)
def test_levy_constant_velocity(
    request: pytest.FixtureRequest,
    driver_type: Union[GaussianDriver, ConditionallyGaussianDriver],
    expected_state_vectors: np.ndarray,
):
    driver = request.getfixturevalue(driver_type)
    transition_model = LevyConstantVelocity(driver=driver)
    compare_transition(transition_model, expected_state_vectors)


@pytest.mark.parametrize(
    "driver_type, expected_state_vectors",
    [
        (
            "gaussian_driver",
            [
                StateVector([[0], [0], [1]]),
                StateVector([[0.64571163], [1.43713489], [1.87426978]]),
                StateVector([[3.16916472], [3.75895461], [2.76936966]]),
                StateVector([[8.26213749], [6.37632425], [2.46536962]]),
            ],
        ),
        (
            "conditionally_gaussian_driver",
            [
                StateVector([[0], [0], [1]]),
                StateVector([[1.26124374], [2.47364757], [2.53440017]]),
                StateVector([[5.30179643], [5.30065052], [1.20670201]]),
                StateVector([[10.28136979], [4.36069052], [0.81064657]]),
            ],
        ),
    ],
)
def test_levy_constant_acceleration(
    request: pytest.FixtureRequest,
    driver_type: Union[GaussianDriver, ConditionallyGaussianDriver],
    expected_state_vectors: np.ndarray,
):
    driver = request.getfixturevalue(driver_type)
    transition_model = LevyConstantAcceleration(driver=driver)
    compare_transition(transition_model, expected_state_vectors)
