import numpy as np
import pytest


@pytest.fixture
def seed() -> int:
    return 0


@pytest.fixture
def mu_W() -> float:
    return 1.0


@pytest.fixture
def sigma_W2() -> float:
    return 1


@pytest.fixture
def expected_jumps_per_sec() -> int:
    return 2


@pytest.fixture
def mock_e_ft_func() -> np.ndarray:
    return lambda dt: np.array([[1.0], [2.0]])
