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


@pytest.fixture
def mock_jump_times() -> np.ndarray:
    return np.array([[1, 1], [2, 2], [3, 3]])


@pytest.fixture
def mock_jump_sizes() -> np.ndarray:
    return np.array([[1.0, 1.0], [0.8, 0.8], [0.6, 0.6]])


@pytest.fixture(scope="session")
def monkey_session():
    with pytest.MonkeyPatch.context() as mp:
        yield mp
