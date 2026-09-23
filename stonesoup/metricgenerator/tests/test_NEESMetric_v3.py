import datetime

import numpy as np
import pytest

from stonesoup.metricgenerator.NEESMetric_v3 import NEESMetric  # Import my KLMetric class
from stonesoup.metricgenerator.manager import MultiManager
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.metric import SingleTimeMetric
from stonesoup.types.particle import Particle
from stonesoup.types.state import GaussianState, ParticleState
from stonesoup.types.state import State
from stonesoup.types.track import Track


# A. Input Validation Tests

def test_neesmetric_no_measured_states():
    """Test NEESMetric with no measured states provided."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    truth_state = GaussianState(
        state_vector=StateVector([[1], [2]]),
        covar=np.eye(2),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([], [], [truth_state], [0])

def test_neesmetric_no_truth_states():
    """Test NEESMetric with no truth states provided."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[1], [2]]),
        covar=np.eye(2),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [], [])

def test_neesmetric_no_states():
    """Test NEESMetric with no measured or truth states provided."""
    generator = NEESMetric()

    metric = generator.compute_over_time_v2([], [], [], [])
    assert isinstance(metric, SingleTimeMetric)
    assert metric.value['ANEES'] == 0.0
    assert metric.value['Average False Tracks'] == 0.0
    assert metric.value['Average Missed Targets'] == 0.0

def test_neesmetric_empty_inputs():
    """Test NEESMetric with empty measured and truth states."""
    generator = NEESMetric()
    measured_states = []
    truth_states = []

    metric = generator.compute_over_time_v2(measured_states, [], truth_states, [])
    assert metric.value['ANEES'] == 0.0
    assert metric.value['Average False Tracks'] == 0.0
    assert metric.value['Average Missed Targets'] == 0.0

def test_neesmetric_null_inputs():
    """Test NEESMetric with None as inputs."""
    generator = NEESMetric()
    measured_states = None
    truth_states = None

    with pytest.raises(TypeError):
        generator.compute_over_time_v2(measured_states, None, truth_states, None)

def test_neesmetric_incorrect_data_types():
    """Test NEESMetric with incorrect data types as inputs."""
    generator = NEESMetric()
    measured_states = "invalid_input"
    truth_states = "invalid_input"

    with pytest.raises(AttributeError):
        generator.compute_over_time_v2(measured_states, None, truth_states, None)

def test_neesmetric_none_timestamps():
    """Test NEESMetric with states missing timestamps."""
    generator = NEESMetric()

    measured_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=np.eye(1),
        timestamp=None
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=np.eye(1),
        timestamp=None
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

# B. Dimension Consistency Tests

def test_neesmetric_mismatched_dimensions():
    """Test NEESMetric with mismatched dimensions between measured and truth states."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[1], [2], [3]]),
        covar=np.eye(3),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1], [2]]),
        covar=np.eye(2),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_mismatched_state_vector_dimensions():
    """Test NEESMetric with mismatched state vector dimensions."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[0], [1]]),  # 2D state vector
        covar=np.eye(2),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),  # 1D state vector
        covar=np.eye(1),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_variable_dimensions():
    """Test NEESMetric with variable dimensions across state pairs."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_states = [
        GaussianState(
            state_vector=StateVector(np.random.rand(2, 1)),
            covar=np.eye(2),
            timestamp=time
        ),
        GaussianState(
            state_vector=StateVector(np.random.rand(3, 1)),
            covar=np.eye(3),
            timestamp=time
        )
    ]
    truth_states = [
        GaussianState(
            state_vector=StateVector(np.random.rand(2, 1)),
            covar=np.eye(2),
            timestamp=time
        ),
        GaussianState(
            state_vector=StateVector(np.random.rand(3, 1)),
            covar=np.eye(3),
            timestamp=time
        )
    ]

    # The NEESMetric should be able to handle states of different dimensions separately
    # If not, we can expect a ValueError
    with pytest.raises(ValueError):
        generator.compute_over_time_v2(measured_states, [0, 1], truth_states, [0, 1])


# C. Numerical Stability Tests

def test_neesmetric_singular_covariance():
    """Test NEESMetric with singular covariance matrix."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    singular_covariance = np.array([[1, 0], [0, 0]])  # Second row is zero
    measured_state = GaussianState(
        state_vector=StateVector([[1], [2]]),
        covar=singular_covariance,
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1], [2]]),
        covar=np.eye(2),
        timestamp=time
    )

    with pytest.raises(np.linalg.LinAlgError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_large_values():
    """Test NEESMetric with very large values in state vectors and covariance matrices."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    large_value = 1e100  # Use a value that's large but within safe computational limits
    measured_state = GaussianState(
        state_vector=StateVector([[large_value], [large_value]]),
        covar=np.eye(2) * large_value,
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[large_value], [large_value]]),
        covar=np.eye(2) * large_value,
        timestamp=time
    )

    # NEES should be zero since the error is zero
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

    assert metric.value['ANEES'] == 0.0

def test_neesmetric_small_values():
    """Test NEESMetric with very small values in state vectors and covariance matrices."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    small_value = 1e-308  # Near the minimum positive normalized float64
    measured_state = GaussianState(
        state_vector=StateVector([[small_value], [small_value]]),
        covar=np.eye(2) * small_value,
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0], [0]]),
        covar=np.eye(2) * small_value,
        timestamp=time
    )

    # Compute the NEES metric
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

    # Expected NEES calculation
    error = truth_state.state_vector - measured_state.state_vector
    inv_covar = np.linalg.inv(measured_state.covar)
    nees_value = (error.T @ inv_covar @ error).item()
    DoF = measured_state.state_vector.shape[0]
    expected_nees = nees_value / DoF

    # Check if the NEES value is as expected
    assert metric.value['ANEES'] == pytest.approx(expected_nees)

def test_neesmetric_extremely_large_numbers():
    """Test NEESMetric with extremely large numbers."""
    generator = NEESMetric()
    time = datetime.datetime.now()
    large_value = 1e308  # Close to the maximum for a float64

    measured_state = GaussianState(
        state_vector=StateVector([[large_value]]),
        covar=np.eye(1),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[large_value + 1]]),
        covar=np.eye(1),
        timestamp=time
    )

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    anees = metric.value['ANEES']
    assert np.isfinite(anees), "ANEES should be finite with large numbers"

def test_neesmetric_extremely_small_numbers():
    """Test NEESMetric with extremely small numbers."""
    generator = NEESMetric()
    time = datetime.datetime.now()
    small_value = 1e-308  # Close to the minimum positive normal float

    measured_state = GaussianState(
        state_vector=StateVector([[small_value]]),
        covar=np.eye(1),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[small_value + 1e-309]]),
        covar=np.eye(1),
        timestamp=time
    )

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    anees = metric.value['ANEES']
    assert np.isfinite(anees), "ANEES should be finite with small numbers"

def test_neesmetric_zero_covariance():
    """Test NEESMetric with zero covariance matrix."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    zero_covariance = np.zeros((1, 1))
    measured_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=zero_covariance,
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=zero_covariance,
        timestamp=time
    )

    with pytest.raises(np.linalg.LinAlgError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_high_dimensional_states():
    """Test NEESMetric with high-dimensional state vectors."""
    generator = NEESMetric()
    time = datetime.datetime.now()
    dimension = 1000

    measured_state = GaussianState(
        state_vector=StateVector(np.random.rand(dimension, 1)),
        covar=np.eye(dimension),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector(np.random.rand(dimension, 1)),
        covar=np.eye(dimension),
        timestamp=time
    )

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    anees = metric.value['ANEES']
    assert anees >= 0, "ANEES should be non-negative"

# D. Timestamp Handling Tests

def test_neesmetric_unordered_timestamps():
    """Test NEESMetric with unordered timestamps."""
    generator = NEESMetric()
    time1 = datetime.datetime.now()
    time2 = time1 - datetime.timedelta(seconds=10)  # Earlier timestamp

    measured_states = [
        GaussianState(
            state_vector=StateVector([[1]]),
            covar=np.eye(1),
            timestamp=time1
        ),
        GaussianState(
            state_vector=StateVector([[2]]),
            covar=np.eye(1),
            timestamp=time2
        )
    ]
    truth_states = [
        GaussianState(
            state_vector=StateVector([[1]]),
            covar=np.eye(1),
            timestamp=time1
        ),
        GaussianState(
            state_vector=StateVector([[2]]),
            covar=np.eye(1),
            timestamp=time2
        )
    ]

    metric = generator.compute_over_time_v2(measured_states, [0, 1], truth_states, [0, 1])
    assert 'ANEES' in metric.value

def test_neesmetric_mismatched_timestamps():
    """Test NEESMetric with mismatched timestamps between measured and truth states."""
    generator = NEESMetric()
    time1 = datetime.datetime.now()
    time2 = time1 + datetime.timedelta(seconds=10)  # Different timestamp

    measured_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=np.eye(1),
        timestamp=time1
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=np.eye(1),
        timestamp=time2
    )

    # Since the timestamps are different, they should be processed separately
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    # ANEES should be zero since there are no matched states at the same timestamp
    assert metric.value['ANEES'] == 0.0

def test_neesmetric_missing_timestamps():
    """Test NEESMetric with states missing timestamps."""
    generator = NEESMetric()

    measured_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1)
        # Missing timestamp
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=datetime.datetime.now()
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_out_of_order_timestamps():
    """Test NEESMetric with out-of-order timestamps."""
    generator = NEESMetric()
    time_now = datetime.datetime.now()
    time_past = time_now - datetime.timedelta(seconds=10)

    measured_states = [
        GaussianState(state_vector=StateVector([[0]]), covar=np.eye(1), timestamp=time_now),
        GaussianState(state_vector=StateVector([[1]]), covar=np.eye(1), timestamp=time_past),
    ]
    truth_states = [
        GaussianState(state_vector=StateVector([[0]]), covar=np.eye(1), timestamp=time_now),
        GaussianState(state_vector=StateVector([[1]]), covar=np.eye(1), timestamp=time_past),
    ]

    metric = generator.compute_over_time_v2(measured_states, [0, 1], truth_states, [0, 1])
    assert metric.value['ANEES'] >= 0.0

def test_neesmetric_varying_states_over_time():
    """Test NEESMetric with varying number of states over time."""
    generator = NEESMetric()
    time1 = datetime.datetime.now()
    time2 = time1 + datetime.timedelta(seconds=1)

    measured_states = [
        GaussianState(state_vector=StateVector([[0]]), covar=np.eye(1), timestamp=time1),
        GaussianState(state_vector=StateVector([[1]]), covar=np.eye(1), timestamp=time2),
        GaussianState(state_vector=StateVector([[2]]), covar=np.eye(1), timestamp=time2),
    ]
    truth_states = [
        GaussianState(state_vector=StateVector([[0]]), covar=np.eye(1), timestamp=time1),
        GaussianState(state_vector=StateVector([[1]]), covar=np.eye(1), timestamp=time2),
    ]

    metric = generator.compute_over_time_v2(measured_states, [0, 1, 2], truth_states, [0, 1])
    assert metric.value['ANEES'] >= 0.0
    assert metric.value['Average False Tracks'] > 0 or metric.value['Average Missed Targets'] > 0

def test_neesmetric_invalid_timestamps():
    """Test NEESMetric with invalid timestamps."""
    generator = NEESMetric()
    measured_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp="invalid_timestamp"  # Invalid timestamp
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=datetime.datetime.now()
    )

    with pytest.raises(TypeError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

# E. Covariance Matrix Tests

def test_neesmetric_non_positive_definite_covariance():
    """Test NEESMetric with non-positive definite covariance matrix."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    non_pos_def_covar = np.array([[1, 2], [2, 1]])  # Not positive definite
    measured_state = GaussianState(
        state_vector=StateVector([[0], [0]]),
        covar=non_pos_def_covar,
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0], [0]]),
        covar=np.eye(2),
        timestamp=time
    )

    with pytest.raises(np.linalg.LinAlgError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_covariance_with_off_diagonal_terms():
    """Test NEESMetric with covariance matrices that have off-diagonal terms."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    covariance = np.array([[2, 0.5], [0.5, 1]])
    measured_state = GaussianState(
        state_vector=StateVector([[1], [2]]),
        covar=covariance,
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1.1], [2.1]]),
        covar=np.eye(2),
        timestamp=time
    )

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    assert 'ANEES' in metric.value

def test_neesmetric_covariance_with_nans():
    """Test NEESMetric with covariance matrices containing NaNs."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    covar_with_nan = np.array([[1, np.nan], [np.nan, 1]])

    measured_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=covar_with_nan,
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=np.eye(2),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_invalid_covariance_shape():
    """Test NEESMetric with invalid covariance matrix shape."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    covariance = np.array([[1, 0]])  # Invalid shape
    with pytest.raises(ValueError):
        measured_state = GaussianState(
            state_vector=StateVector([[1], [2]]),
            covar=covariance,
            timestamp=time
        )
        truth_state = GaussianState(
            state_vector=StateVector([[1], [2]]),
            covar=np.eye(2),
            timestamp=time
        )

        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

# F. State Type Tests

def test_neesmetric_invalid_state_type():
    """Test NEESMetric with invalid state types."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_state = State(
        state_vector=StateVector([[1]]),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=np.eye(1),
        timestamp=time
    )

    with pytest.raises(TypeError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_mixed_state_types():
    """Test NEESMetric with mixed state types."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_state = ParticleState(
        state_vector=None,
        particle_list=[Particle(state_vector=StateVector([[0]]), weight=1.0)],
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=time
    )

    with pytest.raises(TypeError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_inconsistent_state_types():
    """Test NEESMetric with inconsistent state types."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=np.eye(1),
        timestamp=time
    )
    truth_state = ParticleState(
        state_vector=None,
        particle_list=[Particle(state_vector=StateVector([[1]]), weight=1.0)],
        timestamp=time
    )

    with pytest.raises(TypeError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_non_gaussian_states():
    """Test NEESMetric with non-Gaussian states."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_state = State(
        state_vector=StateVector([[0]]),
        timestamp=time
    )
    truth_state = State(
        state_vector=StateVector([[0]]),
        timestamp=time
    )

    with pytest.raises(TypeError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

# G. Invalid Values Tests

def test_neesmetric_nan_in_state_vector():
    """Test NEESMetric with NaN in state vector."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[np.nan]]),
        covar=np.eye(1),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=np.eye(1),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_inf_in_state_vector():
    """Test NEESMetric with Inf in state vector."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[np.inf]]),
        covar=np.eye(1),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=np.eye(1),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_nan_in_covariance():
    """Test NEESMetric with NaN in covariance matrix."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    covariance = np.array([[np.nan]])
    measured_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=covariance,
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=np.eye(1),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_state_vectors_with_nans():
    """Test NEESMetric with state vectors containing NaNs."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[np.nan]]),
        covar=np.eye(1),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

def test_neesmetric_state_vectors_with_infs():
    """Test NEESMetric with state vectors containing Infs."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[np.inf]]),
        covar=np.eye(1),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

# H. Incorrect Usage of the Module

def test_neesmetric_invalid_confidence_threshold():
    """Test NEESMetric with invalid confidence threshold."""
    with pytest.raises(ValueError):
        NEESMetric(confidence_threshold=-1)  # Negative threshold is invalid

def test_neesmetric_method_call_order():
    """Test NEESMetric method call order."""
    generator = NEESMetric()

    # Attempt to call compute_nees_metric_v2 directly without proper inputs
    with pytest.raises(TypeError):
        generator.compute_nees_metric_v2()

# I. Additional Tests

def test_neesmetric_multiple_states_per_timestamp():
    """Test NEESMetric with multiple states per timestamp."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_states = [
        GaussianState(
            state_vector=StateVector([[i]]),
            covar=np.eye(1),
            timestamp=time
        ) for i in range(3)
    ]
    truth_states = [
        GaussianState(
            state_vector=StateVector([[i + 0.1]]),
            covar=np.eye(1),
            timestamp=time
        ) for i in range(3)
    ]

    metric = generator.compute_over_time_v2(measured_states, list(range(3)), truth_states, list(range(3)))
    assert 'ANEES' in metric.value

def test_neesmetric_large_state_vectors():
    """Test NEESMetric with large-dimensional state vectors."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    dim = 100
    measured_state = GaussianState(
        state_vector=StateVector(np.random.rand(dim, 1)),
        covar=np.eye(dim),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector(np.random.rand(dim, 1)),
        covar=np.eye(dim),
        timestamp=time
    )

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    assert 'ANEES' in metric.value

def test_neesmetric_different_number_of_states():
    """Test NEESMetric with different numbers of measured and truth states."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_states = [
        GaussianState(
            state_vector=StateVector([[1]]),
            covar=np.eye(1),
            timestamp=time
        )
    ]
    truth_states = [
        GaussianState(
            state_vector=StateVector([[1]]),
            covar=np.eye(1),
            timestamp=time
        ),
        GaussianState(
            state_vector=StateVector([[2]]),
            covar=np.eye(1),
            timestamp=time
        )
    ]

    metric = generator.compute_over_time_v2(measured_states, [0], truth_states, [0, 1])
    assert 'ANEES' in metric.value

def test_neesmetric_confidence_threshold():
    """Test NEESMetric with a specific confidence threshold."""
    generator = NEESMetric(confidence_threshold=0.5)
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1]]),
        covar=np.eye(1),
        timestamp=time
    )

    # The Mahalanobis distance is sqrt((1-0)^2 / 1) = 1, which is greater than the threshold 0.5
    # So the states should not be assigned (ANEES should be zero
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    assert metric.value['ANEES'] == 0.0

def test_neesmetric_exact_match():
    """Test NEESMetric with exact match between measured and truth states."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[1], [2]]),
        covar=np.eye(2),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1], [2]]),
        covar=np.eye(2),
        timestamp=time
    )

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    assert metric.value['ANEES'] == pytest.approx(0.0)

def test_neesmetric_high_error():
    """Test NEESMetric with high error between measured and truth states."""
    # Increase confidence_threshold to a value larger than the expected cost
    generator = NEESMetric(confidence_threshold=500)
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[0], [0]]),
        covar=np.eye(2),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[10], [10]]),
        covar=np.eye(2),
        timestamp=time
    )

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    # Compute expected NEES value
    error = truth_state.state_vector - measured_state.state_vector
    inv_covar = np.linalg.inv(measured_state.covar)
    nees_value = (error.T @ inv_covar @ error).item()
    DoF = measured_state.state_vector.shape[0]
    expected_nees = nees_value / DoF

    assert metric.value['ANEES'] == pytest.approx(expected_nees)


def test_neesmetric_multiple_timestamps():
    """Test NEESMetric over multiple timestamps."""
    generator = NEESMetric()
    time_start = datetime.datetime.now()

    measured_states = []
    truth_states = []
    for i in range(5):
        time = time_start + datetime.timedelta(seconds=i)
        measured_states.append(
            GaussianState(
                state_vector=StateVector([[i]]),
                covar=np.eye(1),
                timestamp=time
            )
        )
        truth_states.append(
            GaussianState(
                state_vector=StateVector([[i + 0.1]]),
                covar=np.eye(1),
                timestamp=time
            )
        )

    metric = generator.compute_over_time_v2(measured_states, list(range(5)), truth_states, list(range(5)))
    assert 'ANEES' in metric.value

# Additional Tests for Specific Methods

def test_compute_over_time_no_states():
    """Test compute_over_time_v2 with no states provided."""
    generator = NEESMetric()
    measured_states = []
    truth_states = []

    metric = generator.compute_over_time_v2(measured_states, [], truth_states, [])
    assert metric.value['ANEES'] == 0.0
    assert metric.value['Average False Tracks'] == 0.0
    assert metric.value['Average Missed Targets'] == 0.0

def test_compute_nees_metric_unassigned_states():
    """Test compute_nees_metric_v2 with unassigned states."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    # Measured state far from truth state
    measured_state = GaussianState(
        state_vector=StateVector([[1000]]),
        covar=np.eye(1),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=time
    )

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    assert metric.value['ANEES'] == 0.0  # No matched states
    assert metric.value['Average False Tracks'] == 1.0
    assert metric.value['Average Missed Targets'] == 1.0

def test_compute_cost_matrix_high_mahalanobis():
    """Test compute_cost_matrix_v2 with high Mahalanobis distance."""
    generator = NEESMetric(confidence_threshold=2)
    time = datetime.datetime.now()

    # Create states that are far apart
    measured_state = GaussianState(
        state_vector=StateVector([[1000]]),
        covar=np.eye(1),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=time
    )

    cost_matrix = generator.compute_cost_matrix_v2([measured_state], [truth_state], complete=False)
    assert cost_matrix[0, 0] == generator.confidence_threshold, "Cost should be set to confidence_threshold for high Mahalanobis distance"

# Tests Based on NEESMetric_v2

def test_neesmetric_compute_metric_over_time():
    """Test NEESMetric compute_metric over time with known inputs."""
    generator = NEESMetric()
    time_start = datetime.datetime.now()

    # Create tracks with GaussianStates
    tracks = {Track(states=[
        GaussianState(state_vector=StateVector([[i + 0.5]]), covar=np.array([[1]]), timestamp=time_start),
        GaussianState(state_vector=StateVector([[i + 1.5]]), covar=np.array([[1]]),
                      timestamp=time_start + datetime.timedelta(seconds=1))
    ]) for i in range(5)}

    # Create ground truths
    truths = {GroundTruthPath(states=[
        GaussianState(state_vector=StateVector([[i]]), covar=np.array([[1]]), timestamp=time_start),
        GaussianState(state_vector=StateVector([[i + 1]]), covar=np.array([[1]]),
                      timestamp=time_start + datetime.timedelta(seconds=1))
    ]) for i in range(5)}

    manager = MultiManager([generator])
    manager.add_data({'groundtruth_paths': truths, 'tracks': tracks})

    metric = generator.compute_metric(manager)
    anees_value = metric.value['ANEES']

    # Expected NEES value calculation
    expected_nees_value = 0.25  # Since error is 0.5 and covariance is 1, NEES = (0.5^2)/1

    assert anees_value == pytest.approx(expected_nees_value)

def test_neesmetric_negative_values():
    """Test NEESMetric with negative values in state vectors."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    # Measured and truth states with negative values
    measured_state = GaussianState(
        state_vector=StateVector([[-1], [-2], [-3]]),
        covar=np.eye(3),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[-1.5], [-2.5], [-3.5]]),
        covar=np.eye(3),
        timestamp=time
    )

    # Compute NEES metric
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    anees = metric.value['ANEES']

    # Expected NEES calculation
    error = truth_state.state_vector - measured_state.state_vector
    cov_inv = np.linalg.inv(measured_state.covar)
    DoF = measured_state.state_vector.shape[0]
    expected_nees = (error.T @ cov_inv @ error).item() / DoF

    assert anees == pytest.approx(expected_nees)


def test_neesmetric_varying_covariances():
    """Test NEESMetric with varying covariance matrices."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    # Measured states with varying covariances
    measured_states = [
        GaussianState(state_vector=StateVector([[i]]), covar=np.array([[i + 1]]), timestamp=time) for i in range(5)
    ]
    truth_states = [
        GaussianState(state_vector=StateVector([[i + 0.5]]), covar=np.array([[1]]), timestamp=time) for i in range(5)
    ]

    # Compute NEES metric
    metric = generator.compute_over_time_v2(measured_states, list(range(5)), truth_states, list(range(5)))
    anees = metric.value['ANEES']

    # Expected NEES calculation
    nees_values = []
    for i in range(5):
        error = truth_states[i].state_vector - measured_states[i].state_vector
        P_inv = np.linalg.inv(measured_states[i].covar)
        nees_value = (error.T @ P_inv @ error).item()
        nees_values.append(nees_value)
    expected_anees = sum(nees_values) / len(nees_values)  # Not normalized by DoF as DoF=1

    assert anees == pytest.approx(expected_anees)

def test_neesmetric_mismatched_state_counts():
    """Test NEESMetric with mismatched numbers of measured and truth states."""
    generator = NEESMetric()
    time = datetime.datetime.now()

    # Measured states
    measured_states = [
        GaussianState(state_vector=StateVector([[i]]), covar=np.array([[1]]), timestamp=time) for i in range(5)
    ]
    # Truth states (fewer than measured states)
    truth_states = [
        GaussianState(state_vector=StateVector([[i]]), covar=np.array([[1]]), timestamp=time) for i in range(3)
    ]

    # Compute NEES metric
    metric = generator.compute_over_time_v2(measured_states, list(range(5)), truth_states, list(range(3)))

    # Verify that the metric computed ANEES and handled mismatched counts
    anees = metric.value['ANEES']
    assert anees >= 0

def test_neesmetric_different_timestamps_exception():
    """Test NEESMetric with states from different timestamps."""
    generator = NEESMetric()
    time1 = datetime.datetime.now()
    time2 = time1 + datetime.timedelta(seconds=1)

    measured_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.array([[1]]),
        timestamp=time1
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.array([[1]]),
        timestamp=time2
    )

    # Compute NEES metric
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

    # Since timestamps are different, there should be no matched assignments
    assert metric.value['ANEES'] == 0.0
    assert metric.value['Average False Tracks'] == 0.5
    assert metric.value['Average Missed Targets'] == 0.5