import datetime
import numpy as np
import pytest
from stonesoup.metricgenerator.RMSEMetric_v3 import RMSEMetric  # import my RMSEMetric class
from stonesoup.metricgenerator.manager import MultiManager
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.state import State, GaussianState, ParticleState
from stonesoup.types.array import StateVector
from stonesoup.types.track import Track
from stonesoup.types.particle import Particle



# A. Input Validation Tests
# 1. Test with Empty State Vectors
def test_rmsemetric_empty_state_vectors():
    """Test RMSEMetric with empty state vectors."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[]]),  # Empty state vector
        covar=np.eye(0),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[]]),  # Empty state vector
        covar=np.eye(0),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])


# 2. Test with None State Vectors
def test_rmsemetric_none_state_vectors():
    """Test RMSEMetric with None as state vectors."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    measured_state = State(
        state_vector=None,
        timestamp=time
    )
    truth_state = State(
        state_vector=None,
        timestamp=time
    )

    with pytest.raises(ValueError, match="State vectors cannot be None."):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

# 3. Test with Incorrect Data Types
def test_rmsemetric_incorrect_data_types():
    """Test RMSEMetric with incorrect data types as inputs."""
    generator = RMSEMetric()
    measured_states = "invalid_input"
    truth_states = "invalid_input"

    with pytest.raises(ValueError):
        generator.compute_over_time_v2(measured_states, None, truth_states, None)

# B. Dimension Consistency Tests
# 1. Test with Mismatched State Vector Dimensions
def test_rmsemetric_mismatched_state_vector_dimensions():
    """Test RMSEMetric with mismatched state vector dimensions."""
    generator = RMSEMetric()
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

    measured_states = [measured_state]
    truth_states = [truth_state]

    with pytest.raises(ValueError):
        generator.compute_over_time_v2(measured_states, [0], truth_states, [0])

# 2. Test with Different Number of States
def test_rmsemetric_different_number_of_states():
    """Test RMSEMetric with different numbers of measured and truth states."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    measured_states = [
        GaussianState(state_vector=StateVector([[i]]), covar=np.eye(1), timestamp=time)
        for i in range(5)
    ]
    truth_states = [
        GaussianState(state_vector=StateVector([[i]]), covar=np.eye(1), timestamp=time)
        for i in range(3)
    ]

    metric = generator.compute_over_time_v2(measured_states, list(range(5)), truth_states, list(range(3)))
    assert metric.value['Average False Tracks'] > 0 or metric.value['Average Missed Targets'] > 0


# C. Numerical Stability Tests
# 2. Test with Extremely Large Numbers
def test_rmsemetric_extremely_large_numbers():
    """Test RMSEMetric with extremely large numbers."""
    generator = RMSEMetric()
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
    rmse = metric.value['RMSE']
    assert np.isfinite(rmse), "RMSE should be finite with large numbers"


# 3. Test with Extremely Small Numbers
def test_rmsemetric_extremely_small_numbers():
    """Test RMSEMetric with extremely small numbers."""
    generator = RMSEMetric()
    time = datetime.datetime.now()
    small_value = 1e-306 # Close to the minimum positive normal float

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
    rmse = metric.value['RMSE']
    assert np.isfinite(rmse), "RMSE should be finite with small numbers"


# D. Performance Tests
# 1. Test with High-Dimensional State Vectors (beyond that would take some time to run)
def test_rmsemetric_high_dimensional_states():
    """Test RMSEMetric with high-dimensional state vectors."""
    generator = RMSEMetric()
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
    rmse = metric.value['RMSE']
    assert rmse >= 0, "RMSE should be non-negative"


# E. Timestamp Handling Tests
# 1. Test with Missing Timestamps
def test_rmsemetric_missing_timestamps():
    """Test RMSEMetric with states missing timestamps."""
    generator = RMSEMetric()

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


# 2. Test with Mismatched Timestamps
def test_rmsemetric_mismatched_timestamps():
    """Test RMSEMetric with mismatched timestamps."""
    generator = RMSEMetric()
    time1 = datetime.datetime.now()
    time2 = time1 + datetime.timedelta(seconds=1)

    measured_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=time1
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=time2
    )

    # Should not raise an error; the module should handle multiple timestamps
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    # Since there are no matched timestamps, the metric should indicate no matched states
    assert metric.value['RMSE'] == 0.0
    assert metric.value['Average False Tracks'] > 0 or metric.value['Average Missed Targets'] > 0


# F. Covariance Matrix Tests
# 1. Test with Singular Covariance Matrix (In cas the Mahalanobis distance is used)
def test_rmsemetric_singular_covariance():
    """Test RMSEMetric with singular covariance matrix."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    singular_covar = np.array([[1, 1], [1, 1]])  # Singular matrix

    measured_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=singular_covar,
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=np.eye(2),
        timestamp=time
    )

    # Should not raise an error, as it does not invert the covariance matrix
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    rmse = metric.value['RMSE']
    assert rmse == 0.0


# G. State Type Tests
# 1. Test with Mixed State Types
def test_rmsemetric_mixed_state_types():
    """Test RMSEMetric with mixed state types."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    measured_state = State(
        state_vector=StateVector([[0]]),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=time
    )

    # Should work, as long as state_vector is available
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    rmse = metric.value['RMSE']
    assert rmse == 0.0


# 2. Test with Non-Gaussian States
def test_rmsemetric_non_gaussian_states():
    """Test RMSEMetric with non-Gaussian states."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    measured_state = State(
        state_vector=StateVector([[0]]),
        timestamp=time
    )
    truth_state = State(
        state_vector=StateVector([[0]]),
        timestamp=time
    )

    # Should work, as long as state_vector is there
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    rmse = metric.value['RMSE']
    assert rmse == 0.0


def test_rmsemetric_with_particle_states():
    """Test RMSEMetric with valid ParticleState inputs."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    # Create ParticleState instances with valid particles
    measured_state = ParticleState(
        state_vector=None,
        particle_list=[Particle(state_vector=StateVector([[1]]), weight=1.0)],
        timestamp=time
    )
    truth_state = ParticleState(
        state_vector=None,
        particle_list=[Particle(state_vector=StateVector([[2]]), weight=1.0)],
        timestamp=time
    )

    # Compute the metric
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    rmse = metric.value['RMSE']

    # Compute expected RMSE
    expected_rmse = 1.0  # Difference between 1 and 2

    assert rmse == pytest.approx(expected_rmse)


# H. Invalid Values Tests
# 1. Test with State Vectors Containing NaNs
def test_rmsemetric_state_vectors_with_nans():
    """Test RMSEMetric with state vectors containing NaNs."""
    generator = RMSEMetric()
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


# 2. Test with State Vectors Containing Infs
def test_rmsemetric_state_vectors_with_infs():
    """Test RMSEMetric with state vectors containing Infs."""
    generator = RMSEMetric()
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


# I. Incorrect Usage of the Module
# 1. Test with Incorrectly Configured Parameters
def test_rmsemetric_invalid_confidence_threshold():
    """Test RMSEMetric with invalid confidence threshold."""
    with pytest.raises(ValueError):
        RMSEMetric(confidence_threshold=-1)  # Negative threshold is invalid


# 2. Test Calling Methods in Wrong Order
def test_rmsemetric_method_call_order():
    """Test RMSEMetric method call order."""
    generator = RMSEMetric()

    # Attempt to call compute_rmse_metric_v2 directly without proper inputs
    with pytest.raises(TypeError):
        generator.compute_rmse_metric_v2()


# J. Component-Specific Tests
# 1. Test RMSE with Position Components
def test_rmsemetric_position_components():
    """Test RMSEMetric with position components."""
    generator = RMSEMetric(components=[0, 2])  # Specify components
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[1], [2], [3], [4]]),
        covar=np.eye(4),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1.5], [2.5], [3.5], [4.5]]),
        covar=np.eye(4),
        timestamp=time
    )

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    rmse = metric.value['RMSE']

    # Compute expected RMSE over components [0, 2] (assuming Euclidian)
    error = truth_state.state_vector[generator.components, :] - measured_state.state_vector[generator.components, :]
    squared_error = np.sum(error ** 2)
    expected_rmse = np.sqrt(squared_error / 1)

    assert rmse == pytest.approx(expected_rmse)


# 2. Test RMSE with Velocity Components
def test_rmsemetric_velocity_components():
    """Test RMSEMetric with velocity components."""
    generator = RMSEMetric(components=[1, 3])
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[1], [2], [3], [4]]),
        covar=np.eye(4),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1.5], [2.5], [3.5], [4.5]]),
        covar=np.eye(4),
        timestamp=time
    )

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    rmse = metric.value['RMSE']

    # Compute expected RMSE over components [1, 3]
    error = truth_state.state_vector[generator.components] - measured_state.state_vector[generator.components]

    squared_error = np.sum(error ** 2)
    expected_rmse = np.sqrt(squared_error)

    assert rmse == pytest.approx(expected_rmse)
    print(f"Computed RMSE: {rmse}, Expected RMSE: {expected_rmse}")

# K. Additional Tests Based on RMSEMetric_v2
# 1. Test RMSE Computation Over Time with Known Inputs
def test_rmsemetric_compute_metric_over_time():
    """Test RMSEMetric compute_metric over time with known inputs."""
    generator = RMSEMetric()
    time_start = datetime.datetime.now()
    # Create tracks with States
    tracks = {Track(states=[
        State(state_vector=StateVector([[i + 0.5]]), timestamp=time_start),
        State(state_vector=StateVector([[i + 1.5]]), timestamp=time_start + datetime.timedelta(seconds=1))
    ]) for i in range(5)}

    # Create ground truths
    truths = {GroundTruthPath(states=[
        State(state_vector=StateVector([[i]]), timestamp=time_start),
        State(state_vector=StateVector([[i + 1]]), timestamp=time_start + datetime.timedelta(seconds=1))
    ]) for i in range(5)}

    manager = MultiManager([generator])
    manager.add_data({'groundtruth_paths': truths, 'tracks': tracks})

    metric = generator.compute_metric(manager)
    rmse_value = metric.value['RMSE']

    # Expected RMSE value calculation
    expected_rmse_value = 0.5  # Since error is 0.5, RMSE = sqrt(mean(0.5^2)) = 0.5

    assert rmse_value == pytest.approx(expected_rmse_value)
    print(f"Computed RMSE: {rmse_value}, Expected RMSE: {expected_rmse_value}")


# 2. Test RMSE with Negative Values in State Vectors
def test_rmsemetric_negative_values():
    """Test RMSEMetric with negative values in state vectors."""
    generator = RMSEMetric()
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

    # Compute RMSE metric
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    rmse = metric.value['RMSE']

    # Expected RMSE calculation
    error = truth_state.state_vector - measured_state.state_vector

    squared_error = np.sum(error ** 2)
    expected_rmse = np.sqrt(squared_error)

    assert rmse == pytest.approx(expected_rmse)
    print(f"Computed RMSE: {rmse}, Expected RMSE: {expected_rmse}")


# 3. Test RMSE with Multiple States per Timestamp
def test_rmsemetric_multiple_states_per_timestamp():
    """Test RMSEMetric with multiple states per timestamp."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    # Create multiple measured and truth states at the same timestamp
    measured_states = [
        GaussianState(state_vector=StateVector([[i]]), covar=np.array([[1]]), timestamp=time) for i in range(5)
    ]
    truth_states = [
        GaussianState(state_vector=StateVector([[i + 0.5]]), covar=np.array([[1]]), timestamp=time) for i in range(5)
    ]

    # Compute RMSE metric
    metric = generator.compute_over_time_v2(measured_states, list(range(5)), truth_states, list(range(5)))
    rmse = metric.value['RMSE']

    # Expected RMSE calculation
    errors = [0.5] * 5  # Error is 0.5 for each state
    expected_rmse = np.sqrt(np.mean(np.array(errors) ** 2))

    assert rmse == pytest.approx(expected_rmse)
    print(f"Computed RMSE: {rmse}, Expected RMSE: {expected_rmse}")


# 4. Test RMSE with Varying Covariance Matrices
def test_rmsemetric_varying_covariances():
    """Test RMSEMetric with varying covariance matrices."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    # Measured states with varying covariances
    measured_states = [
        GaussianState(state_vector=StateVector([[i]]), covar=np.array([[i + 1]]), timestamp=time) for i in range(5)
    ]
    truth_states = [
        GaussianState(state_vector=StateVector([[i + 0.5]]), covar=np.array([[1]]), timestamp=time) for i in range(5)
    ]

    # Compute RMSE metric
    metric = generator.compute_over_time_v2(measured_states, list(range(5)), truth_states, list(range(5)))
    rmse = metric.value['RMSE']

    # Expected RMSE calculation
    errors = [0.5] * 5  # Error is 0.5 for each state
    expected_rmse = np.sqrt(np.mean(np.array(errors) ** 2))

    assert rmse == pytest.approx(expected_rmse)
    print(f"Computed RMSE: {rmse}, Expected RMSE: {expected_rmse}")


# 5. Test RMSE with Non-Positive Definite Covariance Matrices
def test_rmsemetric_non_positive_definite_covariance():
    """Test RMSEMetric with non-positive definite covariance matrices."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    # Measured state with non-positive definite covariance matrix
    measured_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.array([[-1]]),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.array([[1]]),
        timestamp=time
    )

    # RMSE should not be affected by covariance matrix in this implementation
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    rmse = metric.value['RMSE']
    assert rmse == 0.0


# 6. Test RMSE with Covariance Matrices Having Off-Diagonal Terms
def test_rmsemetric_covariance_with_off_diagonal_terms():
    """Test RMSEMetric with covariance matrices that have off-diagonal terms."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    # Measured states with covariance matrices having off-diagonal terms
    measured_state = GaussianState(
        state_vector=StateVector([[1], [2]]),
        covar=np.array([[1, 0.5], [0.5, 2]]),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1.5], [2.5]]),
        covar=np.array([[1, 0], [0, 1]]),
        timestamp=time
    )

    # Compute RMSE metric
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    rmse = metric.value['RMSE']

    # Expected RMSE calculation
    error = truth_state.state_vector - measured_state.state_vector

    squared_error = np.sum(error ** 2)
    expected_rmse = np.sqrt(squared_error)  # Divide by number of assigned pairs

    assert rmse == pytest.approx(expected_rmse)
    print(f"Computed RMSE: {rmse}, Expected RMSE: {expected_rmse}")


# 7. Test RMSE with Variable Dimensions
def test_rmsemetric_variable_dimensions():
    """Test RMSEMetric with Gaussian state vectors of varying dimensions."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    # Measured state with 3 dimensions
    measured_state = GaussianState(
        state_vector=StateVector([[1], [2], [3]]),
        covar=np.eye(3),
        timestamp=time
    )
    # Truth state with 2 dimensions
    truth_state = GaussianState(
        state_vector=StateVector([[1.5], [2.5]]),
        covar=np.eye(2),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])


def test_rmsemetric_both_position_velocity_components():
    """Test RMSEMetric with both position and velocity components."""
    generator = RMSEMetric(components=[0, 1, 2, 3])
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[1], [2], [3], [4]]),
        covar=np.eye(4),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[2], [3], [4], [5]]),
        covar=np.eye(4),
        timestamp=time
    )

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    rmse = metric.value['RMSE']

    # Compute expected RMSE over components [0, 1, 2, 3]
    error = truth_state.state_vector[generator.components] - measured_state.state_vector[generator.components]

    squared_error = np.sum(error ** 2)
    expected_rmse = np.sqrt(squared_error)

    assert rmse == pytest.approx(expected_rmse)
    print(f"Computed RMSE: {rmse}, Expected RMSE: {expected_rmse}")


# Test RMSE with Out-of-Order Timestamps
def test_rmsemetric_out_of_order_timestamps():
    """Test RMSEMetric with out-of-order timestamps."""
    generator = RMSEMetric()
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
    rmse = metric.value['RMSE']
    assert rmse >= 0.0


# Test RMSE with Variable Confidence Thresholds
def test_rmsemetric_varying_confidence_thresholds():
    """Test RMSEMetric with varying confidence_threshold values."""
    time = datetime.datetime.now()
    measured_states = [
        GaussianState(state_vector=StateVector([[0]]), covar=np.eye(1), timestamp=time),
        GaussianState(state_vector=StateVector([[10]]), covar=np.eye(1), timestamp=time),
    ]
    truth_states = [
        GaussianState(state_vector=StateVector([[1]]), covar=np.eye(1), timestamp=time),
        GaussianState(state_vector=StateVector([[9]]), covar=np.eye(1), timestamp=time),
    ]

    thresholds = [0.5, 2, 5, 10]
    results = []
    for threshold in thresholds:
        generator = RMSEMetric(confidence_threshold=threshold)
        metric = generator.compute_over_time_v2(measured_states, [0, 1], truth_states, [0, 1])
        results.append({
            'threshold': threshold,
            'RMSE': metric.value['RMSE'],
            'Average False Tracks': metric.value['Average False Tracks'],
            'Average Missed Targets': metric.value['Average Missed Targets'],
        })

    # Check that as threshold increases, the number of assignments should not decrease
    previous_assigned = 0
    for result in results:
        num_assigned = 2 - result['Average False Tracks']
        assert num_assigned >= previous_assigned, \
            f"Number of assignments decreased when threshold increased to {result['threshold']}"
        previous_assigned = num_assigned


# Test RMSE with High Measurement Errors
def test_rmsemetric_high_measurement_errors():
    """Test RMSEMetric behavior with high measurement errors."""
    generator = RMSEMetric(confidence_threshold=2000)
    time = datetime.datetime.now()

    # Create states with large errors
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
    rmse = metric.value['RMSE']

    # Since the error is large but within the confidence threshold, RMSE should reflect it
    expected_rmse = 1000.0
    assert rmse == pytest.approx(expected_rmse)
    print(f"Computed RMSE: {rmse}, Expected RMSE: {expected_rmse}")


def test_rmsemetric_no_assignments():
    """Test RMSEMetric when there are no valid assignments."""
    generator = RMSEMetric()
    time = datetime.datetime.now()

    # States are too far apart to be within the confidence threshold
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

    # Set a low confidence threshold to prevent assignment
    generator.confidence_threshold = 1

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    rmse = metric.value['RMSE']

    assert rmse == 0.0
    assert metric.value['Average False Tracks'] == 1
    assert metric.value['Average Missed Targets'] == 1


@pytest.mark.parametrize("components, expected_rmse", [
    ([0, 2], np.sqrt(2)),  # Approximately 1.41421356
    ([1, 3], np.sqrt(2)),  # Approximately 1.41421356
    ([0, 1, 2, 3], 2.0)  # Exact value
])
def test_rmsemetric_components(components, expected_rmse):
    """Test RMSEMetric with different component selections."""
    generator = RMSEMetric(components=components)
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[1], [2], [3], [4]]),
        covar=np.eye(4),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[2], [3], [4], [5]]),
        covar=np.eye(4),
        timestamp=time
    )

    # Compute the metric
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    rmse = metric.value['RMSE']

    # For debugging purposes, print the results
    print(f"Components: {components}, Computed RMSE: {rmse}, Expected RMSE: {expected_rmse}")

    # Assert that the computed RMSE matches the expected value
    assert rmse == pytest.approx(expected_rmse)