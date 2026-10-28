import datetime
import numpy as np
import pytest
from stonesoup.metricgenerator.KLMetric_v3 import KLMetric # Import my KLMetric class
from stonesoup.types.state import State
from stonesoup.types.state import GaussianState, ParticleState
from stonesoup.types.array import StateVector
from stonesoup.types.particle import Particle
from stonesoup.measures.state import KLDivergence

# A. Input Validation Tests
# 1. Test with Empty Inputs
# Purpose: Verify that the module handles empty inputs gracefully.

def test_klmetric_empty_inputs():
    """Test KLMetric with empty measured and truth states."""
    generator = KLMetric()
    measured_states = []
    truth_states = []

    metric = generator.compute_over_time_v2(measured_states, [], truth_states, [])
    assert metric.value['Average KL Divergence'] == 0.0
    assert metric.value['Average False Tracks'] == 0.0
    assert metric.value['Average Missed Targets'] == 0.0

# 2. Test with Null Inputs
# Purpose: Ensure that the module raises an error when inputs are None.

def test_klmetric_null_inputs():
    """Test KLMetric with None as inputs."""
    generator = KLMetric()
    measured_states = None
    truth_states = None

    with pytest.raises(TypeError):
        generator.compute_over_time_v2(measured_states, None, truth_states, None)

# 3. Test with Incorrect Data Types
# Purpose: Verify that the module raises an error when inputs are of incorrect types.

def test_klmetric_incorrect_data_types():
    """Test KLMetric with incorrect data types as inputs."""
    generator = KLMetric()
    measured_states = "invalid_input"
    truth_states = "invalid_input"

    with pytest.raises(AttributeError):
        generator.compute_over_time_v2(measured_states, None, truth_states, None)

# B. Dimension Consistency Tests
# 1. Test with Mismatched State Vector Dimensions
# Purpose: Ensure that the module handles mismatched state vector dimensions properly.

def test_klmetric_mismatched_state_vector_dimensions():
    """Test KLMetric with mismatched state vector dimensions."""
    generator = KLMetric()
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
# Purpose: Verify that the module can handle different numbers of measured and truth states.

def test_klmetric_different_number_of_states():
    """Test KLMetric with different numbers of measured and truth states."""
    generator = KLMetric()
    time = datetime.datetime.now()

    measured_states = [
        GaussianState(state_vector=StateVector([[i]]), covar=np.eye(1), timestamp=time)
        for i in range(5)
    ]
    truth_states = [
        GaussianState(state_vector=StateVector([[i]]), covar=np.eye(1), timestamp=time)
        for i in range(3)
    ]

    # Should not raise an error. Unmatched states will be counted as false tracks or missed targets
    metric = generator.compute_over_time_v2(measured_states, list(range(5)), truth_states, list(range(3)))
    assert metric.value['Average False Tracks'] > 0 or metric.value['Average Missed Targets'] > 0

# C.
# 1. Test with Zero Covariance Matrix
# Purpose: Ensure that the module raises an error when covariance matrices are zero.
def test_klmetric_zero_covariance():
    """Test KLMetric with zero covariance matrix."""
    generator = KLMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.zeros((1, 1)),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=time
    )

    with pytest.raises(np.linalg.LinAlgError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

# 2. Test with Extremely Large Numbers
# Purpose: Verify that the module can handle large numbers without numerical instability.
def test_klmetric_extremely_large_numbers():
    """Test KLMetric with extremely large numbers."""
    generator = KLMetric()
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
    kl_divergence = metric.value['Average KL Divergence']
    assert np.isfinite(kl_divergence), "KL divergence should be finite with large numbers"

# 3. Test with Extremely Small Numbers
# Purpose: Ensure that small numbers do not cause underflow issues.

def test_klmetric_extremely_small_numbers():
    """Test KLMetric with extremely small numbers."""
    generator = KLMetric()
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
    kl_divergence = metric.value['Average KL Divergence']
    assert np.isfinite(kl_divergence), "KL divergence should be finite with small numbers"

# D. Performance Tests
# 1. Test with High-Dimensional State Vectors
# Purpose: Assess performance and correctness with high-dimensional state vectors.

def test_klmetric_high_dimensional_states():
    """Test KLMetric with high-dimensional state vectors."""
    generator = KLMetric()
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
    kl_divergence = metric.value['Average KL Divergence']
    assert kl_divergence >= 0, "KL divergence should be non-negative"

# E. Component Handling Tests
# 1. Test with Invalid Component Indices
# Purpose: Ensure that invalid component indices raise appropriate errors.

def test_klmetric_invalid_component_indices():
    """Test KLMetric with invalid component indices."""
    generator = KLMetric(components=[0, 10])  # Assuming state vectors are smaller

    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=np.eye(2),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=np.eye(2),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

# 2. Test with Component Specification
# Purpose: Verify that the module correctly handles specified components.

def test_klmetric_component_specification():
    """Test KLMetric with specific components."""
    generator = KLMetric(components=[0])

    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=np.array([[1, 0], [0, 1]]),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[1], [1]]),
        covar=np.array([[1, 0], [0, 1]]),
        timestamp=time
    )

    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    kl_divergence = metric.value['Average KL Divergence']
    expected_kl = 0.5  # Since only component 0 differs by 1 with unit variance
    assert kl_divergence == pytest.approx(expected_kl)

# F. Timestamp Handling Tests
# 1. Test with Missing Timestamps
# Purpose: Ensure that states without timestamps are handled appropriately.

def test_klmetric_missing_timestamps():
    """Test KLMetric with states missing timestamps."""
    generator = KLMetric()

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
# Purpose: Verify that states with different timestamps are handled correctly.

def test_klmetric_mismatched_timestamps():
    """Test KLMetric with mismatched timestamps."""
    generator = KLMetric()
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
    assert metric.value['Average KL Divergence'] == 0.0
    assert metric.value['Average False Tracks'] > 0 or metric.value['Average Missed Targets'] > 0

# G. Covariance Matrix Tests
# 1. Test with Singular Covariance Matrix
# Purpose: Ensure that singular covariance matrices raise appropriate errors.

def test_klmetric_singular_covariance():
    """Test KLMetric with singular covariance matrix."""
    generator = KLMetric()
    time = datetime.datetime.now()

    # Singular covariance matrix
    singular_covar = np.array([[1, 1], [1, 1]])  # Singular matrix

    measured_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=singular_covar,
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=singular_covar,
        timestamp=time
    )

    with pytest.raises(np.linalg.LinAlgError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])


# 2. Test with Covariance Matrices Containing NaNs
# Purpose: Verify that covariance matrices with invalid numbers raise errors
def test_klmetric_covariance_with_nans():
    """Test KLMetric with covariance matrices containing NaNs."""
    generator = KLMetric()
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

# H. State Type Tests
# 1. Test with Mixed State Types
# Purpose: Ensure that the module handles mixed state types appropriately.

def test_klmetric_mixed_state_types():
    """Test KLMetric with mixed state types."""
    generator = KLMetric()
    time = datetime.datetime.now()

    measured_state = ParticleState(
        None,  # Set state_vector to None
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

# 2. Test with Non-Gaussian States
# Purpose: Verify that non-Gaussian states raise appropriate errors.

def test_klmetric_non_gaussian_states():
    """Test KLMetric with non-Gaussian states."""
    generator = KLMetric()
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

# I. Particle Weight Tests
# 1. Test with Zero Particle Weights
# Purpose: Ensure that particles with zero weights are handled correctly.

def test_klmetric_zero_particle_weights():
    """Test KLMetric with particles having zero weights."""
    generator = KLMetric()
    time = datetime.datetime.now()


    measured_state = ParticleState(
        state_vector=None,
        particle_list=[Particle(state_vector=StateVector([[0]]), weight=0.0)],
        timestamp=time
    )

    truth_state = ParticleState(
        state_vector=None,
        particle_list=[Particle(state_vector=StateVector([[0]]), weight=1.0)],
        timestamp=time)

    with pytest.raises(TypeError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])


# J. Invalid Values Tests
# 1. Test with State Vectors Containing NaNs
# Purpose: Ensure that state vectors with NaN values raise errors.

def test_klmetric_state_vectors_with_nans():
    """Test KLMetric with state vectors containing NaNs."""
    generator = KLMetric()
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
# Purpose: Verify that state vectors with Inf values raise errors.

def test_klmetric_state_vectors_with_infs():
    """Test KLMetric with state vectors containing Infs."""
    generator = KLMetric()
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
# K. Incorrect Usage of the Module
# 1. Test with Incorrectly Configured Parameters
# Purpose: Ensure that invalid parameter configurations raise errors.

def test_klmetric_invalid_confidence_threshold():
    """Test KLMetric with invalid confidence threshold."""
    with pytest.raises(ValueError):
        KLMetric(confidence_threshold=-1)  # Negative threshold is invalid


# 2. Test Calling Methods in Wrong Order
# Purpose: Verify that methods cannot be called incorrectly.

def test_klmetric_method_call_order():
    """Test KLMetric method call order."""
    generator = KLMetric()

    # Attempt to call compute_kl_metric_v2 directly without proper inputs
    with pytest.raises(TypeError):
        generator.compute_kl_metric_v2()

######################################################################################################################
###################################### 1. Tests for compute_over_time_v2 Method ######################################
######################################################################################################################

# Test with No States Provided
def test_compute_over_time_no_states():
    """Test compute_over_time_v2 with no states provided."""
    generator = KLMetric()
    measured_states = []
    truth_states = []

    metric = generator.compute_over_time_v2(measured_states, [], truth_states, [])
    assert metric.value['Average KL Divergence'] == 0.0
    assert metric.value['Average False Tracks'] == 0.0
    assert metric.value['Average Missed Targets'] == 0.0

# Test with Missing Timestamps
def test_compute_over_time_missing_timestamps():
    """Test compute_over_time_v2 with states missing timestamps."""
    generator = KLMetric()
    measured_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1)
        # Missing timestamp on purpose
    )

    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=datetime.datetime.now()
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

# Test with Out-of-Order Timestamps
def test_compute_over_time_out_of_order_timestamps():
    """Test compute_over_time_v2 with out-of-order timestamps."""
    generator = KLMetric()
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
    assert metric.value['Average KL Divergence'] >= 0.0

# Test with Invalid Timestamps
def test_compute_over_time_invalid_timestamps():
    """Test compute_over_time_v2 with invalid timestamps."""
    generator = KLMetric()
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


######################################################################################################################
###################################### 2. Tests for compute_kl_metric_v2 Method ######################################
######################################################################################################################


# Test with Mismatched State Vector Dimensions
def test_compute_kl_metric_mismatched_dimensions():
    """Test compute_kl_metric_v2 with mismatched state vector dimensions."""
    generator = KLMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=np.eye(2),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=time
    )

    with pytest.raises(ValueError):
        # This will call compute_kl_metric_v2 internally
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

# Test with Unassigned States

def test_compute_kl_metric_unassigned_states():
    """Test compute_kl_metric_v2 with unassigned states."""
    generator = KLMetric()
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
    assert metric.value['Average KL Divergence'] == 0.0  # No matched states
    assert metric.value['Average False Tracks'] == 1.0
    assert metric.value['Average Missed Targets'] == 1.0


# Test with Invalid State Types

def test_compute_kl_metric_invalid_state_types():
    """Test compute_kl_metric_v2 with invalid state types."""
    generator = KLMetric()
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

    with pytest.raises(TypeError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])


######################################################################################################################
###################################### 3. Tests for compute_cost_matrix_v2 Method ####################################
######################################################################################################################


# Test with No States Provided

def test_compute_cost_matrix_no_states():
    """Test compute_cost_matrix_v2 with no states."""
    generator = KLMetric()
    measured_states = []
    truth_states = []

    # Since compute_cost_matrix_v2 is internal, we'll check that compute_over_time_v2 doesn't fail
    metric = generator.compute_over_time_v2(measured_states, [], truth_states, [])
    assert metric.value['Average KL Divergence'] == 0.0

# Test with Singular Covariance Matrix

def test_compute_cost_matrix_singular_covariance():
    """Test compute_cost_matrix_v2 with singular covariance matrix."""
    generator = KLMetric()
    time = datetime.datetime.now()

    singular_covar = np.array([[1, 1], [1, 1]])

    measured_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=singular_covar,
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=singular_covar,
        timestamp=time
    )

    with pytest.raises(np.linalg.LinAlgError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])


# Test with Extreme Mahalanobis Distance

def test_compute_cost_matrix_high_mahalanobis():
    """Test compute_cost_matrix_v2 with high Mahalanobis distance."""
    generator = KLMetric(confidence_threshold=2)
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
    assert cost_matrix[0, 0] == generator.confidence_threshold, "Cost should be set to c for high Mahalanobis distance"

# Test with NaN and Inf Values

def test_compute_cost_matrix_nan_inf_values():
    """Test compute_cost_matrix_v2 with NaN and Inf values."""
    generator = KLMetric()
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[np.nan]]),
        covar=np.eye(1),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[np.inf]]),
        covar=np.eye(1),
        timestamp=time
    )

    with pytest.raises(ValueError):
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

# Test with Mismatched Number of States

def test_compute_cost_matrix_mismatched_numbers():
    """Test compute_cost_matrix_v2 with mismatched numbers of states."""
    generator = KLMetric()
    time = datetime.datetime.now()

    measured_states = [
        GaussianState(state_vector=StateVector([[i]]), covar=np.eye(1), timestamp=time)
        for i in range(3)
    ]
    truth_states = [
        GaussianState(state_vector=StateVector([[i]]), covar=np.eye(1), timestamp=time)
        for i in range(5)
    ]

    # Since compute_cost_matrix_v2 is internal, we'll ensure compute_over_time_v2 works correctly
    metric = generator.compute_over_time_v2(measured_states, list(range(3)), truth_states, list(range(5)))
    assert metric.value['Average KL Divergence'] >= 0.0


######################################################################################################################
################################################## 4.Extra tests #####################################################
######################################################################################################################

# 3.4. Asymmetry Tests
def test_klmetric_asymmetry():
    """Test that KL divergence is asymmetric."""
    time = datetime.datetime.now()

    state1 = GaussianState(
        state_vector=StateVector([[0], [0]]),
        covar=np.eye(2),
        timestamp=time
    )
    state2 = GaussianState(
        state_vector=StateVector([[1], [1]]),
        covar=2 * np.eye(2),
        timestamp=time
    )

    # Compute KL divergence in both directions
    kl_divergence = KLDivergence()
    kl1 = kl_divergence(state1, state2)
    kl2 = kl_divergence(state2, state1)

    assert kl1 != kl2, "KL divergence should be asymmetric"


# 3.5. Threshold Handling Tests
# Test with Varying Confidence Thresholds

def test_klmetric_confidence_threshold():
    """Test KLMetric with varying confidence thresholds."""
    time = datetime.datetime.now()

    measured_state = GaussianState(
        state_vector=StateVector([[0]]),
        covar=np.eye(1),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[3]]),
        covar=np.eye(1),
        timestamp=time
    )

    # Threshold that allows assignment
    generator = KLMetric(confidence_threshold=5)
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    kl_divergence_assigned = metric.value['Average KL Divergence']

    # Threshold that prevents assignment
    generator = KLMetric(confidence_threshold=1)
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    kl_divergence_unassigned = metric.value['Average KL Divergence']

    assert kl_divergence_assigned > 0, "KL divergence should be computed when assigned"
    assert kl_divergence_unassigned == 0, "KL divergence should be zero when not assigned"



#  Alternative Distance Measures
def test_klmetric_with_mahalanobis_distance():
    """Test KLMetric using Mahalanobis distance instead of Euclidean distance."""
    from stonesoup.measures import Mahalanobis

    # Set confidence_threshold higher than Mahalanobis distance
    generator = KLMetric(measure=Mahalanobis(), confidence_threshold=5.1)
    time = datetime.datetime.now()

    # Create measured and truth states with differing means and covariances
    measured_state = GaussianState(
        state_vector=StateVector([[5], [5]]),
        covar=np.array([[2, 0], [0, 2]]),
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0], [0]]),
        covar=np.array([[1, 0], [0, 1]]),
        timestamp=time
    )

    # Compute the metric
    metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])

    # Extract KL divergence and ensure it is computed
    kl_divergence = metric.value['Average KL Divergence']
    assert kl_divergence > 0, "KL divergence should be positive when using Mahalanobis distance"

#  Numerical Stability Tests
def test_klmetric_near_singular_covariance():
    """Test KLMetric with near-singular covariance matrices for numerical stability."""
    generator = KLMetric()
    time = datetime.datetime.now()

    # Near-singular covariance matrix
    epsilon = 1e-10
    near_singular_covar = np.array([[1, 1 - epsilon], [1 - epsilon, 1]])

    measured_state = GaussianState(
        state_vector=StateVector([[0], [1]]),
        covar=near_singular_covar,
        timestamp=time
    )
    truth_state = GaussianState(
        state_vector=StateVector([[0.1], [1.1]]),
        covar=near_singular_covar,
        timestamp=time
    )

    # Try to compute the metric and handle potential numerical issues
    try:
        metric = generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
        kl_divergence = metric.value['Average KL Divergence']
        assert kl_divergence >= 0, "KL divergence should be non-negative"
    except np.linalg.LinAlgError:
        pytest.fail("KLMetric failed due to near-singular covariance matrix")


# Variable Confidence Thresholds
def test_klmetric_varying_confidence_thresholds():
    """Test KLMetric with varying confidence_threshold values."""
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
        generator = KLMetric(confidence_threshold=threshold)
        metric = generator.compute_over_time_v2(measured_states, [0, 1], truth_states, [0, 1])
        results.append({
            'threshold': threshold,
            'Average KL Divergence': metric.value['Average KL Divergence'],
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

# Non-Gaussian States Handling
def test_klmetric_with_particle_states():
    """Test KLMetric with ParticleState inputs."""
    generator = KLMetric()
    time = datetime.datetime.now()

    # Create ParticleState instances with state_vector set to None
    measured_state = ParticleState(
        state_vector=None,
        particle_list=[Particle(state_vector=StateVector([[0]]), weight=1.0)],
        timestamp=time
    )
    truth_state = ParticleState(
        state_vector=None,
        particle_list=[Particle(state_vector=StateVector([[0]]), weight=1.0)],
        timestamp=time
    )

    with pytest.raises(TypeError) as excinfo:
        generator.compute_over_time_v2([measured_state], [0], [truth_state], [0])
    assert "States must be instances of GaussianState." in str(excinfo.value)


#  Stress Tests
#  Test with Large Datasets

def test_klmetric_large_dataset():
    """Test KLMetric with a large dataset."""
    generator = KLMetric()
    time = datetime.datetime.now()
    num_states = 100 # This could be increased and it would work, but would take longer to run.

    measured_states = [
        GaussianState(
            state_vector=StateVector(np.random.rand(3, 1)),
            covar=np.eye(3),
            timestamp=time
        ) for _ in range(num_states)
    ]
    truth_states = [
        GaussianState(
            state_vector=StateVector(np.random.rand(3, 1)),
            covar=np.eye(3),
            timestamp=time
        ) for _ in range(num_states)
    ]

    metric = generator.compute_over_time_v2(measured_states, list(range(num_states)), truth_states,
                                            list(range(num_states)))
    kl_divergence = metric.value['Average KL Divergence']
    assert kl_divergence >= 0, "KL divergence should be non-negative"
