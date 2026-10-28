from stonesoup.types.groundtruth import GroundTruthState
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from gaussianprocess import GaussianProcess
from GaussianProcessPredictor import GaussianProcessPredictor
from Generator import generate_groundtruth  # 导入新的 Ground Truth 函数

# Step 1: Define kernel function
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    Radial Basis Function (RBF) Kernel
    """
    sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# Step 2: Configure parameters
kernel = rbf_kernel
num_train_points = 20  # Number of training points
num_test_points = 5  # Number of test points
win_size = 50  # Sliding window size in seconds
selected_dim = "x,y"  # Options: "x", "y", "x,y"
start_time = datetime.now()

# Noise configuration
noise_mean = 0.0  # Mean of the noise
noise_std = 0.05  # Standard deviation of the noise

# Step 3: Generate training and test data
train_timestamps = [start_time + timedelta(seconds=i) for i in range(num_train_points)]
test_timestamps = [start_time + timedelta(seconds=i) for i in range(num_train_points, num_train_points + num_test_points)]

# Generate Ground Truth using the new function
trajectory_x, trajectory_y = generate_groundtruth(num_train_points + num_test_points, mode="S4", noise_std=noise_std)

# Split trajectory into training and test sets
train_states = [[trajectory_x[i], trajectory_y[i]] for i in range(num_train_points)]
groundtruth_states = [
    GroundTruthState(np.array(state).reshape(-1, 1), timestamp=t)
    for state, t in zip(train_states, train_timestamps)
]

# Step 4: Train Gaussian Process model
gp = GaussianProcess(kernel=kernel)
gp.train(groundtruth_states, win_size)

# Step 5: Predict for test timestamps
predictor = GaussianProcessPredictor(gp)
test_predictions = predictor.predict(test_timestamps, selected_dim)

# Step 6: Plot results including training data, GP predictions, and Ground Truth
plt.figure(figsize=(12, 6))

# Convert timestamps to POSIX for uniformity
train_posix_timestamps = np.array([t.timestamp() for t in train_timestamps])
test_posix_timestamps = np.array([t.timestamp() for t in test_timestamps])
all_posix_timestamps = np.concatenate((train_posix_timestamps, test_posix_timestamps))

# Extract Ground Truth values
groundtruth_x = trajectory_x
groundtruth_y = trajectory_y

if selected_dim in ["x", "y"]:
    # Single dimension: x or y
    dim_index = 0 if selected_dim == "x" else 1

    # Extract training data
    train_states_dim = [state.state_vector[dim_index, 0] for state in groundtruth_states]

    # Predict GP for training data
    train_predictions = predictor.predict(train_timestamps, selected_dim)
    train_mean = train_predictions[dim_index]["mean"].flatten()
    train_std_dev = np.sqrt(np.diag(train_predictions[dim_index]["covariance"]))

    # Predict GP for test data
    test_predictions = predictor.predict(test_timestamps, selected_dim)
    test_mean = test_predictions[dim_index]["mean"].flatten()
    test_std_dev = np.sqrt(np.diag(test_predictions[dim_index]["covariance"]))

    # Plot training data
    plt.plot(train_posix_timestamps, train_states_dim, 'rx', label="Training Data")

    # Plot Ground Truth
    plt.plot(all_posix_timestamps, groundtruth_x if dim_index == 0 else groundtruth_y, 'k--', label="Ground Truth")

    # Plot GP predictions for training data
    plt.plot(train_posix_timestamps, train_mean, 'g-', label="GP Mean (Train)")
    plt.fill_between(
        train_posix_timestamps,
        train_mean - 1.96 * train_std_dev,
        train_mean + 1.96 * train_std_dev,
        color="green",
        alpha=0.2,
        label="95% Confidence Interval (Train)"
    )

    # Plot GP predictions for test data
    plt.plot(test_posix_timestamps, test_mean, 'b-', label="GP Mean (Test)")
    plt.fill_between(
        test_posix_timestamps,
        test_mean - 1.96 * test_std_dev,
        test_mean + 1.96 * test_std_dev,
        color="blue",
        alpha=0.2,
        label="95% Confidence Interval (Test)"
    )

    plt.xlabel("Time (POSIX seconds)")
    plt.ylabel(f"State ({selected_dim})")
    plt.title(f"Gaussian Process Regression for Dimension {selected_dim}")
    plt.legend()
    plt.grid(True)

elif selected_dim == "x,y":
    # Multi-dimension: x and y

    # Extract GP predictions
    train_predictions = predictor.predict(train_timestamps, selected_dim)
    train_mean_x = train_predictions[0]["mean"].flatten()
    train_mean_y = train_predictions[1]["mean"].flatten()
    test_predictions = predictor.predict(test_timestamps, selected_dim)
    test_mean_x = test_predictions[0]["mean"].flatten()
    test_mean_y = test_predictions[1]["mean"].flatten()

    # Extract training data for both dimensions
    train_states_x = [state.state_vector[0, 0] for state in groundtruth_states]
    train_states_y = [state.state_vector[1, 0] for state in groundtruth_states]

    # Plot training data
    plt.plot(train_states_x, train_states_y, 'rx', label="Training Data")

    # Plot Ground Truth
    plt.plot(groundtruth_x, groundtruth_y, 'k--', label="Ground Truth")

    # Plot GP predictions for training data
    plt.plot(train_mean_x, train_mean_y, 'g-', label="GP Mean (Train)")

    # Plot GP predictions for test data
    plt.plot(test_mean_x, test_mean_y, 'b-', label="GP Mean (Test)")

    plt.xlabel("State (x)")
    plt.ylabel("State (y)")
    plt.title("Gaussian Process Regression: x/y Relationship")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
