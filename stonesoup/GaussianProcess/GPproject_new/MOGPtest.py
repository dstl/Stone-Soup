import numpy as np
from datetime import datetime, timedelta
from stonesoup.types.groundtruth import GroundTruthState
from Generator import generate_groundtruth
from MOGP import MultiOutputGP
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Step 1: Define kernel functions
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    Radial Basis Function (RBF) Kernel for single dimension.
    """
    sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

def multi_output_kernel(X1, X2, length_scale=1.0, sigma_f=1.0, corr=0.5):
    """
    Multi-output kernel for joint GP modeling of x and y.
    """
    base_kernel = rbf_kernel(X1, X2, length_scale=length_scale, sigma_f=sigma_f)
    coregionalization_matrix = np.array([[1.0, corr], [corr, 1.0]])  # Correlation between outputs
    return np.kron(coregionalization_matrix, base_kernel)

# Step 3: Generate groundtruth and measurements using S1 mode
num_points = 100
win_size = 20  # Sliding window size
groundtruth_noise_std = 1
measurement_noise_std = 10
start_time = datetime.now()

groundtruth_x, groundtruth_y = generate_groundtruth(num_points, mode="S3", noise_std=groundtruth_noise_std)
measurement_x = groundtruth_x + np.random.normal(0, measurement_noise_std, num_points)
measurement_y = groundtruth_y + np.random.normal(0, measurement_noise_std, num_points)

timestamps = [start_time + timedelta(seconds=i) for i in range(num_points)]

# Step 4: Sliding window logic and prediction with MOGP
mogp_predictions = []
mogp_prediction_timestamps = []

for i in range(5, num_points):  # Start after having enough points for a sliding window
    start_index = max(0, i - win_size)
    window_timestamps = [t.timestamp() for t in timestamps[start_index:i]]
    window_states = np.hstack([measurement_x[start_index:i], measurement_y[start_index:i]])

    # Train MOGP on the current window
    mogp = MultiOutputGP(kernel=multi_output_kernel)
    mogp.train(window_timestamps, window_states)

    # Predict for the next timestamp
    next_timestamp = timestamps[i].timestamp()
    mu_s, _ = mogp.posterior([next_timestamp])
    mogp_predictions.append((mu_s[0, 0], mu_s[1, 0]))
    mogp_prediction_timestamps.append(timestamps[i])

# Step 5: Animation
fig, ax = plt.subplots(figsize=(8, 8))
gt_line, = ax.plot([], [], 'k--', label="Ground Truth")
mogp_line, = ax.plot([], [], 'r-', label="MOGP Predictions")
measurement_scatter = ax.scatter([], [], c='blue', alpha=0.6, label="Measurements")

ax.set_xlim(min(groundtruth_x) - 1, max(groundtruth_x) + 1)
ax.set_ylim(min(groundtruth_y) - 1, max(groundtruth_y) + 1)
ax.set_xlabel("State X")
ax.set_ylabel("State Y")
ax.legend()

def update(frame):
    if frame < len(timestamps):
        gt_line.set_data(groundtruth_x[:frame], groundtruth_y[:frame])

        if frame >= 5:
            mogp_pred_x = [p[0] for p in mogp_predictions[:frame - 5]]
            mogp_pred_y = [p[1] for p in mogp_predictions[:frame - 5]]
            mogp_line.set_data(mogp_pred_x, mogp_pred_y)

        measurement_scatter.set_offsets(np.c_[measurement_x[:frame], measurement_y[:frame]])
    return gt_line, mogp_line, measurement_scatter

ani = animation.FuncAnimation(fig, update, frames=num_points + 1, interval=100, blit=True, repeat=False)
plt.show()

# Step 6: Compute RMSE
aligned_groundtruth_x = groundtruth_x[5:]  # Align groundtruth with MOGP predictions
aligned_groundtruth_y = groundtruth_y[5:]
mogp_pred_x = np.array([p[0] for p in mogp_predictions])
mogp_pred_y = np.array([p[1] for p in mogp_predictions])

rmse_x = np.sqrt(np.mean((mogp_pred_x - aligned_groundtruth_x)**2))
rmse_y = np.sqrt(np.mean((mogp_pred_y - aligned_groundtruth_y)**2))
print(f"RMSE for MOGP (X): {rmse_x}")
print(f"RMSE for MOGP (Y): {rmse_y}")
