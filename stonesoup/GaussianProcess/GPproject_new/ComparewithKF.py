# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 02:55:59 2025

@author: 007
"""
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.groundtruth import GroundTruthState
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from gaussianprocess import GaussianProcess
from GaussianProcessPredictor import GaussianProcessPredictor
from Generator import generate_groundtruth
import matplotlib.pyplot as plt
# Step 1: Define kernel function
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    Radial Basis Function (RBF) Kernel
    """
    sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# Step 2: Configure parameters
kernel = rbf_kernel
num_points = 100
win_size = 20
groundtruth_noise_std = 1
measurement_noise_std = 10
selected_dim = "x,y"
start_time = datetime.now()

# Step 3: Generate groundtruth and measurements
groundtruth_x, groundtruth_y = generate_groundtruth(num_points, mode="S1", noise_std=groundtruth_noise_std)

# Add measurement noise to create measurements
measurement_x = groundtruth_x + np.random.normal(0, measurement_noise_std, num_points)
measurement_y = groundtruth_y + np.random.normal(0, measurement_noise_std, num_points)

timestamps = [start_time + timedelta(seconds=i) for i in range(num_points)]

# Wrap Ground Truth and Measurement data in GroundTruthState objects
groundtruth_states = [
    GroundTruthState(np.array([x, y]).reshape(-1, 1), timestamp=t)
    for x, y, t in zip(groundtruth_x, groundtruth_y, timestamps)
]

measurements = [
    GroundTruthState(np.array([x, y]).reshape(-1, 1), timestamp=t)
    for x, y, t in zip(measurement_x, measurement_y, timestamps)
]

# Step 4: Initialize Kalman Filter
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05), ConstantVelocity(0.05)])
measurement_model = LinearGaussian(ndim_state=4, mapping=(0, 2), noise_covar=np.diag([measurement_noise_std**2] * 2))

initial_state = GaussianState(
    state_vector=np.array([[measurement_x[0]], [0], [measurement_y[0]], [0]]),
    covar=np.diag([1, 1, 1, 1]),
    timestamp=timestamps[0]
)

predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)

# Step 5: Sliding window logic and GP prediction
gp = GaussianProcess(kernel=kernel)
gp_predictions = []
gp_prediction_timestamps = []

for i in range(5, len(timestamps)):  # Start from the 5th data point
    start_index = max(0, i - win_size)
    window_states = measurements[start_index:i]
    gp.train(window_states, win_size)

    predictor_gp = GaussianProcessPredictor(gp)
    next_timestamp = timestamps[i]
    test_prediction = predictor_gp.predict([next_timestamp], selected_dim)
    gp_predictions.append((test_prediction[0]["mean"].flatten()[0], test_prediction[1]["mean"].flatten()[0]))
    gp_prediction_timestamps.append(next_timestamp)

# Step 6: Run Kalman Filter
track = Track()
current_state = initial_state
for i, timestamp in enumerate(timestamps):
    prediction = predictor.predict(current_state, timestamp=timestamp)
    detection = Detection(np.array([[measurement_x[i]], [measurement_y[i]]]), timestamp=timestamp)
    hypothesis = SingleHypothesis(prediction, detection)
    current_state = updater.update(hypothesis)
    track.append(current_state)

kf_filtered_x = [state.state_vector[0, 0] for state in track]
kf_filtered_y = [state.state_vector[2, 0] for state in track]

# Step 7: Visualization
# fig, ax = plt.subplots(figsize=(8, 8))
# gt_line, = ax.plot([], [], 'k--', label="Ground Truth")
# gp_pred_line, = ax.plot([], [], 'ro', label="GP Predictions")
# kf_pred_line, = ax.plot([], [], 'go', label="Kalman Filter")
# measurement_scatter = ax.scatter([], [], c='blue', alpha=0.6, label="Measurements")

# ax.set_xlim(min(groundtruth_x) - 1, max(groundtruth_x) + 1)
# ax.set_ylim(min(groundtruth_y) - 1, max(groundtruth_y) + 1)
# ax.set_xlabel("State X")
# ax.set_ylabel("State Y")
# ax.set_title("Gaussian Process vs Kalman Filter")
# ax.legend()

fig, ax = plt.subplots(figsize=(8, 8))
gt_line, = ax.plot([], [], 'k--', label="Ground Truth")
gp_pred_line, = ax.plot([], [], 'ro', label="GP Predictions")
kf_pred_line, = ax.plot([], [], 'go', label="Kalman Filter")
measurement_scatter = ax.scatter([], [], c='blue', alpha=0.6, label="Measurements")

ax.set_xlim(min(groundtruth_x) - 1, max(groundtruth_x) + 1)
ax.set_ylim(min(groundtruth_y) - 1, max(groundtruth_y) + 1)

# **手动加大字体**
ax.set_xlabel("State X", fontsize=16)
ax.set_ylabel("State Y", fontsize=16)

ax.legend(fontsize=14)  


def update(frame):
    if frame < len(timestamps):
        gt_line.set_data(groundtruth_x[:frame], groundtruth_y[:frame])

        if frame >= 5:
            gp_pred_x = [p[0] for p in gp_predictions[:frame - 5]]
            gp_pred_y = [p[1] for p in gp_predictions[:frame - 5]]
            gp_pred_line.set_data(gp_pred_x, gp_pred_y)

            kf_pred_x = kf_filtered_x[:frame]
            kf_pred_y = kf_filtered_y[:frame]
            kf_pred_line.set_data(kf_pred_x, kf_pred_y)
        
        measurement_scatter.set_offsets(np.c_[measurement_x[:frame], measurement_y[:frame]])
    else:
        gt_line.set_data(groundtruth_x, groundtruth_y)
        gp_pred_x = [p[0] for p in gp_predictions]
        gp_pred_y = [p[1] for p in gp_predictions]
        gp_pred_line.set_data(gp_pred_x, gp_pred_y)

        kf_pred_x = kf_filtered_x
        kf_pred_y = kf_filtered_y
        kf_pred_line.set_data(kf_pred_x, kf_pred_y)

        measurement_scatter.set_offsets(np.c_[measurement_x, measurement_y])
    return gt_line, gp_pred_line, kf_pred_line, measurement_scatter

ani = animation.FuncAnimation(fig, update, frames=len(timestamps) + 1, interval=100, blit=True, repeat=False)
plt.show()

# Step 8: Compute RMSE
rmse_gp_x = np.sqrt(np.mean((np.array([p[0] for p in gp_predictions]) - groundtruth_x[5:])**2))
rmse_gp_y = np.sqrt(np.mean((np.array([p[1] for p in gp_predictions]) - groundtruth_y[5:])**2))
rmse_kf_x = np.sqrt(np.mean((np.array(kf_filtered_x) - groundtruth_x)**2))
rmse_kf_y = np.sqrt(np.mean((np.array(kf_filtered_y) - groundtruth_y)**2))

print(f"RMSE for GP (X): {rmse_gp_x}")
print(f"RMSE for GP (Y): {rmse_gp_y}")
print(f"RMSE for Kalman Filter (X): {rmse_kf_x}")
print(f"RMSE for Kalman Filter (Y): {rmse_kf_y}")
