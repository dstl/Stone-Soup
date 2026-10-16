#!/usr/bin/env python

"""
UAV Tracking Demonstration
==========================
"""
# %%
# Overview

# Items to note:
#
# Assumes a single target track, which simplifies track management.
# There is no clutter, and no missed detections. So 1:1 Data Association.
# Need an initiator and deleter for the tracker.
# GPS updates are 1 sec., we assume radar revisit is the same.
# We are assuming a ground based radar:
#
# - Radar has course elevation resolution and fine bearing resolution.
# - Use range standard deviation of 3.14 m as a replacement
#   for range resolution.


# %%
# Setup: transition model, measurement model, updater and predictor
# -----------------------------------------------------------------

from GP import GaussianProcess
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sensor_network import GP_Sensor
from stonesoup.tracker.simple import SingleTargetTracker
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.measures import Euclidean
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.types.state import GaussianState, State
from stonesoup.simulator.platform import PlatformDetectionSimulator
from stonesoup.sensor.radar.radar import RadarElevationBearingRange
from stonesoup.platform.base import FixedPlatform
from stonesoup.feeder.geo import LLAtoENUConverter
from stonesoup.reader.generic import CSVGroundTruthReader
import numpy as np
from stonesoup.models.transition.linear import (
    ConstantVelocity,
    CombinedLinearGaussianTransitionModel
)
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.models.measurement.nonlinear import (
    CartesianToElevationBearingRange
)
from stonesoup.types.array import CovarianceMatrix


transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(1.0),
     ConstantVelocity(1.0),
     ConstantVelocity(1.0)])

# Model coords = elev, bearing, range. Angles in radians
meas_covar = np.diag([np.radians(np.sqrt(10.0))**2,
                      np.radians(0.6)**2,
                      3.14**2])

meas_covar_trk = CovarianceMatrix(1.0*meas_covar)
meas_model = CartesianToElevationBearingRange(
    ndim_state=6,
    mapping=np.array([0, 2, 4]),
    noise_covar=meas_covar_trk)
predictor = ExtendedKalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model=meas_model)

# %%
# Setup CSV reader & feeder


ground_truth_reader = CSVGroundTruthReader(
    path='UAV_Rot.csv',
    state_vector_fields=['longitude', 'Vx m/s',
                         'latitude', 'Vy m/s', 'altitude (m)'],
    time_field='time',
    path_id_field='groupNb',
)

sensor_location = [-30.948, 50.297311666, 0]  # Radar position [long, lat, alt]
ground_truth_reader = LLAtoENUConverter(
    ground_truth_reader, sensor_location, [0, 2, 4])

# %%
# Define Sensor, Platform and Detector
# ------------------------------------
# The sensor converts the Cartesian coordinates into range,
# bearing and elevation.
# The sensor is then mounted onto a platform (stationary in this case)


sensor = RadarElevationBearingRange(
    position_mapping=[0, 2, 4],
    noise_covar=meas_covar,
    ndim_state=6,
)
platform = FixedPlatform(
    State([0, 0, 0, 0, 0, 0]),  # Sensor at reference point, zero velocity
    position_mapping=[0, 2, 4],
    sensors=[sensor]
)

# Create the detector and initialize it.
detector = PlatformDetectionSimulator(ground_truth_reader, [platform])


# %%
# Setup Initiator class for the Tracker
# ---------------------------------------
# This is just an heuristic initiation:
# Assume most of the deviation is caused by the Bearing measurement error.
# This is then converted into x, y components using the target bearing. For the
# deviation in z,
# we simply use :math:`R\times\sigma_{elev}` (ignore any bearing and range
# deviation components). Velocity covariances are simply based on the expected
# velocity range of the targets.
#
# **NOTE** - The Extended Kalman filter can be very sensitive to the state
# initiation. Using the default :class:`~.SimpleMeasurementInitiator`,
# the estimates tended to diverge over the course of the track when larger
# bearing measurement covariances were used.


class Initiator(SimpleMeasurementInitiator):
    def initiate(self, detections, timestamp, **kwargs):
        MAX_DEV = 400.
        tracks = set()
        measurement_model = self.measurement_model
        for detection in detections:
            state_vector = measurement_model.inverse_function(
                detection)
            model_covar = measurement_model.covar()

            el_az_range = np.sqrt(np.diag(model_covar))  # elev, az, range

            std_pos = detection.state_vector[2, 0]*el_az_range[1]
            stdx = np.abs(std_pos*np.sin(el_az_range[1]))
            stdy = np.abs(std_pos*np.cos(el_az_range[1]))
            stdz = np.abs(detection.state_vector[2, 0]*el_az_range[0])
            if stdx > MAX_DEV:
                print('Warning - X Deviation exceeds limit!!')
            if stdy > MAX_DEV:
                print('Warning - Y Deviation exceeds limit!!')
            if stdz > MAX_DEV:
                print('Warning - Z Deviation exceeds limit!!')
            C0 = np.diag(np.array([stdx, 30.0, stdy, 30.0, stdz, 30.0])**2)

            tracks.add(Track([GaussianStateUpdate(
                state_vector,
                C0,
                SingleHypothesis(None, detection),
                timestamp=detection.timestamp)
            ]))
        return tracks


prior_state = GaussianState(
    np.array([[0], [0], [0], [0], [0], [0]]),
    np.diag([0, 30.0, 0, 30.0, 0, 30.0])**2)
initiator = Initiator(prior_state, meas_model)
# initiator = SimpleMeasurementInitiator(prior_state, meas_model)

# %%
# Setup Deleter for the Tracker
# -----------------------------
# In the simple case of 1 target, we never want to delete the track. Because
# this Deleter is so simple we haven't bothered using a subtype/inheritance
# and instead make use of Python's duck typing.


class MyDeleter:
    def delete_tracks(self, tracks):
        return set()


deleter = MyDeleter()

# %%
# Setup Hypothesiser and Associator
# ---------------------------------
# Since we know there is only one measurement per scan, we can just use the
# :class:`~.NearestNeighbour` associator to achieve our desired result.
meas = Euclidean()
hypothesiser = DistanceHypothesiser(predictor, updater, meas)
associator = NearestNeighbour(hypothesiser)


tracker = SingleTargetTracker(initiator,
                              deleter,
                              detector,
                              associator,
                              updater)

# %%
# Run the Tracker
# ---------------------------------
# We extract the ground truth from the detector
# and then run the tracker.
# While running the tracker we:
#
# - Extract the measurement that is associated with it.
# - Extract the position components of the estimated state vector.
#
# This allows us to plot the measurements, ground truth, and state estimates.
#
# **Note:** The meas_model.inverse_function() returns a state vector, which
# for our CV model consists of [x, vx, y, vy, z, vz].
est_X = []
est_Y = []
meas_X = []
meas_Y = []
true_X = []
true_Y = []
for time, tracks in tracker:
    for ground_truth in ground_truth_reader.groundtruth_paths:
        true_X.append(ground_truth.state_vector[0])
        true_Y.append(ground_truth.state_vector[2])

    # Because this is a single target tracker, I know there is only 1 track.
    for track in tracks:

        # Get the corresponding measurement
        detection = track.states[-1].hypothesis.measurement
        # Convert measurement into xy
        xyz = meas_model.inverse_function(detection)
        meas_X.append(xyz[0])
        meas_Y.append(xyz[2])

        vec = track.states[-1].state_vector
        est_X.append(vec[0])


def downsample_data(x, y, target_points=100):
    x = np.array(x)
    y = np.array(y)
    if len(x) <= target_points:
        return x, y

    indices = np.linspace(0, len(x) - 1, num=target_points, dtype=int)
    x_downsampled = x[indices]
    y_downsampled = y[indices]

    return x_downsampled, y_downsampled


T = 600
true_X1, true_Y1 = downsample_data(true_X, true_Y, T)
meas_X1, meas_Y1 = downsample_data(meas_X, meas_Y, T)
Xm = np.array(meas_X1).reshape(-1, 1)
Ym = np.array(meas_Y1).reshape(-1, 1)


def sliding_window(t, window_size):
    if window_size <= 0:
        raise ValueError("Invalid window size1")
    if window_size > t:
        start_time = 0
    else:
        start_time = t-window_size + 1
    return start_time


def tracking_DGP(T, time_data, x_train, y_train):
    x_data = []
    x_cov = []
    y_data = []
    y_cov = []
    for i in range(N, T):
        X_test_dgp = i
        # print(i)
        time_data_filtered = []
        x_data_filtered = []
        y_data_filtered = []
        SW = sliding_window(i, window_size)
        # print(i)
        for sensor_id in range(len(y_train)):
            indices_to_keep = np.where(
                (np.array(time_data[sensor_id]) >= SW) &
                (np.array(time_data[sensor_id]) < i+1))[0]
            time_data_filtered.append(np.array(time_data[sensor_id])[
                                      indices_to_keep].reshape(-1, 1))
            x_filtered = np.array(x_train[sensor_id])[
                indices_to_keep].reshape(-1, 1)

            x_noise = np.random.normal(0.1, 1, x_filtered.shape)
            # Replace mu and sigma with your chosen values
            x_filtered += x_noise
            x_data_filtered.append(x_filtered)

            y_filtered = (np.array(y_train[sensor_id])[
                indices_to_keep].reshape(-1, 1))
            y_noise = np.random.normal(0.1, 1, y_filtered.shape)
            # Replace mu and sigma with your chosen values
            y_filtered += y_noise
            y_data_filtered.append(y_filtered)

        time_dataF = [np.array(data)
                      for data in time_data_filtered if len(data) >= 3]
        x_trainF = [np.array(data)
                    for data in x_data_filtered if len(data) >= 3]
        y_trainF = [np.array(data)
                    for data in y_data_filtered if len(data) >= 3]

        mu_x, cov_x = update_DGP(x_trainF, time_dataF, X_test_dgp)
        # print(mu_x)

        x_data.append(mu_x)
        x_cov.append(cov_x)
        # x_data = np.vstack((x_data,mu_x))
        # x_cov.vstack(cov_x)

        mu_y, cov_y = update_DGP(y_trainF, time_dataF, X_test_dgp)
        y_data.append(mu_y)
        y_cov.append(cov_y)

    x_data = np.array(x_data).reshape(-1)
    x_cov = np.array(x_cov).reshape(-1)
    y_data = np.array(y_data).reshape(-1)
    y_cov = np.array(y_cov).reshape(-1)

    return x_data, x_cov, y_data, y_cov


def update_DGP(Data_train, Time_train, test_t):
    G = GaussianProcess(kernel_type='SE')
    Time_test = np.array([test_t]).reshape(-1, 1)
    res = G.fit(Time_train, Data_train, flag='DGP')
    l_opt_dgp, sigma_f_opt_dgp, sigma_no_dgp = res.x
    mu_sd, cov_sd = G.distributed_posterior(
        Time_test, Time_train, Data_train, length_scale=l_opt_dgp,
        sigma_f=sigma_f_opt_dgp, sigma_y=sigma_no_dgp
    )
    mu_sa, cov_sa = G.aggregation(mu_sd, cov_sd, Time_test)

    return mu_sa, cov_sa


N = 10
window_size = 30
# Generate sensors and record data


def update(Data_train, Time_train, test_t):
    G = GaussianProcess(kernel_type='SE')
    Time_test = np.array([test_t]).reshape(-1, 1)
    res = G.fit(Time_train, Data_train, flag='GP')
    l_opt, sigma_f_opt, sigma_no = res.x
    mu, cov = G.posterior(Time_test, Time_train, Data_train,
                          length_scale=l_opt, sigma_f=sigma_f_opt,
                          sigma_y=sigma_no)
    return mu, cov


x_data = []
y_data = []
x_cov = []
y_cov = []

rmse_x_list = []
rmse_y_list = []
rmsed_x_list = []

rmsed_y_list = []
Xm = np.array(Xm).reshape(-1, 1)
Ym = np.array(Ym).reshape(-1, 1)


SW = GP_Sensor()

seed = 113
num_sensors = 12
min_distance = 250
range_value = 350
xrange = (2200, 4200)
yrange = (-250, 250)
sensor_data = SW.create_sensor_network_plot(num_sensors, range_value,
                                            min_distance, xrange, yrange, seed)
time_data1, x_data1, y_data1 = SW.track_targetDGP(Xm, Ym, sensor_data)
true_X1 = true_X1[N:]
true_Y1 = true_Y1[N:]
meas_X1 = meas_X1[N:]
meas_Y1 = meas_Y1[N:]

EN = 1
Error_x_gp = [None] * EN
Error_x_dgp = [None] * EN
Error_y_dgp = [None] * EN
Error_y_gp = [None] * EN

Error_y_gp = [None] * EN

for experiment in range(EN):

    print(experiment)
    x_data = []
    y_data = []
    x_cov = []
    y_cov = []

    for i in range(N, T):
        SW = sliding_window(i, window_size)
        Time_train = np.arange(SW, i).reshape(i-SW, 1)
        X_train = Xm[sliding_window(i, window_size):i]
        mu_x, cov_x = update(X_train, Time_train, i)
        x_data.append(mu_x)
        x_cov.append(cov_x)
        # print(i)
        Y_train = Ym[sliding_window(i, window_size):i]
        mu_y, cov_y = update(Y_train, Time_train, i)
        y_data.append(mu_y)
        y_cov.append(cov_y)

    x_data = np.array(x_data).reshape(-1, 1)
    y_data = np.array(y_data).reshape(-1, 1)
    x_cov = np.array(x_cov).reshape(-1, 1)
    y_cov = np.array(y_cov).reshape(-1, 1)

    xd_data, xd_cov, yd_data, yd_cov = tracking_DGP(T, time_data1,
                                                    x_data1, y_data1)

    xd_data = np.array(xd_data).reshape(-1, 1)
    yd_data = np.array(yd_data).reshape(-1, 1)
    xd_cov = np.array(xd_cov).reshape(-1, 1)
    yd_cov = np.array(yd_cov).reshape(-1, 1)

    ####
    # RMSE
    rmse_x = mean_squared_error(true_X1, x_data, squared=False)
    rmse_y = mean_squared_error(true_Y1, y_data, squared=False)
    rmse_x_list.append(rmse_x)
    rmse_y_list.append(rmse_y)

    rmsed_x = mean_squared_error(true_X1, xd_data, squared=False)
    rmsed_y = mean_squared_error(true_Y1, yd_data, squared=False)
    rmsed_x_list.append(rmsed_x)
    rmsed_y_list.append(rmsed_y)

    true_Y1_reshaped = true_Y1.reshape(-1, 1)
    y_data_reshaped = y_data.reshape(-1, 1)
    yd_data_reshaped = yd_data.reshape(-1, 1)
    Error_y_gp[experiment] = abs(true_Y1_reshaped - y_data_reshaped)
    Error_y_dgp[experiment] = abs(true_Y1_reshaped - yd_data_reshaped)

    true_X1_reshaped = true_X1.reshape(-1, 1)
    x_data_reshaped = x_data.reshape(-1, 1)
    xd_data_reshaped = xd_data.reshape(-1, 1)
    Error_x_gp[experiment] = abs(true_X1_reshaped - x_data_reshaped)
    Error_x_dgp[experiment] = abs(true_X1_reshaped - xd_data_reshaped)

fig = plt.figure(figsize=(10, 6))
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(true_X1, true_Y1, 'd-k', label='Truth', markerfacecolor='None')
# ax2.plot(true_X1, true_Y1, 'd-k', label='Truth', markerfacecolor='None')
ax2.plot(meas_X1, meas_Y1, 'xb', label='Measurements')
ax2.plot(x_data, y_data, 'r.', label='Estimates')
plt.xlim(2000, 4500)
plt.ylim(-300, 300)
ax2.set_xlabel('X (m)', fontsize=15)
ax2.set_ylabel('Y (m)', fontsize=15)
# ax2.set_title('Meansurements and Groundtruth', fontsize=20)
ax2.legend(fontsize=12)

fig = plt.figure(figsize=(10, 6))
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(true_X1, true_Y1, 'd-k', label='Truth', markerfacecolor='None')
# ax2.plot(true_X1, true_Y1, 'd-k', label='Truth', markerfacecolor='None')
ax2.plot(x_data, y_data, 'r.', label='Estimates')
plt.xlim(2000, 4500)
plt.ylim(-300, 300)
ax2.set_xlabel('X (m)', fontsize=15)
ax2.set_ylabel('Y (m)', fontsize=15)
# ax2.set_title('GP Tracking Estimates', fontsize=20)
ax2.legend(fontsize=12)


fig, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(true_X1, true_Y1, 'd-k', label='Truth', markerfacecolor='None')
# ax2.plot(meas_X1, meas_Y1, 'xb', label='Measurements')
ax2.plot(xd_data, yd_data, 'r.', label='DGP Estimates')

# Create custom legend entries
sensor_legend = mpatches.Patch(color='orange', label='Sensor')
range_legend = mpatches.Patch(color='blue', alpha=0.1, label='Sensor Range')
sensor_range = range_value
# Plot each sensor's position and range
for sensor in sensor_data:
    position = sensor["position"]
    ax2.scatter(position[0], position[1], c='orange', edgecolor='black')
    sensor_circle = plt.Circle((position[0], position[1]), sensor_range,
                               color='blue', alpha=0.8, fill=False)
    ax2.add_patch(sensor_circle)

# Only label the first sensor and first range to avoid duplicate labels
ax2.scatter([], [], c='orange', label='Sensor', edgecolor='black')
# Invisible point for legend
ax2.add_patch(mpatches.Circle((0, 0), 1, edgecolor='blue', facecolor='none',
                              alpha=0.8, label='Detection Range'))
plt.xlim(2000, 4500)
plt.ylim(-300, 300)
ax2.set_xlabel('X (m)', fontsize=15)
ax2.set_ylabel('Y (m)', fontsize=15)

# Extract existing handles and labels
handles, labels = ax2.get_legend_handles_labels()

# Add the custom legend entries
handles.extend([sensor_legend, range_legend])

# Create the legend with the custom and existing entries
ax2.legend(handles=handles, labels=labels, fontsize=15)

# ax2.set_title('Sensor Positions and DGP Estimates', fontsize=20)

plt.show()

fig = plt.figure(figsize=(10, 6))
ax2 = fig.add_subplot(1, 1, 1)

# Plot the data
ax2.plot(true_X1, 'd-k', label='True Trajectory', markerfacecolor='None',
         markersize=5)
ax2.plot(xd_data, 'r.', label='DGP Estimated Trajectory', markersize=10)
ax2.plot(x_data, 'b.', label='GP Estimated Trajectory', markersize=10)
# Beautifying the plot
ax2.set_xlabel('Time (ms)', fontsize=15)
ax2.set_ylabel('X Position (m)', fontsize=15)
# ax2.set_title('X-Axis Trajectory Estimation', fontsize=20, pad=20)

# Set the legend to not overlap with the data
ax2.legend(loc='upper center', frameon=False, fontsize=15)

# Optionally, set a grid for better readability
ax2.grid(True, linestyle='--', alpha=0.5)

# Tight layout often improves the spacing between subplots
plt.tight_layout()

# Save the figure with a high dpi
plt.savefig('X_Axis_Trajectory_Estimation.png', dpi=300)

plt.show()


fig = plt.figure(figsize=(10, 6))
ax2 = fig.add_subplot(1, 1, 1)

# Plot the data
ax2.plot(true_Y1, 'd-k', label='True Trajectory', markerfacecolor='None',
         markersize=8)
ax2.plot(yd_data, 'r.', label='DGP Estimated Trajectory', markersize=8)
ax2.plot(y_data, 'b.', label='GP Estimated Trajectory', markersize=8)
# Beautifying the plot
ax2.set_xlabel('Time (ms)', fontsize=15)
ax2.set_ylabel('Y Position (m)', fontsize=15)
# ax2.set_title('Y-Axis Trajectory Estimation', fontsize=20, pad=20)

# Set the legend to not overlap with the data
ax2.legend(loc='upper right', frameon=False, fontsize=15)

# Optionally, set a grid for better readability
ax2.grid(True, linestyle='--', alpha=0.5)

# Tight layout often improves the spacing between subplots
plt.tight_layout()

# Save the figure with a high dpi
plt.savefig('Y_Axis_Trajectory_Estimation.png', dpi=300)

plt.show()


rmse_x = mean_squared_error(true_X1, x_data, squared=False)
rmse_y = mean_squared_error(true_Y1, y_data, squared=False)
print('GPX=', rmse_x)
print('GPY=', rmse_y)


rmse_x = mean_squared_error(true_X1, xd_data, squared=False)
rmse_y = mean_squared_error(true_Y1, yd_data, squared=False)
print('DGPX=', rmse_x)
print('DGPY=', rmse_y)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plotting the data over time
ax1.plot(true_Y1, 'd-k', label='True Trajectory', markerfacecolor='None',
         markersize=8)
ax1.plot(y_data, '.', label='GP Estimated Trajectory', markersize=8)
ax1.plot(yd_data, '.', label='DGP Estimated Trajectory', markersize=8)

# Beautifying the plot
ax1.set_xlabel('Time (ms)', fontsize=15)
ax1.set_ylabel('Y Position (m)', fontsize=15)
# ax2.set_title('Y-Axis Trajectory Estimation', fontsize=20, pad=20)

# Set the legend to not overlap with the data
ax1.legend(loc='lower right', frameon=False, fontsize=15)

# Optionally, set a grid for better readability
ax1.grid(True, linestyle='--', alpha=0.5)


# Plotting the RMSE over time
# For demonstration, we'll just plot a constant RMSE value as a straight line

ax2.plot(Error_y_gp[0], label='Error in GP', markersize=8)
ax2.plot(Error_y_dgp[0], label='Error in DGP', markersize=8)
ax2.set_xlabel('Time (ms)', fontsize=15)
ax2.set_ylabel('RMSE', fontsize=15)
ax2.legend()
ax2.grid(True)

# Display the plots
plt.tight_layout()
plt.show()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plotting the data over time
ax1.plot(true_X1, 'd-k', label='True Trajectory', markerfacecolor='None',
         markersize=8)
ax1.plot(x_data, '.', label='GP Estimated Trajectory', markersize=8)
ax1.plot(xd_data, '.', label='DGP Estimated Trajectory', markersize=8)

# Beautifying the plot
ax1.set_xlabel('Time (ms)', fontsize=15)
ax1.set_ylabel('X Position (m)', fontsize=15)

# Set the legend to not overlap with the data
ax1.legend(loc='upper center', frameon=False, fontsize=15)

# Optionally, set a grid for better readability
ax1.grid(True, linestyle='--', alpha=0.5)


# Plotting the RMSE over time
# For demonstration, we'll just plot a constant RMSE value as a straight line
ax2.plot(Error_x_gp[0], label='Error in GP', markersize=8)
ax2.plot(Error_x_dgp[0], label='Error in DGP', markersize=8)
ax2.set_xlabel('Time (ms)', fontsize=15)
ax2.set_ylabel('RMSE', fontsize=15)
ax2.legend()
ax2.grid(True)

# Display the plots
plt.tight_layout()
plt.show()
