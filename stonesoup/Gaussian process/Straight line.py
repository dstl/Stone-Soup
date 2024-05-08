# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 07:39:25 2024

@author: 007
"""
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity

# Set the starting time of the simulation
start_time = datetime.now().replace(microsecond=0)
np.random.seed(1991)
q_x = 0.05
q_y = 0.05
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(q_x), ConstantVelocity(q_y)]
)
timesteps = [start_time]
num_steps = 30

# Lists to store error metrics for multiple experiments
rmse_x_list = []
rmse_y_list = []
rmsed_x_list = []
rmsed_y_list = []

for experiment in range(1):
    truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1],
                                              timestamp=timesteps[0])])

    # Generate truth data for the model
    for k in range(1, num_steps + 1):
        timesteps.append(start_time + timedelta(seconds=k))
        truth.append(GroundTruthState(
            transition_model.function(truth[k - 1], noise=True,
                                      time_interval=timedelta(seconds=1)),
            timestamp=timesteps[k]))

    # Initialize measurement model
    from stonesoup.types.detection import Detection
    from stonesoup.models.measurement.linear import LinearGaussian
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[5, 0], [0, 5]])
    )
    measurements = []
    for state in truth:
        measurement = measurement_model.function(state, noise=True)
        measurements.append(Detection(
            measurement, timestamp=state.timestamp,
            measurement_model=measurement_model
        ))

    # Gaussian Process Tracking
    from GP_Tracking import GP_track
    Xm = []
    Ym = []
    for measurement in measurements:
        Xm.append(measurement.state_vector[0])
        Ym.append(measurement.state_vector[1])
    Xm = np.array(Xm).reshape(-1, 1)
    Ym = np.array(Ym).reshape(-1, 1)

    # Prepare true positions for comparison
    Xt = []
    Yt = []
    for truth_state in truth:
        Xt.append(truth_state.state_vector[0])
        Yt.append(truth_state.state_vector[2])
    Xt = np.array(Xt).reshape(-1, 1)
    Yt = np.array(Yt).reshape(-1, 1)

    A = GP_track()
    Size_win = 30  # Define the sliding window size
    x_data, x_cov, y_data, y_cov = A.tracking(measurements, Size_win)

    # Calculate root mean square error
    from sklearn.metrics import mean_squared_error
    rmse_x = mean_squared_error(Xt[2:], x_data, squared=False)
    rmse_y = mean_squared_error(Yt[2:], y_data, squared=False)
    print('GPX=', rmse_x)
    print('gpy=', rmse_y)

    # Plot results
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(Xm, Ym, 'xb', label='Measurements', markerfacecolor='None')
    ax1.plot(Xt, Yt, 'd-k', label='Truth', markerfacecolor='None')
    ax1.plot(x_data, y_data, 'r.', label='Estimates')
    ax1.set_xlabel('X (m)', fontsize=15)
    ax1.set_ylabel('Y (m)', fontsize=15)
    ax1.legend(fontsize=15)

    plt.tight_layout()
    plt.show()
