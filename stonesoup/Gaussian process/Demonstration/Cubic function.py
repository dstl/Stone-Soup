# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 05:00:02 2024

@author: 007
"""
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
import numpy as np
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.plotter import Plotterly
rmse_x_list = []
rmse_y_list = []
rmsed_x_list = []
rmsed_y_list = []
for experiment in range(1):
    plotter = Plotterly()
    
    truth = GroundTruthPath()
    start_time = datetime.now()
    for n in range(1, 202, 2):
        x = n -100
        y = 1e-4 * (n-100)**3
        varxy = np.array([[0.1, 0.], [0., 0.1]])
        xy = np.random.multivariate_normal(np.array([x, y]), varxy)
        truth.append(GroundTruthState(np.array([[xy[0]], [xy[1]]]),
                                      timestamp=start_time + timedelta(seconds=n)))
    
    # Plot the result
    plotter.plot_ground_truths({truth}, [0, 1])
    plotter.fig
    
    from scipy.stats import multivariate_normal
    from stonesoup.types.detection import Detection
    from stonesoup.models.measurement.linear import LinearGaussian
    
    measurements = []
    for state in truth:
        x, y = multivariate_normal.rvs(
            state.state_vector.ravel(), cov=np.diag([10., 10.]))
        measurements.append(Detection(
            [x, y], timestamp=state.timestamp))
    
    # Plot the result
    plotter.plot_measurements(measurements, [0, 1], LinearGaussian(2, (0, 1), np.diag([0, 0])))
    plotter.fig
    
    
    from GP_Tracking import GP_track
    num_steps = 100
    
    Xm = [];Ym = []
    for k in range(num_steps-1):
        truth_x = [state.state_vector[0] for state in truth]
        truth_y = [state.state_vector[1] for state in truth]
    
    for measurement in measurements:
        Xm.append(measurement.state_vector[0])
        # Xt.append(measurement.timestamp)
        Ym.append(measurement.state_vector[1])
        # Yt.append(measurement.timestamp)   
    Xm = np.array(Xm).reshape(-1, 1) 
    Ym = np.array(Ym).reshape(-1, 1)   
    
    A = GP_track()
    Size_win = 100  #sliding Window
    x_data,x_cov , y_data,y_cov = A.tracking(measurements, Size_win)
    
    fig = plt.figure(figsize=(10, 6))
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.plot(Xm, Ym, 'xb', label='Measurements', markerfacecolor='None')
    ax2.plot(truth_x, truth_y, 'd-k', label='Truth', markerfacecolor='None')
    ax2.plot(x_data, y_data, 'r.', label='Estimates')
    # plt.xlim(200, 200)
    # plt.ylim(-100, 100)
    ax2.set_xlabel('X (m)',fontsize=15)
    ax2.set_ylabel('Y (m)',fontsize=15)
    # ax2.set_title('GP Tracking Estimates', fontsize=20)
    ax2.legend(fontsize=15) 
    truth_x = np.array(truth_x).reshape(-1, 1) 
    truth_y = np.array(truth_y).reshape(-1, 1) 
    from sklearn.metrics import mean_squared_error
    rmse_x = mean_squared_error(truth_x[2:], x_data, squared=False)
    rmse_y = mean_squared_error(truth_y[2:], y_data, squared=False)
    print('GPX =',rmse_x)
    print('GPY =',rmse_y)
    rmse_x_list.append(rmse_x)
    rmse_y_list.append(rmse_y)
    
    
    
    
    from sensor_network import GP_Sensor
    # Generate sensors and record data
    
    SW = GP_Sensor()
    
    seed = 233; num_sensors = 50 
    min_distance = 15; range_value = 30
    xrange = (-200, 200)
    yrange = (-100, 100)
    
    sensor_data= SW.create_sensor_network_plot(num_sensors, range_value, min_distance, xrange, yrange, seed)
    time_data1, x_data1, y_data1 = SW.track_targetDGP(Xm, Ym, sensor_data)
    
    x_data, x_cov , y_data, y_cov, x_s, y_s, t_s = A.tracking_DGP(measurements,time_data1, x_data1, y_data1)
    
    import matplotlib.patches as mpatches
    fig, ax2 = plt.subplots(figsize=(10, 6))
    sensor_range = range_value
    ax2.plot(Xm, Ym, 'xb', label='Measurements', markerfacecolor='None')
    ax2.plot(truth_x, truth_y, 'd-k', label='Truth', markerfacecolor='None')
    ax2.plot(x_data, y_data, 'r.', label='DGP Estimates')
    
    # Create custom legend entries
    sensor_legend = mpatches.Patch(color='orange', label='Sensor')
    range_legend = mpatches.Patch(color='blue', alpha=0.1, label='Sensor Range')
    
    # Plot each sensor's position and range
    for sensor in sensor_data:
        position = sensor["position"]
        ax2.scatter(position[0], position[1], c='orange', edgecolor='black')
        sensor_circle = plt.Circle((position[0], position[1]), sensor_range, color='blue', alpha=0.8, fill=False)
        ax2.add_patch(sensor_circle)
    
    # Only label the first sensor and first range to avoid duplicate labels
    ax2.scatter([], [], c='orange', label='Sensor', edgecolor='black')  # Invisible point for legend
    # ax2.add_patch(mpatches.Circle((0, 0), 1, color='blue', alpha=0.1, label='Sensor Range'))  # Invisible circle for legend
    ax2.add_patch(mpatches.Circle((0, 0), 1, edgecolor='blue', facecolor='none', alpha=0.8, label='Detection Range'))
    plt.xlim(-120, 105)
    plt.ylim(-120, 120)
    ax2.set_xlabel('X (m)',fontsize=15)
    ax2.set_ylabel('Y (m)',fontsize=15)
    
    # Extract existing handles and labels
    handles, labels = ax2.get_legend_handles_labels()
    
    # Add the custom legend entries
    handles.extend([sensor_legend, range_legend])
    
    # Create the legend with the custom and existing entries
    ax2.legend(handles=handles, labels=labels, fontsize=15)
    
    # ax2.set_title('Sensor Positions and DGP Estimates', fontsize=20)
    
    from sklearn.metrics import mean_squared_error
    rmse_x = mean_squared_error(truth_x[3:], x_data, squared=False)
    rmse_y = mean_squared_error(truth_y[3:], y_data, squared=False)
    print('DGPX =',rmse_x)
    print('DGPY =',rmse_y)
    rmsed_x_list.append(rmse_x)
    rmsed_y_list.append(rmse_y)