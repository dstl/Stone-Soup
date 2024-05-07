# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 07:39:25 2024

@author: 007
"""
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity

# And the clock starts
start_time = datetime.now().replace(microsecond=0)
np.random.seed(1991)
q_x = 0.05
q_y = 0.05
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])
timesteps = [start_time]
num_steps = 30

rmse_x_list = []
rmse_y_list = []
rmsed_x_list = []
rmsed_y_list = []
for experiment in range(1):
    truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])

    # np.random.seed(experiment+100)
    for k in range(1, num_steps + 1):
        timesteps.append(start_time+timedelta(seconds=k))  # add next timestep to list of timesteps
        truth.append(GroundTruthState(
            transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=timesteps[k]))
    
        
    # from stonesoup.plotter import AnimatedPlotterly
    # plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
    # plotter.plot_ground_truths(truth, [0, 2])
    # plotter.fig
    transition_model.matrix(time_interval=timedelta(seconds=1))
    transition_model.covar(time_interval=timedelta(seconds=1))
    from stonesoup.types.detection import Detection
    from stonesoup.models.measurement.linear import LinearGaussian
    measurement_model = LinearGaussian(
        ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
        mapping=(0, 2),  # Mapping measurement vector index to state index
        noise_covar=np.array([[5, 0],  # Covariance matrix for Gaussian PDF
                              [0, 5]])
        )
    measurement_model.covar()
    measurement_model.matrix()
    measurements = []
    for state in truth:
        measurement = measurement_model.function(state, noise=True)
        measurements.append(Detection(measurement,
                                      timestamp=state.timestamp,
                                      measurement_model=measurement_model))    
     
    
    ## GP
    from GP_Tracking import GP_track
    Xm = [];Ym = []
    for k in range(num_steps-1):
        truth_x = [state.state_vector[0] for state in truth]
        truth_y = [state.state_vector[2] for state in truth]
    
    for measurement in measurements:
        Xm.append(measurement.state_vector[0])
        # Xt.append(measurement.timestamp)
        Ym.append(measurement.state_vector[1])
        # Yt.append(measurement.timestamp)   
    Xm = np.array(Xm).reshape(-1, 1) 
    Ym = np.array(Ym).reshape(-1, 1)
    
    ## 
    Xt = [];Yt= []
    for k in range(num_steps-1):
        truth_x = [state.state_vector[0] for state in truth]
        truth_y = [state.state_vector[2] for state in truth]
    
    for truth in truth:
        Xt.append(truth.state_vector[0])
        # Xt.append(measurement.timestamp)
        Yt.append(truth.state_vector[2])
        # Yt.append(measurement.timestamp)   
    Xt = np.array(Xt).reshape(-1, 1) 
    Yt = np.array(Yt).reshape(-1, 1)  
      
    A = GP_track()
    
    Size_win = 30   #sliding Window
    x_data,x_cov , y_data,y_cov = A.tracking(measurements, Size_win)
    
    truth_x = np.array(truth_x).reshape(-1, 1) 
    truth_y = np.array(truth_y).reshape(-1, 1) 
    from sklearn.metrics import mean_squared_error
    rmse_x = mean_squared_error(truth_x[2:], x_data, squared=False)
    rmse_y = mean_squared_error(truth_y[2:], y_data, squared=False)
    print('GPX=', rmse_x)
    print('gpy=', rmse_y)
    
    
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(Xm, Ym, 'xb', label='Measurements', markerfacecolor='None')
    ax1.plot(truth_x, truth_y, 'd-k', label='Truth', markerfacecolor='None')
    ax1.plot(x_data, y_data, 'r.', label='Estimates')
    ax1.set_xlabel('X (m)',fontsize=15)
    ax1.set_ylabel('Y (m)',fontsize=15)
    # ax1.set_title('GP', fontsize=25, pad=10)
    ax1.legend(fontsize=15)
    
    
    
    
    
    
    ##DGP
    import matplotlib.patches as mpatches
    from sensor_network import GP_Sensor
    # Generate sensors and record data
    SW = GP_Sensor()
    
    
    seed = 1100; num_sensors = 10
    min_distance = 5; range_value = 40
    xrange = (-20, 55)
    yrange = (-20, 95)
    
    # seed = 1100; num_sensors = 15
    # min_distance = 3; range_value = 20
    # xrange = (-20, 55)
    # yrange = (-20, 95)
    
    
    
    sensor_data= SW.create_sensor_network_plot(num_sensors, range_value, min_distance, xrange, yrange, seed)
    
    time_data1, x_data1, y_data1 = SW.track_targetDGP(Xt, Yt, sensor_data)
    x_data, x_cov , y_data, y_cov, x_s, y_s, t_s = A.tracking_DGP(measurements, time_data1, x_data1, y_data1)
    
    fig = plt.figure(figsize=(10, 6))
    ax2 = fig.add_subplot(1, 1, 1)
    sensor_range = range_value
    ax2.plot(Xm, Ym, 'xb', label='Measurements', markerfacecolor='None')
    ax2.plot(truth_x, truth_y, 'd-k', label='Truth', markerfacecolor='None')
    ax2.plot(x_data, y_data, 'r.', label='Estimates')
    
    x_range = (0, 45)
    y_range = (-5, 100)
    
    for sensor in sensor_data:
        position = sensor["position"]
        if x_range[0] <= position[0] <= x_range[1] and y_range[0] <= position[1] <= y_range[1]:
            ax2.scatter(position[0], position[1], c='orange', edgecolor='black')
            sensor_circle = plt.Circle((position[0], position[1]), sensor_range, color='blue', alpha=0.8,  fill=False)
            ax2.add_patch(sensor_circle)
    
    ax2.scatter([], [], c='orange', label='Sensor', edgecolor='black')  # Invisible point for legend
    # ax2.add_patch(mpatches.Circle((0, 0), 1, eagecolor='blue', alpha=0.1, label='Detection Range'))
    ax2.add_patch(mpatches.Circle((0, 0), 1, edgecolor='blue', facecolor='none', alpha=0.8, label='Detection Range'))

    ax2.set_xlabel('X (m)',fontsize=15)
    ax2.set_ylabel('Y (m)',fontsize=15)
    ax2.legend(loc='upper left',fontsize=15)
    # ax2.set_title('DGP', fontsize=25, pad=10)
    ax2.set_xlim(-20, 60)
    ax2.set_ylim(-20, 100)
    plt.tight_layout()  
    plt.show()
    
    
    from sklearn.metrics import mean_squared_error
    rmse_x = mean_squared_error(truth_x[3:], x_data, squared=False)
    rmse_y = mean_squared_error(truth_y[3:], y_data, squared=False)
    print('DGPX=', rmse_x)
    print('DGPX=', rmse_y)
    rmsed_x_list.append(rmse_x)
    rmsed_y_list.append(rmse_y)
    def sliding_window(t, window_size):
        if window_size <= 0:
            raise ValueError("Invalid window size1")
        if window_size > t:
            start_time = 0
        else:
            start_time = t-window_size + 1
        return start_time
    from GP import GaussianProcess
    
    def update(Data_train, Time_train, test_t):
        G = GaussianProcess (kernel_type='SE')
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
    training_data = {}
    window_size = 40
    Xm = np.array(Xm).reshape(-1, 1)
    Ym = np.array(Ym).reshape(-1, 1)
    for i in range(5, np.max(t_s)):
        SW = sliding_window(i, window_size)
        Time_train = np.arange(SW, i).reshape(i-SW, 1)
        indices = [index for index, value in enumerate(t_s) if value < i]
        Time_train = [t_s[index] for index in indices]
        X_train = [x_s[index] for index in indices]
        Y_train = [y_s[index] for index in indices]
        Time_train = np.array(Time_train).reshape(-1, 1)
        X_train = np.array(X_train).reshape(-1, 1)
        
        # X_train = Xm[sliding_window(i, window_size):i]    
        mu_x, cov_x = update(X_train, Time_train, i)
        x_data.append(mu_x)
        x_cov.append(cov_x)
        # print(i)
        # Y_train = Ym[sliding_window(i, window_size):i]
        mu_y, cov_y = update(Y_train, Time_train, i)
        y_data.append(mu_y)
        y_cov.append(cov_y)
    # tnew = np.array(time_dataF)
    # ynew = np.array(y_trainF)
    x_data = np.array(x_data).reshape(-1, 1)
    y_data = np.array(y_data).reshape(-1, 1)
    rmse_x = mean_squared_error(truth_x[6:], x_data, squared=False)
    rmse_y = mean_squared_error(truth_y[6:], y_data, squared=False)
    print('all data GPX=' , rmse_x)
    print('all data GPY=', rmse_y)
    rmse_x_list.append(rmse_x)
    rmse_y_list.append(rmse_y)
# Xm = np.array(Xm).reshape(-1, 1) 
# Ym = np.array(Ym).reshape(-1, 1)  
# A = GP_track()

# Size_win = 30   #sliding Window
# x_data,x_cov, y_data,y_cov = A.trackingA(measurements, Size_win, sensor_counter)
# # fig = plt.figure(figsize=(10, 6))
# # ax2 = fig.add_subplot(1, 1, 1)
# # ax2.plot(Xm, Ym, 'xb', label='Measurements', markerfacecolor='None')
# # ax2.plot(truth_x, truth_y, 'd-k', label='Truth', markerfacecolor='None')
# # ax2.plot(x_data, y_data, 'r.', label='Estimates')
# # # plt.xlim(200, 200)
# # # plt.ylim(-100, 100)
# # ax2.set_xlabel('X (m)')
# # ax2.set_ylabel('Y (m)')

# fig = plt.figure(figsize=(10, 6))
# ax1 = fig.add_subplot(1, 1, 1)
# # 第一个子图
# ax1.plot(Xm, Ym, 'xb', label='Measurements', markerfacecolor='None')
# ax1.plot(truth_x, truth_y, 'd-k', label='Truth', markerfacecolor='None')
# ax1.plot(x_data, y_data, 'r.', label='Estimates')
# ax1.set_xlabel('X (m)',fontsize=15)
# ax1.set_ylabel('Y (m)',fontsize=15)
# # ax1.set_title('GP', fontsize=25, pad=10)
# ax1.legend(fontsize=15)

# # ax2.legend(fontsize=12) 
# truth_x = np.array(truth_x).reshape(-1, 1) 
# truth_y = np.array(truth_y).reshape(-1, 1) 
# from sklearn.metrics import mean_squared_error
# rmse_x = mean_squared_error(truth_x[2:], x_data, squared=False)
# rmse_y = mean_squared_error(truth_y[2:], y_data, squared=False)
# print(rmse_x)
# print(rmse_y)
