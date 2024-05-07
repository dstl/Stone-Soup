#!/usr/bin/env python
# coding: utf-8

# 
# # An introduction to Stone Soup: using the Gaussian process methods for tracking
# 

# This notebook is designed to introduce to the basic features of Stone Soup using Gaussian process and Distributed Gaussian process to solve a single target scenario as an example.

# ## A nearly-constant velocity example
# We're going to set up a simple scenario in which a target moves at constant velocity with the
# addition of some random noise, (referred to as a *nearly constant velocity* model).
# As is customary in Python scripts we begin with some imports. (These ones allow us access to
# mathematical and timing functions.)

# In[1]:


import numpy as np
from datetime import datetime, timedelta


# ### Simulate a target
# We consider a 2d Cartesian scheme where the state vector is
# $[x \ \dot{x} \ y \ \dot{y}]^T$.  That is, we'll model the target motion as a position
# and velocity component in each dimension. The units used are unimportant, but do need to be
# consistent.
# 
# To start we'll create a simple truth path, sampling at 1 second intervals. We'll do this by employing one of Stone Soup's native transition models. These inputs are required:

# In[2]:


from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity

# And the clock starts
start_time = datetime.now().replace(microsecond=0)


# We note that it can sometimes be useful to fix our random number generator in order to probe a
# particular example repeatedly.
# A 'truth path' is created starting at (0,0) moving to the NE at one distance unit per (time)
# step in each dimension.

# In[3]:


np.random.seed(1991)
q_x = 0.05
q_y = 0.05
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])
timesteps = [start_time]
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])
num_steps = 30
for k in range(1, num_steps + 1):
    timesteps.append(start_time+timedelta(seconds=k))  # add next timestep to list of timesteps
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))


# Thus the ground truth is generated and we can plot the result.Stone Soup has a few in-built plotting classes which can be used to plot
# ground truths, measurements and tracks in a consistent format. An animated plotter that uses
# Plotly graph objects can be accessed via the class :class:`AnimatedPlotterly` from Stone Soup
# as below.Note that the animated plotter requires a list of timesteps as an input, and that 'tail_length'
# is set to 0.3. This means that each data point will be on display for 30% of the total
# simulation time. Also note that the mapping argument is [0, 2] because those are the x and
# y position indices from our state vector.

# In[4]:


from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig


# ### Simulate measurements
# 
# We'll use one of Stone Soup's measurement models in order to generate
# measurements from the ground truth. For the moment we assume a 'linear' sensor which detects the
# position, but not velocity, of a target.
# We're going to need a :class:`~.Detection` type to
# store the detections, and a :class:`~.LinearGaussian` measurement model.
# We're going to need a :class:`~.Detection` type to
# store the detections, and a :class:`~.LinearGaussian` measurement model.

# In[5]:


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
plotter.plot_measurements(measurements, [0, 2])
plotter.fig    


# At this stage you should have a moderately linear ground truth path (dotted line) with a series
# of simulated measurements overplotted (blue circles). Take a moment to fiddle with the numbers in
# $Q$ and $R$ to see what it does to the path and measurements.
# 
# 

# # Gaussian Process methods for Tracking
# Gaussian Process for Tracking
# Gaussian Processes (GPs) provide a flexible approach for modeling and inferring about functions. In tracking problems, when the relationship between states and measurements is non-linear, GPs can be advantageous.

# # Methodological Background of Gaussian Process Methods for Object Tracking
# A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution. It's often defined as:
# $$
# f(x) \sim \mathcal{G} \mathcal{P}\left(m(x), k\left(x, x^{\prime}\right)\right)
# $$
# 
# Where:
# - $f(x)$ is the function to be modeled.
# - $m(x)$ is the mean function, usually set to zero.
# - $k\left(x, x^{\prime}\right)$ is the covariance function, or kernel, which dictates the shape and structure of the functions sampled from the GP.
# 
# The essence of the GP is the kernel, as it encapsulates the assumptions about the function's properties.

# Prediction: Given the nature of Gaussian Processes, we can express the prediction in terms of means and covariances:
# $$
# \begin{aligned}
# & m_*(x)=k\left(x_*, X\right)\left[k(X, X)+\sigma^2 I\right]^{-1} Y \\
# & \Sigma_*(x)=k\left(x_*, x_*\right)-k\left(x_*, X\right)\left[k(X, X)+\sigma^2 I\right]^{-1} k\left(X, x_*\right)
# \end{aligned}
# $$
# 
# Where $X$ are training inputs, $Y$ are training outputs, and $x_*$ is a test input.
# 1. Update: Given a measurement, we update our belief (posterior) about the function. This is done by conditioning the GP on the new data.
# 2. Hyperparameter Learning: The performance of the GP is sensitive to the choice of kernel parameters. These can be optimized by maximizing the log marginal likelihood of the observed data.

# # Implementation
# Implementation
# Similar to the Kalman filter process you've shown, for the GP tracking:
# 1. Define a prior GP based on certain kernel functions.
# 2. At each time step, use the GP to predict the state.
# 3. Upon receiving a measurement, update the GP.
# 4. Optionally, re-estimate hyperparameters if necessary.

# In[6]:


from GP_Tracking import GP_track


# In[7]:


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


# In[8]:


A = GP_track()
Size_win = 30   #sliding Window
x_data,x_cov , y_data,y_cov = A.tracking(measurements, Size_win)


# # Visualizations
# Plot 'Tracking process'which represent the ground truth, measurements, and GP result over time.

# In[9]:


from visualize_tracking import visualize
visualize(state, truth, measurements, x_data, y_data, num_steps,Size_win)


# # Generate Sensor Network and Record Data
# The code snippet below demonstrates how to generate a sensor network and record the data associated with the sensors.

# In[10]:


from sensor_network import GP_Sensor
# Generate sensors and record data

SW = GP_Sensor()

seed = 2333; num_sensors = 60 
min_distance = 3; range_value = 8
xrange = (-5, 45)
yrange = (-5, 85)

sensor_data= SW.create_sensor_network_plot(num_sensors, range_value, min_distance, xrange, yrange, seed)
time_data1, x_data1, y_data1 = SW.track_targetDGP(Xm, Ym, sensor_data)


# # A Distributed Gaussian Process method for Tracking
# Distributed Gaussian Processes (DGPs) are a methodology tailored for handling large datasets by dividing the data into subsets and running a Gaussian process independently on each. The results from these subsets are then aggregated to produce a global model. This framework is especially suited for sensor networks where sensors might be dispersed across various locations, each collecting distinct data.
# 
# The first type of methods is the product of experts (PoEs) approach. The idea is to multiply the local predictive probability distributions for overall predictions. Given the data $D^{(i)}$ collected by sensor $i$, the PoE predicts a function value $f\left(\mathbf{x}_*\right)$ at a corresponding test input $\mathbf{x}_*$ according to
# $$
# p\left(f\left(\mathbf{x}_*\right) \mid \mathbf{x}_*, D\right)=\prod_{i=1}^M p_i\left(f\left(\mathbf{x}_*\right) \mid \mathbf{x}_*, D^{(i)}\right),
# $$
# where $M$ is the number of GP experts and represents the number of active sensors which have measurements. Since the product of these Gaussian predictions is proportional to a Gaussian distribution, the aggregated predictive mean and variance can be calculated with closed form as
# $$
# \begin{aligned}
# \mu_*^{\mathrm{POE}} & =\left(\sigma_*^{\mathrm{PPE}}\right)^2 \sum_{i=1}^M \sigma_i^{-2}\left(\mathbf{x}_*\right) \mu_i\left(\mathbf{x}_*\right), \\
# \left(\sigma_*^{\mathrm{PoE}}\right)^{-2} & =\sum_{i=1}^M \sigma_i^{-2}\left(\mathbf{x}_*\right),
# \end{aligned}
# $$
# where $\mu_i\left(\mathbf{x}_*\right)$ and $\sigma_i^2\left(\mathbf{x}_*\right)$ represent the predictive mean and variance of GP expert $i$, respectively, which can be calculated based on $(6)$ and (7).

# In[11]:


x_data, x_cov , y_data, y_cov = A.tracking_DGP(measurements,time_data1, x_data1, y_data1)


# # Visualizations
# Plot 'Distributed Gaussian Process Tracking'which represent the ground truth, measurements, and DGP result over time.

# In[12]:


from visualize_tracking import visualize_DGP
visualize_DGP(state, truth, measurements, x_data, y_data, num_steps,sensor_data)


# ## References
# [1] X. Liu, J. George, T. Pham, L. Mihaylova, Gaussian Process Upper Confidence Bounds in Distributed Point Target Tracking over Wireless Sensor Networks, IEEE Journal of Selected Topics in Signal Processing, vol. 17, no. 1, pp. 295-310, 2023.
# 
# [2] X. Liu, C. Lyu, J. George, T. Pham and L. Mihaylova, A Learning Distributed Gaussian Process Approach for Target Tracking over Sensor Networks,Proceedings of the 25th International Conference on Information Fusion, Linköping, Sweden, 2022
# 
# [3] C. Lyu, X, Liu, L. Mihaylova, Efficient Factorisation-based Gaussian Process Approaches for Online Tracking, Proceedings of the International Conference on Information Fusion 2022, Linköping.
