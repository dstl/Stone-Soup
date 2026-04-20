#!/usr/bin/env python
# coding: utf-8

# # Applications of the Quadratic Distance to Multi-Target Tracking
# 
# This tutorial demonstrates the usage of the quadratic distance in the context of multi-target tracking sensor management and performance evaluation. An accessible background section is provided which gives ample detail on the formulation of the quadratic distance, the mean quadratic error and the quadratic information gain. A simulation is then constructed in order to provide guidance and intuition on how to use the Stonesoup implementation of these tools. 
# 
# ## Motivation
# Many problems require the comparison of multi-target states, the simplest being the assessment of estimation accuracy where the notion of 'good' in the question "how good is my multi-target state estimate?" is governed by the choice of function used to compare the true and estimated states. There exists an infinite number of functions which may be used for this purpose; a subset of these functions which abide by the axioms of symmetry, positive-definiteness and the triangle inequality are termed metrics or distances. Functions which do not abide by these axioms are often termed measures (not to be confused with objects appearing under the same name in the field of measure theory). Metrics provide a qunatitative comparison which is objective, whereas measures tend to be subjective and based on qualitative features. The notion of quadratic error (or Euclidean error) is ubiquitous in mathematical analysis. Being able to consider the quadratic distance between multi-target state representations allows for the application of a vast range of analysis techniques to the problems of multi-target tracking, namely tools from the domains of estimation theory, decision theory and information theory just to name a few may be utilised. The quadratic distance for point patterns, developed in [1], is a metric on the space of point patterns (multi-target state representations). The metric can compare, point patterns (sets of target states), distribution mixtures (sets of target state estimates with uncertainty) and random point patterns, descriped by point processes. This notion allows for a well understood geometric intuition to be applied in the context of multi-target tracking.
# 
# ## Background
# The quadratic distance is a widely used concept across many domains. The most familiar example is that of the metric in $d$-dimensional Euclidean space
# $$
# d(\boldsymbol{x},\boldsymbol{y})=\sqrt{\sum\limits_i^d(\boldsymbol{x}_i - \boldsymbol{y}_i)^2}, \quad\quad \boldsymbol{x},\boldsymbol{y} \in \mathbb{R}^d.
# $$
# Multi-object states are not represented by vectors but rather sets of vectors. This metric is well understood for the vector case but is less developed for the case of sets. Recent work has developed a quadratic distance between point patterns, i.e., sets of single target states, and their random counterparts point processes. Consider two multi-target state representations in the form of two sets of single target states
# $$
# X = \{x_i, \dots, x_{N}\}, \quad Y = \{y_i, \dots, y_{M}\}.
# $$
# These multi-object states may be described by counting measures, i.e., functions which return the number of elements of a set which fall into a particular region of the element space, $A$ for example
# $$
# \#_X(A)=\sum\limits_{\hat x\in X}\mathbb 1_A(\hat x), \quad \#_Y(A)=\sum\limits_{\hat y\in Y} \mathbb 1_A(\hat y),
# $$
# where $\mathbb 1_A(y)=1$ if $y\in A$, and  $\mathbb 1_A(y)=0$ otherwise. If we instead consider the integral operator kernel form of the counting measures, given by the Dirac delta mixtures
# $$
# \varphi(x) = \sum\limits_{\hat x\in X}\delta(x-\hat x), \quad \psi(y) = \sum\limits_{\hat y\in Y}\delta(y-\hat y),
# $$
# as the model which describes these point patterns, then we can compute the quadratic distance between these descriptions using the following expression
# $$
# \mathcal{Q}_\Lambda(\varphi - \psi) = \int (\varphi(x)-\psi(x))\Lambda(x,y)(\varphi(y)-\psi(y))\mathrm dx \mathrm dy.
# $$
# This is the quadratic distance. The above expression is the inner product of $(\varphi - \psi)$ with itself, weighted by the symmetric, positive-definite kernel $\Lambda(x,y)$. Expanding this expression gives the following
# $$
# \begin{aligned}
#     \mathcal{Q}_\Lambda(\varphi - \psi) &= \int \varphi(x)\Lambda(x,y)\varphi(y)\mathrm dx \mathrm dy \\
#     &-2 \int \varphi(x)\Lambda(x,y)\psi(y)\mathrm dx \mathrm dy \\
#     &+ \int \psi(x)\Lambda(x,y)\psi(y)\mathrm dx \mathrm dy.
# \end{aligned}
# $$
# The objects, or representations, $\varphi$ and $\psi$, may be given by any form
# $$
# \varphi(x) = \sum\limits_{\hat x\in X}f(x, \hat x),
# $$
# where the bivariate function $f(\cdot, \cdot)$ is a distribution, i.e., the integral of this function over its domain is equal to 1. Other than the Dirac delta which is the simplest example, another common example would be that of the Gaussian mixture
# $$
# \varphi(x) = \sum\limits_{(\hat x, \hat P)\in X}\mathcal N(x; \hat x,\hat P).
# $$
# This multi-target state representation may be chosen in the case where one wishes to express the uncertainty regarding the state of each target within the population using the covariance matrix $\hat P$. We now go on to discuss the implications of choosing the kernel, $\Lambda(x,y)$.
# 
# ### Kernels
# The kernel $\Lambda(x,y)$ may be any symmetric, positive-definite function, i.e.,
# $$
# \begin{aligned}
#     \Lambda(x,y)&: \mathbb{R}^d \times \mathbb{R}^d \mapsto \mathbb{R};\\
#     \Lambda(x,y)&=\Lambda(y,x);\\
#     \int \varphi(x)&\Lambda(x,y)\varphi(y)\mathrm dx \mathrm dy > 0, \quad \forall\,\, \varphi \in \mathcal M,
# \end{aligned}
# $$
# where $\mathcal M$ is the space of multi-target state represented by counting measures. The kernel determines the nature of the metric by assigning to each pair of targets a weight which expresses the strength of the relationship between them. For example, the Gaussian kernel
# $$
# \Lambda(x,y) = \exp(-\frac{1}{2}(x-y)^\top R^{-1} (x-y)),
# $$
# parametrised by the covariance matrix $R$, assigns a weight to the pair $(x, y)$ which decays exponentially with the distance between them in the state space. The distance between these state vectors is measured by the Euclidean distance weighted by the inverse covariance matrix $R^{-1}$. A large covariance implies that the distance between these points is considered small even if they are sparsely seperated. Conversely, if the covariance is small, then the distance between points close in proximity may be considered large. Infinitely many kernel choices may be considered and the interested reader is referred to the following resources [4]. For the remainder of this tutorial, the Gaussian kernel is considered.
# 
# ### The Mean Squared Error for point patterns
# The concept of Mean Squared Error is well known throughout many disciplines. In estimation theory, it is used to assess the quality of a particular choice of estimator. By considering the quadratic distance, or error, between a point process, $\boldsymbol X$, and a counting measure, $\varphi$, we obtain a metric between the random object and the deterministic object. Consider the deterministic object to be an estimator of the unknown object modelled by the point process $\boldsymbol X$. If we take the expectation of this quadratic error with respect to the point process $\boldsymbol X$, we arrive at an expression for the Mean Quadratic Error (MQE) of $\varphi$ as an estimator of $X$:
# $$
# \begin{aligned}
#     \mathbb E_{\boldsymbol X}\big[\mathcal Q_\Lambda(\boldsymbol X - \varphi)\big] &=  \mathbb E_{\boldsymbol X}\bigg[\int X(x)\Lambda(x,y)X(y)\mathrm dx \mathrm dy\bigg] \\
#     &-2 \mathbb E_{\boldsymbol X}\bigg[\int \varphi(x)\Lambda(x,y)\psi(y)\mathrm dx \mathrm dy \bigg]\\
#     &+ \int \psi(x)\Lambda(x,y)\psi(y)\mathrm dx \mathrm dy.
# \end{aligned}
# $$
# This can be factorised into the following decomposition
# $$
# \begin{aligned}
#     \mathbb E_{\boldsymbol X}\big[\mathcal Q_\Lambda(\boldsymbol X - \varphi)\big] &=  \int \Lambda(x,y)\mathrm{cov}_{\boldsymbol X}(x,y)\mathrm dx \mathrm dy \\
#     &+ \mathcal Q_\Lambda(\mathbb E_{\boldsymbol X}[X]-\varphi),
# \end{aligned}
# $$
# which can be interpreted in terms of the well known bias-variance decomposition of the MSE [1]. The first term is the kernel smoothed covariance of the point process $\boldsymbol X$ and the second term is the squared bias of the estimator $\varphi$.
# 
# ### The Quadratic Information Gain
# In the context of multi-target filtering, it is common to make decisions regarding the actions of available sensors prior to obtaining measurements. In order to do this the action maximising the information gain is chosen. The information gain is commonly formulated as a function of the predicted and updated probability distirbutions. If we consider the above expression in the case where $\boldsymbol X$ denotes the predicted point process and $\varphi(\boldsymbol Z)$ denotes the intensity of the posterior point process as a function of the measurmeent point process, then we can develop the following expression of information gain based on the MQE
# 
# $$
# \begin{aligned}
#     \mathbb E_{\boldsymbol Z}\big[\mathbb E_{\boldsymbol X}\big[\mathcal Q_\Lambda(\boldsymbol X - \varphi)\big]\big] &=  \int \Lambda(x,y)\mathrm{cov}_{\boldsymbol X}(x,y)\mathrm dx \mathrm dy \\
#     &+ \mathbb E_{\boldsymbol Z}\big[\mathcal Q_\Lambda(\mathbb E_{\boldsymbol X}[X]-\varphi)\big].
# \end{aligned}
# $$
# 
# ## Implementation
# We consider the implementation of the above tools for the Gaussian kernel case. Depending on the input type, the implementation varies and as such we discuss implementations for the quadratic distance between point patterns, between a point pattern and a gaussian mixture, and between gaussian mixtures. The implemented expressions may be found in [1]. The implementation of the MQE and the QIG is developed for the specific case of the Gaussian mixture PHD filter [1][3]. The implementation of these tools is bespoke to the filtering method considered since the computation requires knowledge of the second order moment of the posterior point process [1][3].

# # Simulation
# We now develop a multi-target tracking scenario in which the quadratic distance is used to perform online sensor management of rotating range-bearing sensors and offline performance evaluation. The multi-target filter considered in this simulation will be the Gaussian Mixture PHD filter.
# 
# ### Ground Truths
# Firstly, a time-varying multi-target population is generated. Increasing the number of timesteps, number of initial targets, death probability or birth probability will chnage the difficulty of the multi-target tracking scenario.

# In[46]:


# Imports for plotting
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (14, 12)
plt.style.use('seaborn-v0_8-colorblind')
# Other general imports
import numpy as np
from ordered_set import OrderedSet
from stonesoup.types.state import State, TaggedWeightedGaussianState
from stonesoup.types.track import Track
from stonesoup.types.array import CovarianceMatrix
from datetime import datetime, timedelta
start_time = datetime.now().replace(microsecond=0)

np.random.seed(2007)

truths_by_time = []

# Create transition model
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(0.3), ConstantVelocity(0.3)))

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
start_time = datetime.now()
truths = OrderedSet()  # Truths across all time
current_truths = set()  # Truths alive at current time
start_truths = set()
number_steps = 50
timesteps = [start_time + timedelta(seconds=k) for k in range(number_steps)]

death_probability = 0
birth_probability = 0

# Initialize truths.
num_init_targs = 5
initial_states = []
    
truths_by_time.append([])
for i in range(num_init_targs):
    x, y = initial_position = np.random.uniform(-30, 30, 2)  # Range [-30, 30] for x and y
    x_vel, y_vel = (np.random.rand(2))*2 - 1  # Range [-1, 1] for x and y velocity
    state = GroundTruthState([x, x_vel, y, y_vel], timestamp=start_time)
    
    initial_states.append([x, x_vel, y, y_vel])
    
    truth = GroundTruthPath([state])
    current_truths.add(truth)
    truths.add(truth)
    start_truths.add(truth)
    truths_by_time[0].append(state)

# Simulate over time
for k in range(1, number_steps):
    timestep = start_time + timedelta(seconds=k)

    # Update existing truths
    for truth in current_truths:
        prev_state = truth[-1]  # Last state in the path
        new_state_vector = transition_model.function(prev_state, noise=True, time_interval=timedelta(seconds=1))
        new_state = GroundTruthState(new_state_vector, timestamp=timestep)
        truth.append(new_state)  # Always append a single GroundTruthState

    # Birth new targets
    for _ in range(np.random.poisson(birth_probability)):
        x, y = np.random.uniform(0, 120, 2)
        x_vel, y_vel = np.random.uniform(-1, 1, 2)
        state = GroundTruthState([x, x_vel, y, y_vel], timestamp=timestep)
        truth = GroundTruthPath([state])
        truths.add(truth)
        current_truths.add(truth)

    # Death 
    for truth in current_truths.copy():
        if np.random.rand() <= death_probability:
            current_truths.remove(truth)


# plot ground truths
from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps, tail_length=1, sim_duration=10)
plotter.plot_ground_truths(truths, [0, 2])
plotter.fig


# ### Sensors
# Next, :class:`~.RadarRotatingBearingRange` sensor objects are initialised. In this simulation, we will considered two methods of tracking the previously generated targets: one in which the sensors are tasked randomly and another in which the sensors are tasked optimally according to the QIG. Two sets of sensors are created: one for the random sensor manager and another for the quadratic distance based sensor manager. The dwell centre of each sensor in the configuration may controlled by the sensor manager. 

# In[48]:


from stonesoup.types.state import StateVector
from stonesoup.sensor.radar.radar import RadarRotatingBearingRange
from stonesoup.types.angle import Angle

# sensor parameters
n_sensors = 1
sens_range = 1000
sens_fov = np.radians(90)
sens_res = Angle(np.radians(90))
sens_noise = np.array([[np.radians(0.5) ** 2, 0],
                              [0, 1 ** 2]])
sens_rpm = 120
surveillance_area = (0.5 * sens_range**2 * sens_fov)


sensor_setA = set()
for n in range(0, n_sensors):
    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=sens_noise,
        ndim_state=4,
        position=np.array([[10], [n * 50]]),
        rpm=sens_rpm,
        fov_angle=sens_fov,
        dwell_centre=StateVector([0.0]),
        max_range=sens_range,
        resolution=sens_res
    )
    sensor_setA.add(sensor)
for sensor in sensor_setA:
    sensor.timestamp = start_time

sensor_setB = set()
for n in range(0, n_sensors):
    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=sens_noise,
        ndim_state=4,
        position=np.array([[10], [n * 100]]),
        rpm=sens_rpm,
        fov_angle=sens_fov,
        dwell_centre=StateVector([0.0]),
        max_range=sens_range,
        resolution=sens_res
    )
    sensor_setB.add(sensor)

for sensor in sensor_setB:
    sensor.timestamp = start_time

#global multi-target detection parameters: can be changed for each sensor
probability_detection = 0.99
clutter_rate = 0
clutter_spatial_density = clutter_rate/surveillance_area


# ### Initialise GM-PHD Filter and Sensor Managers
# We initialise the GM-PHD filters and the sensor manager objects. The random sensor manager requires little set up whereas the QIG manager requires information regarding the filter. We also specify the particular Gaussian kernel to be used by the QIG reward function. Due to the nature of the posterior intensity of the Gaussian Mixture PHD filter, the expectation with respect to measurements must be approximated numerically, hence, the number of samples with with this will be done must be specified [3]. 

# In[51]:


from stonesoup.types.track import Track
from stonesoup.types.array import CovarianceMatrix
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.updater.pointprocess import PHDUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer

# predictor
kalman_predictor = KalmanPredictor(transition_model)

# updater
extended_kalman_updater = ExtendedKalmanUpdater(measurement_model=None)

updater = PHDUpdater(
    extended_kalman_updater,
    clutter_spatial_density=clutter_spatial_density,
    prob_detection=probability_detection,
    prob_survival=1-death_probability)

# hypothesisers
base_hypothesiser = DistanceHypothesiser(kalman_predictor, extended_kalman_updater, Mahalanobis(), missed_distance=3)

hypothesiser = GaussianMixtureHypothesiser(base_hypothesiser, order_by_detection=True)

# reducer
merge_threshold = 5
prune_threshold = 1E-8

reducer = GaussianMixtureReducer(
    prune_threshold=prune_threshold,
    pruning=True,
    merge_threshold=merge_threshold,
    merging=True)

# birth model
birth_covar = CovarianceMatrix(np.diag([1000, 2, 1000, 2]))
birth_component = TaggedWeightedGaussianState(
    state_vector=[0, 0, 0, 0],
    covar=birth_covar**2,
    weight=0.25,
    tag='birth',
    timestamp=start_time
)

# prior multi-target estimate
covar = CovarianceMatrix(np.diag([1, 0.5, 1, 0.5]))

tracksA = set()
tracksB = set()
for truth_vect in initial_states:
    new_track = TaggedWeightedGaussianState(
            state_vector=truth_vect,
            covar=covar**2,
            weight=0.9,
            tag='birth',
            timestamp=start_time)
    tracksA.add(Track([new_track]))
    tracksB.add(Track([new_track]))

reduced_statesA = set([track[-1] for track in tracksA])
reduced_statesB = set([track[-1] for track in tracksB])


# sensor managers initialisation
from stonesoup.sensormanager import RandomSensorManager
from stonesoup.sensormanager import GreedySensorManager
from stonesoup.sensormanager.reward import QuadraticInformationGain

randomsensormanager = RandomSensorManager(sensor_setA)

# kernel parameters
kernel_cov = 0.0001*np.eye(4)

# filter parameters dictionary
filter_data_dict = {'filter model':'GMPHD',
                    'state dimension':4,
                    'detection probability':probability_detection,
                    'surveillance area':surveillance_area,
                    'survival probability':1-death_probability,
                    'clutter rate':clutter_rate,
                    'predictor':kalman_predictor,
                    'updater': extended_kalman_updater}

reward_function = QuadraticInformationGain(num_samples=50, filter_data=filter_data_dict, kernel='Gaussian', kernel_parameters={'covariance':kernel_cov})

greedysensormanager = GreedySensorManager(sensor_setB, reward_function=reward_function)


# ### Random sensor management 
# We now track the targets by randomly selecting the dwell centre of each sensor.

# In[54]:


from stonesoup.types.detection import Clutter
from stonesoup.types.detection import TrueDetection
from scipy.stats import uniform
from ordered_set import OrderedSet
from collections import defaultdict
import copy

all_gaussiansA = []
tracks_by_timeA = []
state_threshold = 0.75
hypotheses1 = []
all_measurementsA = []
sensor_history_A = defaultdict(dict)

for n, timestep in enumerate(timesteps[1:]):
    print(f"Timestep {n+2}", end="\r")
    tracks_by_timeA.append([])
    all_gaussiansA.append([])

    current_state = reduced_statesA

    ####### sensor management #######
    chosen_actions = randomsensormanager.choose_actions(current_state, timestep)

    for chosen_action in chosen_actions:
        for sensor, actions in chosen_action.items():
            sensor.add_actions(actions)

    measurementsA = set()
    for sensor in sensor_setA:
        sensor.act(timestep)
        sensor_history_A[timestep][sensor] = copy.copy(sensor)
     
        # target detections 
        alive_truths = []
        for truth in truths:
            try:
                alive_truths.append(truth[timestep]) 
            except IndexError:
                # This truth not alive at this time. Skip this iteration of the for loop.
                continue
                
        measurementsA |= sensor.measure(OrderedSet(gt for gt in alive_truths if np.random.rand() <= probability_detection), noise=True)
        
        
        # Generate clutter at this time-step
        for _ in range(np.random.poisson(clutter_rate)):
            
            theta = uniform.rvs(sensor.dwell_centre.item()-sens_fov/2, sens_fov)
            r = uniform.rvs(0, sens_range)
            measurementsA.add(Clutter(np.array([[theta], [r]]), timestamp=timestep,
                                        measurement_model=sensor.measurement_model))

    all_measurementsA.append(measurementsA)
    #################################
 
    birth_component.timestamp = timestep
    current_state.add(birth_component)
    
    
    
    # Generate the set of hypotheses
    hypotheses = hypothesiser.hypothesise(current_state,
                                          measurementsA,
                                          timestamp=timestep,
                                          # keep our hypotheses ordered by detection, not by track
                                          order_by_detection=True)
    hypotheses1.append(Track(hypotheses))
    
    # Turn the hypotheses into a GaussianMixture object holding a list of states
    updated_states = updater.update(hypotheses)

    # Prune and merge the updated states into a list of reduced states
    reduced_statesA = set(reducer.reduce(updated_states))

    for reduced_state in reduced_statesA:
        if reduced_state.weight > 0.05: all_gaussiansA[n].append(reduced_state)

        tag = reduced_state.tag
        # Here we check to see if the state has a sufficiently high weight to consider being added.
        if reduced_state.weight > state_threshold:
            # Check if the reduced state belongs to a live track
            for track in tracksA:
                track_tags = [state.tag for state in track.states]

                if tag in track_tags:
                    track.append(reduced_state)
                    tracks_by_timeA[n].append(reduced_state)
                    break
            else:
                new_track = Track(reduced_state)
                tracksA.add(new_track)
                tracks_by_timeA[n].append(reduced_state)

# extract means from tracks
meansA = set()
for track in tracksA:
    new_mean = []
    for state in track:
        new_mean.append(State(state_vector=state.state_vector.copy(),
                              timestamp=state.timestamp))
    meansA.add(Track(new_mean))

# plotting
import plotly.graph_objects as go
from stonesoup.functions import pol2cart
from stonesoup.functions import cart2pol

def plot_sensor_fov(fig_, sensor_set, sensor_history):
    # Plot sensor field of view
    trace_base = len(fig_.data)
    for _ in sensor_set:
        fig_.add_trace(go.Scatter(mode='lines',
                                  line=go.scatter.Line(color='black',
                                                       dash='dash')))

    for frame in fig_.frames:
        traces_ = list(frame.traces)
        data_ = list(frame.data)

        timestring = frame.name
        timestamp = datetime.strptime(timestring, "%Y-%m-%d %H:%M:%S.%f")

        for n_, sensor_ in enumerate(sensor_set):
            x = [0, 0]
            y = [0, 0]

            if timestamp in sensor_history:
                sensor_ = sensor_history[timestamp][sensor_]
                for i, fov_side in enumerate((-1, 1)):
                    range_ = min(getattr(sensor_, 'max_range', np.inf), 100)
                    x[i], y[i] = pol2cart(range_,
                                          sensor_.dwell_centre[0, 0]
                                          + sensor_.fov_angle / 2 * fov_side) \
                        + sensor_.position[[0, 1], 0]
            else:
                continue

            data_.append(go.Scatter(x=[x[0], sensor_.position[0], x[1]],
                                    y=[y[0], sensor_.position[1], y[1]],
                                    mode="lines",
                                    line=go.scatter.Line(color='black',
                                                         dash='dash'),
                                    showlegend=False))
            traces_.append(trace_base + n_)

        frame.traces = traces_
        frame.data = data_           

# Plot the tracks
plotterA = AnimatedPlotterly(timesteps, tail_length=1, sim_duration=10)
plotterA.plot_sensors(sensor_setA)
plotterA.plot_ground_truths(truths, [0, 2])
plotterA.plot_measurements(all_measurementsA, [0, 2])
plotterA.plot_tracks(tracksA, [0, 2], uncertainty=True, plot_history=False)
plot_sensor_fov(plotterA.fig, sensor_setA, sensor_history_A)
plotterA.fig


# ### Quadratic information gain sensor management
# Now, the same targets are tracked, but the sensors are optimally controlled so as to maximise the information gain. This can also be interpreted in terms of uncertainty reduction as the actions maximising the QIG correspond to those which minimise the posterior uncertainty. However, it must be stated that if the number of samples used to approximate the QIG is low, then this relationship may not always hold. This leads to a trade-off between optimality and computational cost.

# In[56]:


from stonesoup.types.detection import Clutter
from scipy.stats import uniform
from collections import defaultdict
import copy

all_gaussiansB = []
tracks_by_timeB = []
state_threshold = 0.75
hypotheses2 = []
all_measurementsB = []
sensor_history_B = defaultdict(dict)

for n, timestep in enumerate(timesteps[1:]):
    print(f"Timestep {n+2}", end="\r")
    tracks_by_timeB.append([])
    all_gaussiansB.append([])

    current_state = reduced_statesB
    

    ####### sensor management #######
    chosen_actions = greedysensormanager.choose_actions(current_state, timestep)

    for chosen_action in chosen_actions:
        for sensor, actions in chosen_action.items():
            sensor.add_actions(actions)

    measurementsB = set()
    for sensor in sensor_setB:
        sensor.act(timestep)
        sensor_history_B[timestep][sensor] = copy.copy(sensor)
        
        # target detections 
        alive_truths = []
        for truth in truths:
            try:
                alive_truths.append(truth[timestep]) 
            except IndexError:
                # This truth not alive at this time. Skip this iteration of the for loop.
                continue
                
        measurementsB |= sensor.measure(OrderedSet(gt for gt in alive_truths if np.random.rand() <= probability_detection), noise=True)
        
        
        # Generate clutter at this time-step
        for _ in range(np.random.poisson(clutter_rate)):
            
            theta = uniform.rvs(sensor.dwell_centre.item()-sens_fov/2, sens_fov)
            r = uniform.rvs(0, sens_range)
            measurementsB.add(Clutter(np.array([[theta], [r]]), timestamp=timestep,
                                        measurement_model=sensor.measurement_model))


    all_measurementsB.append(measurementsB)
    #################################


    birth_component.timestamp = timestep
    current_state.add(birth_component)
    
    # Generate the set of hypotheses
    hypotheses = hypothesiser.hypothesise(current_state,
                                          measurementsB,
                                          timestamp=timestep,
                                          # keep our hypotheses ordered by detection, not by track
                                          order_by_detection=True)
    hypotheses2.append(Track(hypotheses))
    
    # Turn the hypotheses into a GaussianMixture object holding a list of states
    updated_states = updater.update(hypotheses)

    # Prune and merge the updated states into a list of reduced states
    reduced_statesB = set(reducer.reduce(updated_states))

    for reduced_state in reduced_statesB:
        if reduced_state.weight > 0.05: all_gaussiansB[n].append(reduced_state)

        tag = reduced_state.tag
        # Here we check to see if the state has a sufficiently high weight to consider being added.
        if reduced_state.weight > state_threshold:
            # Check if the reduced state belongs to a live track
            for track in tracksB:
                track_tags = [state.tag for state in track.states]

                if tag in track_tags:
                    track.append(reduced_state)
                    tracks_by_timeB[n].append(reduced_state)
                    break
            else:  
                new_track = Track(reduced_state)
                tracksB.add(new_track)
                tracks_by_timeB[n].append(reduced_state)
                
# extract means from tracks
meansB = set()
for track in tracksB:
    new_mean = []
    for state in track:
        new_mean.append(State(state_vector=state.state_vector.copy(),
                              timestamp=state.timestamp))
    meansB.add(Track(new_mean))


            
# Plot the tracks
plotterB = AnimatedPlotterly(timesteps, tail_length=1, sim_duration=10)
plotterB.plot_sensors(sensor_setB)
plotterB.plot_ground_truths(truths, [0, 2])
plotterB.plot_measurements(all_measurementsB, [0, 2])
plotterB.plot_tracks(tracksB, [0, 2], uncertainty=True, plot_history=False)
plot_sensor_fov(plotterB.fig, sensor_setB, sensor_history_B)
plotterB.fig


# ### Performance Evaluation
# Now we will evaluate the performance of the two sensor management methods using the quadratic distance and the MQE. Firstly, the MQE will be used to evaluate the quality of the posterior point process as an estimator of the true population state. This assessment of quality will take into account the second order moment of the posterior distribution, i.e., the uncertainty in the number of targets. If this is large, then the estimation quality is deemed to be lower. The quadratic distance is also used to compare the performance of the two filters, but in this case will only operate on the first order moment of the posterior distribution. Using the quadratic distance, we consider the error between the truths and the extracted means from the gaussian mixture intensity, the truths and the gaussian mixture intensity. These comparisons are considered for the random and QIG sensor management cases.

# In[58]:


from stonesoup.metricgenerator.quadraticdistance import QuadraticDistance, MeanQuadraticError
from stonesoup.metricgenerator.manager import MultiManager

# gaussian kernel covariance matrix for metrics
kernel_cov = 100*np.eye(4)

# setup metric generators & multimanager
quaderr_pp1 = QuadraticDistance(kernel='Gaussian', kernel_parameters={'covariance':kernel_cov}, generator_name='Truths - Means (Random)',
                         tracks_key='means1', truths_key='truths')

quaderr_pt1 = QuadraticDistance(kernel='Gaussian', kernel_parameters={'covariance':kernel_cov}, generator_name='Truths - Tracks (Random)',
                         tracks_key='tracks1', truths_key='truths')

quaderr_pp2 = QuadraticDistance(kernel='Gaussian', kernel_parameters={'covariance':kernel_cov}, generator_name='Truths - Means (Quadratic)',
                         tracks_key='means2', truths_key='truths')

quaderr_pt2 = QuadraticDistance(kernel='Gaussian', kernel_parameters={'covariance':kernel_cov}, generator_name='Truths - Tracks (Quadratic)',
                         tracks_key='tracks2', truths_key='truths')

quaderr_tt = QuadraticDistance(kernel='Gaussian', kernel_parameters={'covariance':kernel_cov}, generator_name='Tracks (Random) - Tracks (Quadratic)',
                         tracks_key='tracks1', truths_key='tracks2')

mquaderr_posterior_truth1 = MeanQuadraticError(filter_data=filter_data_dict, kernel='Gaussian', kernel_parameters={'covariance':kernel_cov}, 
                                              generator_name='Truths - Posterior (Random)', tracks_key='tracks1', hypotheses_key='hypotheses1', truths_key='truths')

mquaderr_posterior_truth2 = MeanQuadraticError(filter_data=filter_data_dict, kernel='Gaussian', kernel_parameters={'covariance':kernel_cov}, 
                                              generator_name='Truths - Posterior (Quadratic)', tracks_key='tracks2', hypotheses_key='hypotheses2', truths_key='truths')

metric_manager = MultiManager([quaderr_pp1, quaderr_pt1, quaderr_pp2, quaderr_pt2, quaderr_tt, mquaderr_posterior_truth1, mquaderr_posterior_truth2])

# add data to manager
metric_manager.add_data({'truths': truths,
                         'means1': meansA,
                         'tracks1': tracksA,
                         'hypotheses1': hypotheses1,
                         'means2': meansB,
                         'tracks2': tracksB,
                         'hypotheses2': hypotheses2}, overwrite=False)

# compute metrics
metrics = metric_manager.generate_metrics()

# collect metrics
from stonesoup.plotter import MetricPlotter

graph = MetricPlotter()
graph.plot_metrics(metrics, generator_names=['Truths - Means (Random)',
                                             'Truths - Means (Quadratic)',
                                             'Truths - Tracks (Random)',
                                             'Truths - Tracks (Quadratic)',
                                             'Tracks (Random) - Tracks (Quadratic)',
                                             'Truths - Posterior (Random)',
                                             'Truths - Posterior (Quadratic)'], 
                                             color=['red', 'orange', 'brown', 'green', 'blue', 'black'])

# update y-axis label and title; other subplots are displaying auto-generated title and labels
graph.axes[1].set(ylabel='Quadratic Distance', title='Quadratic Distance over time')
graph.axes[0].set(ylabel='Mean Quadratic Error', title='Mean Quadratic Error over time')
plt.savefig("plot.pdf", dpi=300)
plt.show()


# ## Interpretation of metrics
# ### Mean quadratic error (graph 1)
# The MQE, while not strictly a mathematical metric, provides insight into the quality of the estimator of the true multi-target state. This measure of quality can be interpreted as the sum of the uncertainty of the estimator and the similarity of the mean of the estimator to the true multi-target state, i.e., the bias of the estimator. In this case, the posterior of the two filters is considered, at each timestep, as an estimator of the multi-target state. This estimator, in the case of the GM-PHD filter, takes the form of a point process which is constructed as the sum of a Poisson point process and N-Bernoulli point processes. The first moment of this point process, the intensity, is propagated by the filter. This intensity is given by a Gaussian mixture. The second moment however, is not. However, the second moment is considered in the computation of the MQE and hence, the uncertainty of the posterior is accounted for in this evaluation. This uncertainty can be interpreted as the uncertainty in the estimated number of targets. The bias component of the MQE is computed as the quadratic distance between the truth point patterns and the Gaussian mixture intensities. This bias computation takes into account the localisation uncertainty encoded in the Gaussian mixture by the covariance of each component. The MQE can thus be interpreted as a combination of uncertainty in the number of targets and the accuracy of the estimate weighted by the uncertainty in individual state estimates.
# 
# #### Truths - Posterior (Random)(RED): MQE of the posterior of the randomly managed filter as an estimator of the truths
# First, we consider the time-varying posterior of the GM-PHD filter whose updates are performed using observations from randomly controlled sensors, as the estimator of the truth. One would expect the uncertainty of the estimates provided by this filter to be high as the sensors are not optimally tasked with the goal of collecting observations of relevant targets. The bias of this estimator is also expected to be high, as without suitable observations, the filter will not be able to accurately estimate the states of the targets. The graph for this filter has a large variance which is to be expected of a random sensor control scheme.
# 
# #### Truths - Posterior (Quadratic)(YELLOW): MQE of the posterior of the QIG managed filter as an estimator of the truths
# Now consider the time-varying posterior of the GM-PHD filter whose updates are performed using observations from optimally controlled sensors as the estimator of the truth. Up to the approximation quality of the reward function, the sensors in this case are being tasked such that the chosen sensor configuration is the one which minimises the posterior uncertainty. Due to this, we would expect the MQE of this estimator to be lower than that of the filter based on randomly managed sensors - even if the accuracy of the filters are both kept constant. The bias of this estimator is expected to be less than that of the previously discussed filter as observations are obtained for tracked sensors, leading to more accurate estimates.
# 
# ### Quadratic Distance (graph 2)
# The quadratic distance in this case is used to evaluate the estimation accuracy of the tracks produced by the filters. A useful feature of this metric is that it can accomodate a variety of inputs. here we use this feature to show the difference between the full Gaussian mixture and the extracted means as estimates of the true multi-target state. When comparing the full Gaussian mixtures (tracks) to the truths, we see that the penalty is reduced. This can be interpreted as a reduction in penalty in the case of uncertainty. For example, an estimate which is wrong and has high confidence will be intuitively punished more than a worng estimate with lower confidence. We also use this feature to compare the tracks of the two filters which allows for the direct investigation of how and when the two methods differ.
# 
# #### Truths - Means (Random)(RED): Quadratic distance between the truths and the extracted means from the randomly managed filter
# This comparison considers the estimates of the true states to be a set of points given by the means of the gaussian mixture whose component weight falls above some threshold. This can be viewed as an approximation of the full estimate which is given by the Gaussian mixture. Furthermore the reduction process of the Gaussian mixture intensity introduces further approximation before means are extracted. This comparison receives the largest penalty over the duration of the simulation.
# 
# #### Truths - Means (Quadratic)(YELLOW): Quadratic distance between the truths and the extracted means from the QIG managed filter
# This comparison shows a decrease in penalty when compared to the above, indicating that the state estimation accuracy of the QIG managed filter is greater than that of the randomly managed filter.
# 
# #### Truths - Tracks (Random)(BROWN): Quadratic distance between the truths and the Gaussian tracks from the randomly managed filter
# The second largest penalty is awarded to the tracks fo the randomly managed filter. This is slightly smaller than the means of the randomly managed filter's intensity due to the inclusion of uncertainty in local state estimates.
# 
# #### Truths - Tracks (Quadratic)(GREEN): Quadratic distance between the truths and the Gaussian tracks from the QIG managed filter
# Here, the results indicate that this estimate of the truths is the best performer since it is accurate and provides uncertainty information for each state estimate.
# 
# #### Tracks (Random) - Tracks (Quadratic)(BLUE): Quadratic distance between the Gaussian tracks of the randomly and QIG managed filters
# Intuitively from the above results, the difference between the two tracks is far smaller than the difference between the truth and each of these tracks. This indicates that the tracks of the two objects are relatively similar. Analysing this graph provides insight into when the two filters perform differently and how significant this difference is.
# 
# ### Notes on kernel covariance magnitude
# If the kernel used for the computation of these metrics is the same as that used in the computation of the QIG reward function, then analysis may be made regarding the actual error induced by the sensor configurations. The reward function in the optimal sensor management approach seeks to minimise the expected MQE and subsequently the quadratic distance. However, due to the numerical approximation of the reward function, this may not always be successfully acheived. 
# 
# By selecting a kernel with larger kernel covariance magnitude, the metric will show more varied behaviour. If the covariance magnitude of the Gaussian kernel is small, then, unless extremely accurate and certain, the performance of the filters will appear to be invariably poor. this oparameter may be selected depending on the notion of successful estimation in the particular scenario at hand. If only estimation which is perfect is to be rewarded, then the kernel should be made strict by selecting a narrow kernel covariance.

# ### Conclusions
# We have presented the quadratic distance in the context of sensor management and performance assessment. We encourage the incorporation of implementations for different kernels and different filter parameterisations. If one wishes to do this, please copy and paste the current implementation for the Gaussian kernel or GM-PHD filter and alter it accordingly to accomodate the desired kernel and filter.

# ### References
# [1] Daniel E. Clark, Idyano Leroy, Peter R. Richards, Sean M. O Rourke, Quadratic error 
#    for point patterns. TechRxiv. July, 2025.
# 
# [2] Daniel E. Clark, Peter R. Richards, Sean M. O Rourke, A Functional Quadratic Form Distance for Multi-Target Tracking Performance Assessment. 
# 28th International Conference on Information Fusion. July, 2025.
# 
# [3] Peter R. Richards, Idyano Leroy, Daniel E. Clark, A Quadratic Reward for Information-Driven Sensor
# Management in Multi-Target Tracking. 29th International Conference on Information Fusion. June, 2026.
# 
# [4] Marc G. Genton, Classes of Kernels for Machine Learning: 
# A Statistics Perspectiv, Journal of Machine Learning Research 2, pp. 299-312, 2001.e
# 
