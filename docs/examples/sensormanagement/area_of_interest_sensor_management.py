#!/usr/bin/env python

"""
============================
3 - Areas of interest/access
============================
"""

# %%
# This notebook introduces sensor management which factors in environmental information
# such as areas of interest.

# %%
# Setting Up the Scenario
# -----------------------
# We generate a ground truth of a target following a constant velocity with an amount of
# noise that makes the truth interesting.

import numpy as np
from datetime import datetime, timedelta


from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantVelocity)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

np.random.seed(1990)

start_time = datetime.now().replace(second=0, microsecond=0)

transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(10), ConstantVelocity(10)])

truths = []
truth = GroundTruthPath([GroundTruthState([-450, 5, 450, -5], timestamp=start_time)])
duration = 120
timesteps = [start_time]

for k in range(1, duration):
    timesteps.append(start_time+timedelta(seconds=k))
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))
truths.append(truth)

# %%
# Visualising the ground truth
# ----------------------------

from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps, tail_length=1)
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig

# %%
# Creating the Platform and Sensor
# --------------------------------

from stonesoup.platform import MovingPlatform
from stonesoup.movable.max_speed import MaxSpeedActionableMovable
from stonesoup.sensor.radar.radar import RadarRotatingBearingRange
from stonesoup.types.angle import Angle
from stonesoup.types.state import State, StateVector

sensor = RadarRotatingBearingRange(
    position_mapping=(0, 2),
    noise_covar=np.array([[np.radians(5)**2, 0],
                          [0, 20**2]]),
    ndim_state=4,
    rpm=30,
    fov_angle=np.radians(360),
    dwell_centre=StateVector([np.radians(0)]),
    max_range=300,

    resolution=Angle(np.radians(360)))

target_sensor1 = RadarRotatingBearingRange(
    position_mapping=(0, 2),
    noise_covar=np.array([[np.radians(1)**2, 0], [0, 1**2]]),
    ndim_state=4,
    rpm=30,
    fov_angle=np.radians(360),
    dwell_centre=StateVector([np.radians(90)]),
    max_range=100,
    resolution=Angle(np.radians(180)))

platform = MovingPlatform(
    movement_controller=MaxSpeedActionableMovable(
        states=[State([-500, 500],
                      timestamp=start_time)],
        position_mapping=(0, 1),
        action_mapping=(0, 1),
        resolution=10,
        angle_resolution=np.pi/2,
        max_speed=100),
    sensors=[sensor])

# %%
# Creating a Predictor and Updater
# --------------------------------
#
# Next we need some tracking components: a predictor and an updater.

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model=None)

# %%
# Creating Areas of Interest
# --------------------------
# The default area is defined by the x, y minimum and maximum coordinates as
# :math:`x, y \in (-\infty, \infty)`.

from stonesoup.sensormanager.shape import AreaOfInterest

area1 = AreaOfInterest(xmax=0, interest=4, access=0)
area2 = AreaOfInterest(xmin=0, interest=7, access=5)
area3 = AreaOfInterest(xmin=1500, interest=10, access=9)

# %%
# Creating a Sensor Manager
# -------------------------

from stonesoup.sensormanager.reward import (
    UncertaintyRewardFunction, MultiplicativeRewardFunction, FOVInteractionRewardFunction,
    AOIAccess2DRewardFunction)

from stonesoup.sensormanager.base import BruteForceSensorManager

reward_func_A = UncertaintyRewardFunction(predictor, updater)

reward_func_B = FOVInteractionRewardFunction(
    predictor, updater, sensor_fov_radius=sensor.max_range,
    target_fov_radius=target_sensor1.max_range,
    fov_scale=0.75)

reward_func_C = FOVInteractionRewardFunction(
    predictor, updater, sensor_fov_radius=sensor.max_range,
    target_fov_radius=target_sensor1.max_range + 100,  # extra cautious!
    fov_scale=1)

reward_func_AB = MultiplicativeRewardFunction([reward_func_A, reward_func_B])
reward_func_AC = MultiplicativeRewardFunction([reward_func_A, reward_func_C])


reward_func_aoi = AOIAccess2DRewardFunction(default_reward=reward_func_AC,
                                            interest_thresholds={3: reward_func_A,
                                                                 5: reward_func_AB,
                                                                 9: reward_func_AC},
                                            areas=[area1, area2, area3],
                                            target_mapping=[0, 2])

sensormanager = BruteForceSensorManager(sensors={sensor},
                                        platforms={platform},
                                        reward_function=reward_func_aoi,)

# %%
# Creating a Track
# ----------------

from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

prior1 = GaussianState(truths[0][0].state_vector,
                       covar=np.diag([0.5, 0.5, 0.5, 0.5] + np.random.normal(0, 5e-4, 4)),
                       timestamp=start_time)

tracks = {Track([prior1])}

# %%
# Creating a Hypothesiser and Data Associator
# -------------------------------------------
#
# The final tracking components required are the hypothesiser and data associator.

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=5)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)

# %%
# Running the Tracking Loop
# -------------------------
#
# At each timestep the sensor manager generates the optimal actions for our sensor and
# platform. The sensor manager's :meth:`~.SensorManager.choose_actions` method is called.

from ordered_set import OrderedSet
from collections import defaultdict
import copy
from stonesoup.measures import Euclidean
from stonesoup.sensor.sensor import Sensor

all_measurements = set()
sensor_history = defaultdict(dict)
for timestep in timesteps[1:]:
    print(timestep)

    chosen_actions = sensormanager.choose_actions(tracks, timestep)
    measurements = set()

    for chosen_action in chosen_actions:
        for actionable, actions in chosen_action.items():
            actionable.add_actions(actions)
            actionable.act(timestep)
            if isinstance(actionable, Sensor):
                measurements |= actionable.measure(OrderedSet(truth[timestep] for truth in truths),
                                                   noise=True)

    all_measurements.update(measurements)
    sensor_history[timestep][sensor] = copy.deepcopy(sensor)
    hypotheses = data_associator.associate(tracks, measurements, timestep)
    for track in tracks:
        hypothesis = hypotheses[track]
        if hypothesis.measurement:
            post = updater.update(hypothesis)
            track.append(post)
        else:
            track.append(hypothesis.prediction)

# %%
# Plotting
# --------
#
# The FOV-based reward function in an actionable platform is able to follow the target as it moves
# across the action space, keeping it within the sensor's range but outside the FOV of the target.
# As the target moves through areas of interest, the sensor manager adjusts the sensor's FOV to
# prioritise tracking the target despite the change of access and interest parameters of each area.

from stonesoup.plotter import plot_sensor_fov
import plotly.graph_objects as go
from stonesoup.platform.base import PathBasedPlatform

plotter = AnimatedPlotterly(timesteps, tail_length=0.1)
plotter.plot_ground_truths(truths, [0, 2], mode='lines', line=dict(dash=None))
plotter.plot_tracks(tracks, mapping=(0, 2))
plotter.plot_measurements(all_measurements, mapping=(0, 2))

track_list = list(tracks)
target_platform1 = PathBasedPlatform(path=track_list[0], sensors=[target_sensor1],
                                     position_mapping=[0, 2])
target_sensor_history1 = defaultdict(dict)

for timestep in timesteps[1:]:
    target_platform1.move(timestep)
    target_sensor_history1[timestep][target_sensor1] = copy.deepcopy(target_sensor1)
target_sensor_set1 = {target_sensor1}
sensor_set = {sensor}

plot_sensor_fov(plotter.fig, sensor_set, sensor_history)

plot_sensor_fov(plotter.fig, target_sensor_set1, target_sensor_history1, label="Target FOV",
                color="red")
plotter.fig.add_trace(go.Scatter(x=[1500, 1500], y=[-300, 500], mode='lines',
                                 line=dict(color='#888'),
                                 name='Area Boundaries',
                                 showlegend=False))
plotter.fig.add_trace(go.Scatter(x=[0, 0], y=[-300, 500], mode='lines',
                                 line=dict(color='#888'),
                                 name='Area Boundaries',
                                 showlegend=False))
plotter.fig.add_trace(go.Scatter(x=[-400, 800, 2000],
                                 y=[0, 0, 0], mode="text", name="Areas of Interest",
                                 line=dict(color='#888'),
                                 text=["Interest: 4</br></br>Access: 0",
                                       "Interest: 7</br></br>Access: 5",
                                       "Interest: 10</br></br>Access: 9"]))
plotter.fig

# %%
# Metrics
# -------
#

from stonesoup.metricgenerator.ospametric import OSPAMetric
ospa_generatorA = OSPAMetric(c=40, p=1,
                             generator_name='3 areas of interest',
                             tracks_key='tracksA',
                             truths_key='truths')

from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
siap_generatorA = SIAPMetrics(position_measure=Euclidean((0, 2)),
                              velocity_measure=Euclidean((1, 3)),
                              generator_name='3 areas of interest',
                              tracks_key='tracksA',
                              truths_key='truths')

from stonesoup.dataassociator.tracktotrack import TrackToTruth
associator = TrackToTruth(association_threshold=30)

from stonesoup.metricgenerator.uncertaintymetric import SumofCovarianceNormsMetric
uncertainty_generatorA = SumofCovarianceNormsMetric(generator_name='3 areas of interest',
                                                    tracks_key='tracksA')

from stonesoup.metricgenerator.manager import MultiManager

metric_manager = MultiManager([ospa_generatorA,
                               siap_generatorA,
                               uncertainty_generatorA],
                              associator=associator)

metric_manager.add_data({'truths': truths, 'tracksA': tracks})

metrics = metric_manager.generate_metrics()

# %%
# Plot OSPA metric

from stonesoup.plotter import MetricPlotter

fig = MetricPlotter()
fig.plot_metrics(metrics, metric_names=['OSPA distances'])

# %%
# Plot SIAP metrics

fig2 = MetricPlotter()
fig2.plot_metrics(metrics, metric_names=['SIAP Position Accuracy at times',
                                         'SIAP Velocity Accuracy at times'])

# %%
# Plot uncertainty metric

fig3 = MetricPlotter()
fig3.plot_metrics(metrics, metric_names=['Sum of Covariance Norms Metric'])
