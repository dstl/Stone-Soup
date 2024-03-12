import numpy as np
import datetime


import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


np.random.seed(1000)

# %%
# Initialise ground truth
# ^^^^^^^^^^^^^^^^^^^^^^^
# Here are some configurable parameters associated with the ground truth. We specify a fixed number of initial
# targets with no births and deaths.

# The simulator requires a distribution for birth targets, so we specify a dummy one.
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState
initial_state_mean = StateVector([[0], [0], [0], [0]])
initial_state_covariance = CovarianceMatrix(np.eye(4))
start_time = datetime.datetime.now().replace(microsecond=0)
initial_state = GaussianState(initial_state_mean, initial_state_covariance, start_time)
timestep_size = datetime.timedelta(seconds=1)
number_of_steps = 50

# Initial truth states for fixed number of targets
preexisting_states = [[-20, 5, 0, 10], [20, -5, 0, 10]]

# %%
# Create the transition model - default set to 2d nearly-constant velocity
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
q_x = 1.0
q_y = 1.0
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])

# %%
# Put this all together in a multi-target simulator.
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=initial_state,
    preexisting_states=preexisting_states,
    timestep=timestep_size,
    number_steps=number_of_steps,
    birth_rate=0.0,
    death_probability=0.0
)

# %%
# Initialise the measurement models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The simulated ground truth will then be passed to a simple detection simulator. This again has a
# number of configurable parameters, e.g. where clutter is generated and at what rate, and
# detection probability. This implements similar logic to the code in the previous tutorial section
# :ref:`auto_tutorials/09_Initiators_&_Deleters:Generate Detections and Clutter`.
from stonesoup.models.measurement.linear import LinearGaussian
measurement_model = LinearGaussian(
    ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0, 2),  # Mapping measurement vector index to state index
    noise_covar=np.array([[1, 0],  # Covariance matrix for Gaussian PDF
                          [0, 1]])
    )

# probability of detection
detection_probability = 0.9

# clutter will be generated uniformly in this are around the target
meas_range = np.array([[-1, 1], [-1, 1]])*1000

# currently use very low clutter rate since PMHT seems to struggle with clutter
# rate is in mean number of clutter points per scan
clutter_rate = 1.0e-3

# %%
# The detection simulator
from stonesoup.simulator.simple import SimpleDetectionSimulator
detection_sim = SimpleDetectionSimulator(
    groundtruth=groundtruth_sim,
    measurement_model=measurement_model,
    detection_probability=detection_probability,
    meas_range=meas_range,
    clutter_rate=clutter_rate
)

# %%
# Create the tracker components
# -----------------------------
# In this example a Kalman filter is used with global nearest neighbour (GNN) associator. Other
# options are, of course, available.
#

# %%
# Predictor
# ^^^^^^^^^
# Initialise the predictor using the same transition model as generated the ground truth. Note you
# don't have to use the same model.
# We also need to specify a smoother since PMHT does smoothing on batches of measurements
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.smoother.kalman import KalmanSmoother
predictor = KalmanPredictor(transition_model)
smoother = KalmanSmoother(transition_model)

# %%
# Updater
# ^^^^^^^
# Initialise the updater using the same measurement model as generated the simulated detections.
# Note, again, you don't have to use the same model (noise covariance).
from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# %%
# Data associator, Initiator and Deleter
# ^^^^^^^^^^^^^^^
# The algorithm currently assumes a fixed number of tracks, with data association built in, so these components are not
# required

# Initial estimate for tracks
init_means = preexisting_states
init_cov = np.diag([1.0, 1.0, 1.0, 1.0])
init_priors = [GaussianState(StateVector(init_mean), init_cov, timestamp=start_time) for init_mean in init_means]

# %%
# Run the Tracker
# ---------------
# With the components created, the multi-target tracker component is created, constructed from
# the components specified above.

# Number of measurement scans to run over for each batch
batch_len = 10

# Number of scans to overlap between batches
overlap_len = 5

# Maximum number of iterations to run each batch over (currently there is no convergence test so this is the actual
# number of iterations)
max_num_iterations = 10

# Whether to update the prior data association values during iterations (True or False)
update_log_pi = True

from stonesoup.tracker.pmht_tracker import PMHTTracker

pmht = PMHTTracker(
    detector=detection_sim,
    predictor=predictor,
    smoother=smoother,
    updater=updater,
    meas_range=meas_range,
    clutter_rate=clutter_rate,
    detection_probability=detection_probability,
    batch_len=batch_len,
    overlap_len=overlap_len,
    init_priors=init_priors,
    max_num_iterations=max_num_iterations,
    update_log_pi=update_log_pi)

groundtruth = set()
#detections = set()
tracks = set()

for time, ctracks in pmht:
    print(time)
    groundtruth.update(groundtruth_sim.groundtruth_paths)
    #detections.update(detection_sim.detections)
    tracks.update(ctracks)

# %%
# And plot them:
# (This could be replaced with AnimatedPlotterly, but I (PRH) prefer to test with matplotlib)
#
for truth in groundtruth:
    truth_x = np.array([x.state_vector.flatten() for x in truth.states])
    plt.plot(truth_x[:, 0], truth_x[:, 2], color='g', marker='s')
#for detection in detections:
#        z = detection.state_vector.flatten()
#        plt.plot(z[0], z[1], color='b', marker='x', linestyle='')
for track in pmht.tracks:
    track_x = np.array([x.state_vector.flatten() for x in track])
    plt.plot(track_x[:, 0], track_x[:, 2], color='r', marker='s')
plt.show()
