#!/usr/bin/env python

import copy
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.types.detection import Detection, TrueDetection, Clutter
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.state import GaussianState
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, ExtendedKalmanUpdater, UnscentedKalmanUpdater
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.smoother.kalman import KalmanSmoother
from stonesoup.types.array import StateVector

from stonesoup.tracker.pmht import ProbabilisticMultiHypothesisTracker


def read_meas(filename, measurement_model, timesteps):
    measdim = 2
    meas = []
    f = open(filename, 'r')

    scannum = 0
    for line in f.readlines():
        fields = line.split(' ')
        nmeas = int(fields[0])
        ptr = 1
        measurement_set = set()
        for m in range(nmeas):
            thismeas = np.zeros((measdim,))
            for k in range(measdim):
                thismeas[k] = float(fields[ptr])
                ptr += 1
            measurement_set.add(Detection(state_vector=thismeas, timestamp=timesteps[scannum],
                                          measurement_model=measurement_model))
        meas.append(measurement_set)
        scannum += 1
    f.close()
    return meas

def read_truth(filename, timesteps):
    ntargets = 2
    statedim = 4
    truths = [GroundTruthPath() for i in range(ntargets)]
    f = open(filename, 'r')

    scannum = 0
    for line in f.readlines():
        fields = line.split(' ')
        ptr = 0
        for t in range(ntargets):
            thistruth = np.zeros((statedim,))
            for k in range(statedim):
                thistruth[k] = float(fields[ptr])
                ptr += 1
            truths[t].append(GroundTruthState(thistruth, timestamp=timesteps[scannum]))
        scannum += 1
    return truths


def simulate_truth_and_measurements(init_means, timesteps, transition_model, measurement_model,
                                    prob_detect):

    # Simulate initial state from truth (probably should sample from initial mean)
    truths = [GroundTruthPath(GroundTruthState(x, timestamp=timesteps[0])) for x in init_means]
    for k in range(1, num_steps):
        for truth in truths:
            truth.append(GroundTruthState(
                transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
                timestamp=timesteps[k]))

    all_measurements = [set()] # no measurements on initial target states
    for k in range(1, num_steps):
        measurement_set = set()
        for truth in truths:
            # Generate actual detection from the state with a 10% chance that no detection is received.
            if np.random.rand() <= prob_detect:
                measurement = measurement_model.function(truth[k], noise=True)
                measurement_set.add(TrueDetection(state_vector=measurement,
                                                  groundtruth_path=truth,
                                                  timestamp=truth[k].timestamp,
                                                  measurement_model=measurement_model))
            # Generate clutter at this time-step (if we later set lambda > 0)
            #truth_x = truth[k].state_vector[0]
            #truth_y = truth[k].state_vector[2]
            #for _ in range(np.random.poisson(np.exp(log_clutter_density)*clutter_V)):
            #    x = uniform.rvs(truth_x - 10, 20)
            #    y = uniform.rvs(truth_y - 10, 20)
            #    measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp,
            #                                measurement_model=measurement_model))
        all_measurements.append(measurement_set)

    return truths, all_measurements


np.random.seed(1991)#(2991)#

start_time = datetime.now().replace(microsecond=0)

# We want a 2d simulation, so we'll do:
q_x = 1.0#0.1#
q_y = 1.0#0.1#
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])

measurement_model = LinearGaussian(
    ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0, 2),  # Mapping measurement vector index to state index
    noise_covar=np.array([[1, 0],  # Covariance matrix for Gaussian PDF
                          [0, 1]])
    )

prob_detect = 0.9 # assume no clutter or missed detections for now
log_clutter_density = -20
log_clutter_volume = np.log(10000)

num_steps = 50# len(all_measurements) #
timesteps = [start_time]
for k in range(1, num_steps):
    timesteps.append(start_time + timedelta(seconds=k))

# Initial estimate for tracks
init_means = [[-20, 5, 0, 10], [20, -5, 0, 10]]
init_cov = np.diag([1.0, 1.0, 1.0, 1.0])

#truths = read_truth('Data/truth.txt', timesteps)
#all_measurements = read_meas('Data/meas.txt', measurement_model, timesteps)

truths, all_measurements = simulate_truth_and_measurements(init_means, timesteps, transition_model, measurement_model,
                                                           prob_detect)

# Annoying unpacking because GaussianState wants single component values
#init_priors = [GaussianState([[init_mean[k]] for k in range(transition_model.ndim)],
#                        init_cov, timestamp=start_time) for init_mean in init_means]
init_priors = [GaussianState(StateVector(init_mean), init_cov, timestamp=start_time) for init_mean in init_means]

batch_len = 50#10
overlap_len = 0#5
maxniter = 10
update_log_pi = True#False#

predictor = KalmanPredictor(transition_model)
smoother = KalmanSmoother(transition_model)
updater = UnscentedKalmanUpdater(measurement_model) # KalmanUpdater(measurement_model)

# Initialise PMHT
pmht = ProbabilisticMultiHypothesisTracker(
    log_clutterDensity = log_clutter_density, log_clutterVolume = log_clutter_volume,
    prob_detect = prob_detect, predictor = predictor, smoother = smoother, updater = updater,
    init_priors = init_priors, overlap_len = overlap_len)

# Add first batch of measurements
pmht.add_measurements(all_measurements[:batch_len], timesteps[:batch_len])

# Run PMHT on this
pmht.do_iterations(maxniter, update_log_pi = update_log_pi)# all_measurements[:batch_len], timesteps[:batch_len],

old_index = batch_len

# Run remaining measurement batches
while old_index < num_steps:

    new_index = old_index + (batch_len - overlap_len)
    if new_index > num_steps:
        new_index = num_steps
    pmht.add_measurements(all_measurements[old_index:new_index], timesteps[old_index:new_index])
    pmht.do_iterations(maxniter, update_log_pi)
    old_index = new_index

#for x in pmht.tracks[0]:
#    print(x.covar)

# Plot the resulting tracks
for truth in truths:
    truth_x = np.array([x.state_vector.flatten() for x in truth.states])
    plt.plot(truth_x[:, 0], truth_x[:, 2], color='g', marker='s')
for track in pmht.tracks:
    track_x = np.array([x.state_vector.flatten() for x in track])
    plt.plot(track_x[:, 0], track_x[:, 2], color='r', marker='s')
for scan in all_measurements[1:]:
    if len(scan) > 0:
        z = np.array([m.state_vector.flatten() for m in scan])
        plt.plot(z[:, 0], z[:, 1], color='b', marker='x', linestyle='')
#plt.axis('equal')
plt.show()
