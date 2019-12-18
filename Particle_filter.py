from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity, ConstantAcceleration, ConstantTurn
from functions_for_particle_filter import create_truth_data, create_prior, compute_metric
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.types.numeric import Probability
from stonesoup.types.state import ParticleState
from stonesoup.types.detection import Detection
from stonesoup.types.particle import Particle
from scipy.stats import multivariate_normal
from stonesoup.types.track import Track
import matplotlib.pyplot as plt
from datetime import datetime
from random import random
from tqdm import tqdm
import numpy as np
import os

time = datetime.now()
np.random.seed(100)
# 20 Bugged,
DRONE_FILE = 0


# Creates the Ground Truth Path, the initial start time of the model and the original x,y,z,t path
NUMBER_OF_PARTICLES = 100
NOISE_COVARIANCE = 0.1
DATA_REDUCTION = 1  # (0, 1]
DATA_DIR = "C:/Users/i_jenkins/Documents/Python Scripts"
SAVE_DIR = "C:/Work/Drone_Tracking/EDITTS-Drone-Tracking/scripts/Particle_Filter_Experiments/particle_filter_results"
truth, start_time, location = create_truth_data(DRONE_FILE, DATA_REDUCTION)

file_list = os.listdir(DATA_DIR)
title_parse = file_list[DRONE_FILE].lower().split(" ")


measurements = []
for i in truth:
    measurements.append(Detection(i.state_vector.ravel(), timestamp=i.timestamp))

transition_model = CombinedLinearGaussianTransitionModel((ConstantVelocity(NOISE_COVARIANCE),
                                                          ConstantVelocity(NOISE_COVARIANCE),
                                                          ConstantVelocity(NOISE_COVARIANCE)))
predictor = ParticlePredictor(transition_model)

measurement_model_track = LinearGaussian(
    6,  # Number of state dimensions (position and velocity in 3D)
    (0, 2, 4),  # Mapping measurement dimensions to state dimensions
    np.diag([0.1, 0.1, 0.1]))  # Covariance matrix for Gaussian PDF

measurement_model_truth = LinearGaussian(
    3,  # Number of state dimensions (position in 3D)
    (0, 1, 2),  # Mapping measurement dimensions to state dimensions
    np.diag([0.1, 0.1, 0.1]))  # Covariance matrix for Gaussian PDF

resampler = SystematicResampler()
updater = ParticleUpdater(measurement_model_track, resampler)

x_0, v_x, a_x, y_0, v_y, a_y, z_0, v_z, a_z = create_prior(location)

samples = multivariate_normal.rvs(np.array([x_0, v_x, a_x, y_0, v_y, a_y, z_0, v_z, a_z]),
                                  np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1]), size=NUMBER_OF_PARTICLES)

# Now when defining the particles, must define which dynamic model they are using
# 1 = ConstantVelocity, 2 = ConstantAcceleration, 3 = ConstantTurn
particles = [Particle(sample.reshape(-1, 1), prior_state=sample.reshape(-1, 1),
                      weight=Probability(1/NUMBER_OF_PARTICLES), dynamic_model=0) for sample in samples]

prior = ParticleState(particles, timestamp=start_time)

particle_track = []
track = Track()
for measurement in tqdm(measurements):
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    particle_track.append(prediction)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

try:
    os.mkdir(f"{SAVE_DIR}/{file_list[DRONE_FILE]}")
except FileExistsError:
    print("Folder already exists")

ax = plt.axes(projection="3d")
print(track[0].state_vector)
print(track[0].particles[0].prior_state)
print("stupid bug")
ax.plot3D(np.array([state.particles[0].prior_state[0] for state in track]).flatten(),
          np.array([state.particles[0].prior_state[3] for state in track]).flatten(),
          np.array([state.particles[0].prior_state[6] for state in track]).flatten(), color='c', label='PF')

ax.plot3D(np.array([state.state_vector[0] for state in truth]).flatten(),
          np.array([state.state_vector[1] for state in truth]).flatten(),
          np.array([state.state_vector[2] for state in truth]).flatten(), linestyle="--", color='coral', label='Truth')

# for particle in particle_track:
#     particle = particle.particles
#     for positions in particle:
#         positions = positions.state_vector
#         particle_positions = [positions[i] for i in range(6) if i % 2 == 0]
#         ax.scatter(particle_positions[:][0], particle_positions[:][1], particle_positions[:][2])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(loc="upper right")
# plt.savefig(f"{SAVE_DIR}/{file_list[DRONE_FILE]}/{title_parse[3]} Drone with {NUMBER_OF_PARTICLES}"
#             f" particles with {DATA_REDUCTION * 100}% Noise Covariance {NOISE_COVARIANCE} data.png", dpi=2000)
plt.show()

track = track[:1250]
truth = truth[:1250]

# Will compute and plot the gospa metric, requires: Track path, Truth path, cutoff distance between track and truth and
# a norm value p, the gospa metric here shows the distance between the truth and its respective track estimate
compute_metric(track, truth, SAVE_DIR, DRONE_FILE, title_parse, NUMBER_OF_PARTICLES,
               DATA_REDUCTION, cutoff_distance=10, p=2)
