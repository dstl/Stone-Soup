from functions_for_particle_filter import import_track_data, create_prior, read_synthetic_csv
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity, ConstantAcceleration, ConstantTurn, ConstantPosition, LinearTurn
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.predictor.multi_model import MultiModelPredictor
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.types.numeric import Probability
from stonesoup.types.state import ParticleState
from stonesoup.types.detection import Detection
from stonesoup.types.particle import Particle
from scipy.stats import multivariate_normal
from stonesoup.types.track import Track
from datetime import timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from random import random, randint, seed
from tqdm import tqdm
import numpy as np
import os

seed(100)
DRONE_FILE = -11
DATA_DIR = "P:/DASA/EDITTS Drone Tracking/GFI/GPS Tracking"
# DATA_DIR = "C:/Work/Drone_Tracking/EDITTS-Drone-Tracking/data/raw/"
SAVE_DIR = "C:/Work/Drone_Tracking/multi_model_results"
FIXED_WING = {"g2", "g4", "maja", "bixler", "x8", "kahu"}
ROTARY_WING = {"g6", "f550", "drdc"}

NUMBER_OF_PARTICLES = 350
noise_covariance = 0.01
constant_turn_covariance = [0.1, 0.1]
turn_rate_left = 0.5
turn_rate_right = -0.5
DATA_REDUCTION = 1  # (0, 1]
percentage_of_first_model = 1  # (0, 1]

# Just getting the name of the file as to have a unique way to save the results later.

file_list = os.listdir(DATA_DIR)
print(file_list)
print(file_list[DRONE_FILE])
title_parse = file_list[DRONE_FILE].lower().split(" ")
if title_parse[3] in FIXED_WING:
    print("Fixed Wing")
    SAVE_DIR = "C:/Work/Drone_Tracking/multi_model_results/Fixed_Wing"
elif title_parse[3] in ROTARY_WING:
    print("Rotary Wing")
    SAVE_DIR = "C:/Work/Drone_Tracking/multi_model_results/Rotary_Wing"

# Import in the track data of the object you wish to follow, here contains position (x, y, z) and time t, but only need
# require position of the object.

location = import_track_data(DRONE_FILE, DATA_REDUCTION, DATA_DIR)
# location = read_synthetic_csv(DATA_DIR + file_list[DRONE_FILE])
location = location[int(len(location) * 0.15): int(len(location) * 0.20)]

# Now create the GroundTruth by iterating through the imported track and appending the GroundTruthState to the variable
# truth, with the state_vector as the position at each time step and assuming the timestamp to be increasing by 1.

truth = GroundTruthPath()
start_time = datetime.now()
for t, element in enumerate(location):
    position = np.array([element[0], element[1], element[2]])
    position = position.reshape(3, 1)
    truth.append(GroundTruthState(state_vector=position, timestamp=start_time + timedelta(seconds=t)))

# Iterate through the truth and form the Detection objects, this will create our measurements used later in the PF

measurements = []
for i in truth:
    measurements.append(Detection(i.state_vector.ravel(), timestamp=i.timestamp))

# Create a tuple of CombinedLinearGaussianTransitionModels (the tuple can consist of one element), these models will be
# used within the Particle Filter.
# The data I am using contains x, y and z so consists of 9 state vectors since we have position, velocity and
# acceleration in all 3 dimensions

dynamic_model_list = [
                      CombinedLinearGaussianTransitionModel((ConstantVelocity(noise_covariance),
                                                             ConstantVelocity(noise_covariance),
                                                             ConstantVelocity(noise_covariance))),
                      CombinedLinearGaussianTransitionModel((ConstantAcceleration(noise_covariance),
                                                             ConstantAcceleration(noise_covariance),
                                                             ConstantAcceleration(noise_covariance))),
                      CombinedLinearGaussianTransitionModel((ConstantVelocity(noise_covariance),
                                                             ConstantVelocity(noise_covariance),
                                                             ConstantAcceleration(noise_covariance))),
                      CombinedLinearGaussianTransitionModel((ConstantAcceleration(noise_covariance),
                                                             ConstantVelocity(noise_covariance),
                                                             ConstantVelocity(noise_covariance))),
                      CombinedLinearGaussianTransitionModel((ConstantVelocity(noise_covariance),
                                                             ConstantAcceleration(noise_covariance),
                                                             ConstantVelocity(noise_covariance))),
                      CombinedLinearGaussianTransitionModel((ConstantTurn(constant_turn_covariance, turn_rate_left),
                                                             ConstantVelocity(noise_covariance))),
                      CombinedLinearGaussianTransitionModel((ConstantTurn(constant_turn_covariance, turn_rate_right),
                                                             ConstantVelocity(noise_covariance))),
                      CombinedLinearGaussianTransitionModel((ConstantAcceleration(noise_covariance),
                                                             ConstantAcceleration(noise_covariance),
                                                             ConstantVelocity(noise_covariance))),
                      CombinedLinearGaussianTransitionModel((ConstantPosition(noise_covariance),
                                                             ConstantPosition(noise_covariance),
                                                             ConstantVelocity(noise_covariance))),
                      CombinedLinearGaussianTransitionModel((LinearTurn(turn_rate_left, noise_covariance),
                                                             ConstantAcceleration(noise_covariance))),
                     ]

# Here we create the State Vector mapping matrix, this tells the predictor which state vectors within our whole state
# space (position, velocity and acceleration in 3D) are needed for each model.
# For example
# Given a state space of [x, vx, ax, y, vy, ay, z, vz, az] and dynamic models:
# model1 = CombinedLinearGaussianTransitionModel(ConstantVelocity, ConstantVelocity, ConstantVelocity)
# model2 = CombinedLinearGaussianTransitionModel(ConstantAcceleration, ConstantAcceleration, ConstantAcceleration)
# The position mapping array would be:
# ((0, 1, 3, 4, 6, 7),
#  (0, 1, 2, 3, 4, 5, 6, 7, 8))
# Since ConstantVelocity requires only position and velocity but ConstantAcceleration requires all 9 state variables.

model_mapping3D = [
                   [0, 1, 3, 4, 6, 7],              # CV CV CV
                   [0, 1, 2, 3, 4, 5, 6, 7, 8],     # CA CA CA
                   [0, 1, 3, 4, 6, 7, 8],           # CV CV CA
                   [0, 1, 2, 3, 4, 6, 7],           # CA CV CV
                   [0, 1, 3, 4, 5, 6, 7],           # CV CA CV
                   [0, 1, 3, 4, 6, 7],              # CTL CV
                   [0, 1, 3, 4, 6, 7],              # CTR CV
                   [0, 1, 2, 3, 4, 5, 6, 7],        # CA CA CV
                   [0, 1, 3, 4, 6, 7],              # H H CV
                   # [0, 1, 2, 3, 4, 5, 6, 7, 8]      # LT CA
                  ]

# Now form the transition matrix, this holds the probabilities that one model can switch into another. Row 1 represents
# the first entry in the dynamic_model_list, row 2 represents the second entry and so on.
# For instance with this example there is a 2% (0.02) chance for our ConstantVelocity model to change into a
# ConstantAcceleration model, and a 55% (0.55) chance for our ConstantTurn model to stay as ConstantTurn.

transition = np.zeros((len(dynamic_model_list), len(dynamic_model_list)))
for i in range(len(dynamic_model_list)):
    for j in range(len(dynamic_model_list)):
        if j == i:
            transition[i][i] = 1 - ((len(transition) - 1) * 0.05)
        else:
            transition[i][j] = 0.05
print(transition)

# Now we want to create the MultiModel predictor, this class allows the PF to use multiple models in its predictions.

multi_model = MultiModelPredictor(transition, dynamic_model_list, model_mapping3D,
                                  transition_model=dynamic_model_list)

# Form a mapping matrix that points to the position state vectors in this case it forms a 3x9 matrix:
# [[1, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 1, 0, 0, 0, 0, 0], These are the positions of our x, y, z position vectors.
#  [0, 0, 0, 0, 0, 0, 1, 0, 0]]

measurement_model = LinearGaussian(
    9,  # Number of state dimensions (position, velocity and acceleration in 3D)
    (0, 3, 6),  # Locations of our position variables within the entire state space
    np.diag([0.1, 0.1, 0.1]))  # Covariance matrix for Gaussian PDF

# Create the resampler and updater

resampler = SystematicResampler()
updater = ParticleUpdater(measurement_model, resampler)

# Create a prior value for our particles, here it is just the initial position, velocity and acceleration.

x_0, v_x, a_x, y_0, v_y, a_y, z_0, v_z, a_z = create_prior(location)

# Create the particles. Here we are creating random positions around the prior value using a diagonal matrix of 1's
# as our covariance matrix, the covariance matrix can take any floating point values, larger values will create a wider
# range of initial particles.

samples = multivariate_normal.rvs(np.array([x_0, v_x, a_x, y_0, v_y, a_y, z_0, v_z, a_z]),
                                  np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1]), size=NUMBER_OF_PARTICLES)

# When declaring the particles you must add the dynamic model attribute, this is an integer that cannot exceed the
# number of models that can be used by the PF. In this case 0 is ConstantVelocity and 1 is ConstantAcceleration
# Below is a way to determine that a certain percentage of the particles begin with a specific model.

start_model = []
for i in range(NUMBER_OF_PARTICLES):
    random_int = random()
    if random_int < percentage_of_first_model:
        start_model.append(0)
    elif random_int >= percentage_of_first_model:
        start_model.append(1)

particles = [Particle(sample.reshape(-1, 1), weight=Probability(1/NUMBER_OF_PARTICLES),
                      dynamic_model=start_model[randint(0, NUMBER_OF_PARTICLES - 1)]) for sample in samples]

# Setting the prior state to be these particles

prior_state = ParticleState(particles, timestamp=start_time)

# Now iterate through the measurements made before, implementing the particle filter on these values.

track = Track()
dynamic_model_split = []
effective_sample_size = []
for iteration, measurement in enumerate(tqdm(measurements)):
    prediction, dynamic_model_proportions = multi_model.predict(prior_state, timestamp=measurement.timestamp)
    dynamic_model_split.append(dynamic_model_proportions)
    hypothesis = SingleHypothesis(prediction, measurement)
    post, n_eff = updater.update(hypothesis)
    print(n_eff)
    if n_eff < 10 and iteration > 30:
        break
    effective_sample_size.append(n_eff)
    track.append(post)
    prior_state = track[-1]

# Creating a folder to store our results in, excepts the fact that the folder may have already been created.

try:
    os.mkdir(f"{SAVE_DIR}/{file_list[DRONE_FILE]}")
except FileExistsError:
    print("Folder already exists")


# Plot our results of the particle filter vs the actual true path the object followed.

number_of_models = len(dynamic_model_list)
ax = plt.axes(projection="3d")

ax.plot3D(np.array([state.particles[0].state_vector[0] for state in track]).flatten(),
          np.array([state.particles[0].state_vector[3] for state in track]).flatten(),
          np.array([state.particles[0].state_vector[6] for state in track]).flatten(), color='c', label='PF')

ax.plot3D(np.array([state.state_vector[0] for state in truth]).flatten(),
          np.array([state.state_vector[1] for state in truth]).flatten(),
          np.array([state.state_vector[2] for state in truth]).flatten(), linestyle="--", color='coral', label='Truth')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(loc="upper right")
plt.savefig(f"{SAVE_DIR}/{file_list[DRONE_FILE]}/{NUMBER_OF_PARTICLES} Particles Number of Models {number_of_models}"
            f" Data Path.png", dpi=2000)
plt.title("True path and predicted PF path")
plt.show()


plt.plot(range(len(effective_sample_size)), effective_sample_size)
plt.xlabel("Timestep")
plt.ylabel("Effective Sample Size")
plt.title("Effective sample size at a given timestep")
plt.savefig(f"{SAVE_DIR}/{file_list[DRONE_FILE]}/{NUMBER_OF_PARTICLES} Particles Number of Models {number_of_models}"
            f" Data Effective Sample Size.png", dpi=2000)
plt.show()


# dynamic_model_split = dynamic_model_split[100:150]
dynamic_model_plot = [[element[j] for element in dynamic_model_split] for j in range(len(dynamic_model_split[0]))]

for i, line in enumerate(dynamic_model_plot):
    plt.plot(range(len(dynamic_model_split)), line)
plt.legend([f"Model {i}" for i in range(len(dynamic_model_plot))])
plt.title("Number of Particles for each model")
plt.xlabel('Timestep')
plt.ylabel('Number of Particles')
plt.savefig(f"{SAVE_DIR}/{file_list[DRONE_FILE]}/{NUMBER_OF_PARTICLES} Particles Number of Models {number_of_models}"
            f" Data Dynamic Model Choice.png", dpi=2000)
plt.show()


# Create a basic metric that calculates the difference between the path the particle filter predicted and the true
# path that the object followed.
difference = [np.linalg.norm(track[i].state_vector[[0, 3, 6]] - truth[i].state_vector) for i in range(len(truth))]
sum_of_difference = sum(difference)
print(difference)
print(sum_of_difference)

# Plot this difference metric to asses how well our model performed.
plt.plot(range(len(difference)), difference)
plt.xlabel('Timestep')
plt.ylabel('Difference')
plt.title("Distance Metric")
plt.text(100, 10, f"Total distance : {sum_of_difference}")
plt.savefig(f"{SAVE_DIR}/{file_list[DRONE_FILE]}/{NUMBER_OF_PARTICLES} Particles Number of Models {number_of_models}"
            f" Metric.png", dpi=2000)
plt.show()
