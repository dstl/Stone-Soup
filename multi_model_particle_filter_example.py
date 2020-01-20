from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity, ConstantAcceleration, ConstantTurn
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from functions_for_particle_filter import import_track_data, create_prior
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
from datetime import datetime
from random import random, randint, seed
from tqdm import tqdm
import numpy as np
import os

seed(100)
DRONE_FILE = 15
DATA_DIR = "P:/DASA/EDITTS Drone Tracking/GFI/GPS Tracking"
SAVE_DIR = "C:/Work/Drone_Tracking/EDITTS-Drone-Tracking/ThirdParty/Stone-Soup/multi_model_results"

NUMBER_OF_PARTICLES = 2
noise_covariance = 0.01
constant_turn_covariance = [0.1, 0.1]
turn_rate = 0.1
DATA_REDUCTION = 1  # (0, 1]
percentage_of_first_model = 1  # (0, 1]

# Import in the track data of the object you wish to follow, here contains position (x, y, z) and time t, but only need
# require, position of the object.
location = import_track_data(DRONE_FILE, DATA_REDUCTION, DATA_DIR)

# Now create the GroundTruth by iterating through the imported track and appending the GroundTruthState to the variable
# truth, with the state_vector as the position at each time step and assuming the timestamp to be increasing by 1.
truth = GroundTruthPath()
start_time = datetime.now()
for t, element in enumerate(location):
    position = np.array([element[0], element[1], element[2]])
    position = position.reshape(3, 1)
    truth.append(GroundTruthState(state_vector=position, timestamp=start_time + timedelta(seconds=t)))

# Just getting the name of the file as to have a unique way to save the results later.
file_list = os.listdir(DATA_DIR)
title_parse = file_list[DRONE_FILE].lower().split(" ")

# Iterate through the truth and form the Detection objects, this will create our measurements used later in the PF
measurements = []
for i in truth:
    measurements.append(Detection(i.state_vector.ravel(), timestamp=i.timestamp))

# Create a tuple of CombinedLinearGaussianTransitionModels (the tuple can consist of one element), these models will be
# used within the Particle Filter.
# The data I am using contains x, y and z so consists of 9 state vectors since we have position, velocity and
# acceleration in all 3 dimensions
dynamic_model_list = (
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

                      CombinedLinearGaussianTransitionModel((ConstantTurn(constant_turn_covariance, turn_rate),
                                                             ConstantVelocity(noise_covariance)))

                       )

# Now form the transition matrix, this holds the probabilities that one model can switch into another. Row 1 represents
# the first entry in the dynamic_model_list, row 2 represents the second entry and so on.
# For instance with this example there is a 2% (0.02) chance for our ConstantVelocity model to change into a
# ConstantAcceleration model, and a 55% (0.55) chance for our ConstantTurn model to stay as ConstantTurn.
transition = (
              (0.9, 0.02, 0.02, 0.02, 0.02, 0.02),     # CA CA CA
              (0.10, 0.7, 0.05, 0.05, 0.05, 0.05),     # CV CV CV
              (0.2, 0.05, 0.60, 0.05, 0.05, 0.05),     # CV CV CA
              (0.1, 0.05, 0.05, 0.7, 0.05, 0.05),      # CA CV CV
              (0.1, 0.05, 0.05, 0.05, 0.7, 0.05),      # CV CA CV
              (0.25, 0.05, 0.05, 0.05, 0.05, 0.55),    # CT CV
              )

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
model_mapping = (
                 (0, 1, 3, 4, 6, 7),           # CV CV CV
                 (0, 1, 2, 3, 4, 5, 6, 7, 8),  # CA CA CA
                 (0, 1, 3, 4, 6, 7, 8),        # CV CV CA
                 (0, 1, 2, 3, 4, 6, 7),        # CA CV CV
                 (0, 1, 3, 4, 5, 6, 7),        # CV CA CV
                 (0, 1, 3, 4, 6, 7)            # CT CV
                 )

# Now we want to create the MultiModel predictor, this class allows the PF to use multiple models in its predictions.
multi_model = MultiModelPredictor(transition, dynamic_model_list, model_mapping,
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
for measurement in tqdm(measurements):
    prediction = multi_model.predict(prior_state, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)
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
plt.savefig(f"{SAVE_DIR}/{file_list[DRONE_FILE]}/{NUMBER_OF_PARTICLES}"
            f" Particles, {DATA_REDUCTION * 100}% Data, {noise_covariance} Noise Covariance.png", dpi=2000)
plt.show()

# Create a basic metric that calculates the difference between the path the particle filter predicted and the true
# path that the object followed.
difference = [np.linalg.norm(track[i].state_vector[[0, 3, 6]] - truth[i].state_vector) for i in range(len(truth))]
print(difference)

# Plot this difference metric to asses how well our model performed.
plt.plot(range(len(difference)), difference)
plt.xlabel('Timestep')
plt.ylabel('Difference')
plt.savefig(f"{SAVE_DIR}/{file_list[DRONE_FILE]}/{NUMBER_OF_PARTICLES}"
            f" Particles, {DATA_REDUCTION * 100}% Data, {noise_covariance} Noise Covariance Metric.png", dpi=2000)
plt.show()
