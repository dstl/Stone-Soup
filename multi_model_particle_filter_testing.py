from functions_for_particle_filter import import_track_data, create_prior, read_synthetic_csv, \
    form_transition_matrix, form_detection_transition_matrix
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity, ConstantAcceleration, ConstantTurn, ConstantPosition, LinearTurn
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.predictor.multi_model import MultiModelPredictor
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.updater.particle import ParticleUpdater, MultiModelParticleUpdater
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
DRONE_FILE = 5
DATA_DIR = "P:/DASA/EDITTS Drone Tracking/GFI/GPS Tracking"
# DATA_DIR = "C:/Work/Drone_Tracking/EDITTS-Drone-Tracking/data/raw/"
SAVE_DIR = "C:/Work/Drone_Tracking/multi_model_results"
FIXED_WING = {"g2", "g4", "maja", "bixler", "x8", "kahu"}
ROTARY_WING = {"g6", "f550", "drdc"}

NUMBER_OF_PARTICLES = 700
rw_cv_noise_covariance = 0.04
fw_cv_noise_covariance = 0.0008
rw_hover_noise_covariance = 0.001
constant_turn_covariance = [0.1, 0.1]
turn_rate_left = 0.5
turn_rate_right = -0.5
DATA_REDUCTION = 1  # (0, 1]
percentage_of_first_model = 0.5  # (0, 1]

file_list = os.listdir(DATA_DIR)
print(file_list)
print(file_list[DRONE_FILE])
title_parse = file_list[DRONE_FILE].lower().split(" ")
if title_parse[3] in FIXED_WING:
    print("Fixed Wing")
    model_type = "Fixed Wing"
    SAVE_DIR = "C:/Work/Drone_Tracking/multi_model_results/Fixed_Wing"
elif title_parse[3] in ROTARY_WING:
    print("Rotary Wing")
    model_type = "Rotary Wing"
    SAVE_DIR = "C:/Work/Drone_Tracking/multi_model_results/Rotary_Wing"

location = import_track_data(DRONE_FILE, DATA_REDUCTION, DATA_DIR)
# location = read_synthetic_csv(DATA_DIR + file_list[DRONE_FILE])

ax = plt.axes(projection="3d")
ax.plot3D(location[:, 0],
          location[:, 1],
          location[:, 2])


# location = location[int(len(location) * 0): int(len(location) * 0.05)]
location = location[500:550]

ax.plot3D(location[:, 0],
          location[:, 1],
          location[:, 2])
plt.show()

truth = GroundTruthPath()
start_time = datetime.now()
for t, element in enumerate(location):
    position = np.array([element[0], element[1], element[2]])
    position = position.reshape(3, 1)
    truth.append(GroundTruthState(state_vector=position, timestamp=start_time + timedelta(seconds=t)))

measurements = []
for i in truth:
    measurements.append(Detection(i.state_vector.ravel(), timestamp=i.timestamp))

dynamic_model_list_RW = [
                        # Rotary Wing
                        CombinedLinearGaussianTransitionModel((ConstantVelocity(rw_cv_noise_covariance),
                                                               ConstantVelocity(rw_cv_noise_covariance),
                                                               ConstantVelocity(rw_cv_noise_covariance))),
                        CombinedLinearGaussianTransitionModel((ConstantPosition(rw_hover_noise_covariance),
                                                               ConstantPosition(rw_hover_noise_covariance),
                                                               ConstantVelocity(rw_hover_noise_covariance))),
                        # CombinedLinearGaussianTransitionModel((LinearTurn(turn_rate_left, rw_cv_noise_covariance),
                        #                                        ConstantAcceleration(rw_cv_noise_covariance))),
                        ]

dynamic_model_list_FW = [
                       # Fixed Wing
                       CombinedLinearGaussianTransitionModel((ConstantVelocity(fw_cv_noise_covariance),
                                                              ConstantVelocity(fw_cv_noise_covariance),
                                                              ConstantVelocity(fw_cv_noise_covariance))),
                       CombinedLinearGaussianTransitionModel((ConstantAcceleration(fw_cv_noise_covariance),
                                                              ConstantAcceleration(fw_cv_noise_covariance),
                                                              ConstantAcceleration(fw_cv_noise_covariance))),
                       # CombinedLinearGaussianTransitionModel((ConstantVelocity(noise_covariance),
                       #                                        ConstantVelocity(noise_covariance),
                       #                                        ConstantAcceleration(noise_covariance))),
                       # CombinedLinearGaussianTransitionModel((ConstantAcceleration(noise_covariance),
                       #                                        ConstantVelocity(noise_covariance),
                       #                                        ConstantVelocity(noise_covariance))),
                       # CombinedLinearGaussianTransitionModel((ConstantVelocity(noise_covariance),
                       #                                        ConstantAcceleration(noise_covariance),
                       #                                        ConstantVelocity(noise_covariance))),
                       # CombinedLinearGaussianTransitionModel((ConstantTurn(constant_turn_covariance, turn_rate_left),
                       #                                        ConstantVelocity(fw_cv_noise_covariance))),
                       # CombinedLinearGaussianTransitionModel((ConstantTurn(constant_turn_covariance, turn_rate_right),
                       #                                        ConstantVelocity(fw_cv_noise_covariance))),
                       # CombinedLinearGaussianTransitionModel((ConstantAcceleration(noise_covariance),
                       #                                        ConstantAcceleration(noise_covariance),
                       #                                        ConstantVelocity(noise_covariance))),
                       ]

dynamic_model_list = [*np.array(dynamic_model_list_RW), *np.array(dynamic_model_list_FW)]

detection_matrix_split = [len(dynamic_model_list_RW), len(dynamic_model_list_FW)]

model_mapping = [
                   # Rotary Wing
                   [0, 1, 3, 4, 6, 7],              # CV CV CV
                   [0, 1, 3, 4, 6, 7],              # H H CV
                   # [0, 1, 2, 3, 4, 5, 6, 7, 8],     # CT CA

                   # Fixed Wing
                   [0, 1, 3, 4, 6, 7],              # CV CV CV
                   [0, 1, 2, 3, 4, 5, 6, 7, 8],     # CA CA CA
                   # [0, 1, 3, 4, 6, 7, 8],           # CV CV CA
                   # [0, 1, 2, 3, 4, 6, 7],           # CA CV CV
                   # [0, 1, 3, 4, 5, 6, 7],           # CV CA CV
                   # 0, 1, 3, 4, 6, 7],              # CTL CV
                   # 0, 1, 3, 4, 6, 7],              # CTR CV
                   # [0, 1, 2, 3, 4, 5, 6, 7],        # CA CA CV
                  ]


transition = form_detection_transition_matrix(detection_matrix_split, [0.05, 0.05])

measurement_model = LinearGaussian(
    ndim_state=9,  # Number of state dimensions (position, velocity and acceleration in 3D)
    mapping=(0, 3, 6),  # Locations of our position variables within the entire state space
    noise_covar=np.diag([0.1, 0.1, 0.1]))

multi_model = MultiModelPredictor(transition, model_mapping, transition_model=dynamic_model_list)

resampler = SystematicResampler()
updater = MultiModelParticleUpdater(measurement_model=measurement_model,
                                    resampler=resampler,
                                    transition_matrix=transition,
                                    position_mapping=model_mapping,
                                    transition_model=dynamic_model_list
                                    )

x_0, v_x, a_x, y_0, v_y, a_y, z_0, v_z, a_z = create_prior(location)

samples = multivariate_normal.rvs(np.array([x_0, v_x, a_x, y_0, v_y, a_y, z_0, v_z, a_z]),
                                  np.diag([0, 0, 0, 0, 0, 0, 0, 0, 0]), size=NUMBER_OF_PARTICLES)

start_model = []
for i in range(NUMBER_OF_PARTICLES):
    random_int = random()
    if random_int < percentage_of_first_model:
        start_model.append(0)
    elif random_int >= percentage_of_first_model:
        start_model.append(detection_matrix_split[0])

particles = [Particle(sample.reshape(-1, 1), weight=Probability(1/NUMBER_OF_PARTICLES),
                      dynamic_model=[start_model[randint(0, NUMBER_OF_PARTICLES - 1)],
                                     start_model[randint(0, NUMBER_OF_PARTICLES - 1)]]) for sample in samples]

"""particles = [Particle(sample.reshape(-1, 1), weight=Probability(1/NUMBER_OF_PARTICLES),
                      dynamic_model=start_model[randint(0, NUMBER_OF_PARTICLES - 1)]) for sample in samples]"""

prior_state = ParticleState(particles, timestamp=start_time)

track = Track()
dynamic_model_split = []
effective_sample_size = []
weighted_sum_per_model = []
counter = 0
for iteration, measurement in enumerate(tqdm(measurements)):
    prediction, dynamic_model_proportions = multi_model.predict(prior_state, timestamp=measurement.timestamp,
                                                                multi_craft=True)
    dynamic_model_split.append(dynamic_model_proportions)
    weighted_sum_per_model.append([sum([p.weight for p in prediction.particles if p.dynamic_model == j])
                                   for j in range(len(transition))])
    hypothesis = SingleHypothesis(prediction, measurement)
    post, n_eff = updater.update(hypothesis)
    print(n_eff)
    # if n_eff < 10 and iteration > 30:
    #     counter += 1
    # if counter > 10:
    #     break
    effective_sample_size.append(n_eff)
    track.append(post)
    prior_state = track[-1]

particle_path = [[track[i].particles[j].state_vector for i in range(len(track))]
                 for j in range(len(track[0].particles))]

try:
    os.mkdir(f"{SAVE_DIR}/{file_list[DRONE_FILE]}")
except FileExistsError:
    print("Folder already exists")

number_of_models = len(dynamic_model_list)
ax = plt.axes(projection="3d")

ax.plot3D(np.array([state.particles[0].state_vector[0] for state in track]).flatten(),
          np.array([state.particles[0].state_vector[3] for state in track]).flatten(),
          np.array([state.particles[0].state_vector[6] for state in track]).flatten(), color='c', label='PF')

ax.plot3D(np.array([state.state_vector[0] for state in truth]).flatten(),
          np.array([state.state_vector[1] for state in truth]).flatten(),
          np.array([state.state_vector[2] for state in truth]).flatten(), linestyle="--", color='coral', label='Truth')

# x.scatter(np.array([[particle[i][0] for particle in particle_path]
#                     for i in range(len(track)) if i % 10 == 0]).flatten(),
#           np.array([[particle[i][3] for particle in particle_path]
#                     for i in range(len(track)) if i % 10 == 0]).flatten(),
#           np.array([[particle[i][6] for particle in particle_path]
#                     for i in range(len(track)) if i % 10 == 0]).flatten(),
#                      linestyle="--", color='blue', label='Particles')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(loc="upper right")
plt.savefig(f"{SAVE_DIR}/{file_list[DRONE_FILE]}/rw {rw_cv_noise_covariance} fw {fw_cv_noise_covariance}.png", dpi=2000)
plt.title("True path and predicted PF path")
plt.show()


plt.plot(range(len(effective_sample_size)), effective_sample_size)
plt.xlabel("Timestep")
plt.ylabel("Effective Sample Size")
plt.title("Effective sample size at a given timestep")
plt.savefig(f"{SAVE_DIR}/{file_list[DRONE_FILE]}/rw {rw_cv_noise_covariance} fw {fw_cv_noise_covariance}.png", dpi=2000)
plt.show()


# dynamic_model_split = dynamic_model_split[100:150]
dynamic_model_plot = [[element[j] for element in dynamic_model_split] for j in range(len(dynamic_model_split[0]))]

for i, line in enumerate(dynamic_model_plot):
    plt.plot(range(len(dynamic_model_split)), line)
plt.legend([f"Model {i}" for i in range(len(dynamic_model_plot))])
plt.title("Number of Particles for each model")
plt.xlabel('Timestep')
plt.ylabel('Number of Particles')
plt.savefig(f"{SAVE_DIR}/{file_list[DRONE_FILE]}/rw {rw_cv_noise_covariance} fw {fw_cv_noise_covariance}.png", dpi=2000)
plt.show()


probability_of_each_craft = []
for i, element in enumerate(range(len(detection_matrix_split))):
    craft_sum = np.cumsum(detection_matrix_split)
    temp = []
    for entry in weighted_sum_per_model:
        if i == 0:
            temp.append(
                sum([entry[k] for k in range(detection_matrix_split[i])])
            )
        else:
            temp.append(
                sum([entry[k] for k in range(craft_sum[i - 1], craft_sum[i])])
            )
    probability_of_each_craft.append([*temp])

sum_of_propbs = sum([sum(probability_of_each_craft[i]) for i in range(len(probability_of_each_craft))])

for i, line in enumerate(probability_of_each_craft):
    plt.plot(range(len(probability_of_each_craft[0])), line)
plt.legend([f"Model {i}" for i in range(len(probability_of_each_craft[0]))])
plt.title("Probability of each craft")
plt.xlabel('Timestep')
plt.ylabel('Probability')
plt.savefig(f"{SAVE_DIR}/{file_list[DRONE_FILE]}/rw {rw_cv_noise_covariance} fw {fw_cv_noise_covariance}.png", dpi=2000)
plt.show()

print(f"Probability of Rotary Wing is : {sum(probability_of_each_craft[0]) / sum_of_propbs}")
print(f"Probability of Fixed Wing is : {sum(probability_of_each_craft[1]) / sum_of_propbs}")
print(f"Model Actually is : {model_type}")


difference = [np.linalg.norm(track[i].state_vector[[0, 3, 6]] - truth[i].state_vector) for i in range(len(track))]
sum_of_difference = sum(difference)
print(difference)
print(sum_of_difference)

plt.plot(range(len(difference)), difference)
plt.xlabel('Timestep')
plt.ylabel('Difference')
plt.title("Distance Metric")
plt.text(100, 10, f"Total distance : {sum_of_difference}")
plt.savefig(f"{SAVE_DIR}/{file_list[DRONE_FILE]}/rw {rw_cv_noise_covariance} fw {fw_cv_noise_covariance}.png", dpi=2000)
plt.show()
