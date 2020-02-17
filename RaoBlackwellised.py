from functions_for_particle_filter import import_track_data, create_prior, read_synthetic_csv, \
    form_transition_matrix, form_detection_transition_matrix, PlotData
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity, ConstantAcceleration, ConstantTurn, ConstantPosition, LinearTurn
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.predictor.multi_model import MultiModelPredictor, RaoBlackwellisedMultiModelPredictor
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.resampler.particle import SystematicResampler, RaoBlackwellisedSystematicResampler
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.updater.particle import ParticleUpdater, MultiModelParticleUpdater, RaoBlackwellisedParticleUpdater
from stonesoup.types.numeric import Probability
from stonesoup.types.state import ParticleState
from stonesoup.types.detection import Detection
from stonesoup.types.particle import Particle, RaoBlackwellisedParticle
from scipy.stats import multivariate_normal
from stonesoup.types.track import Track
from datetime import timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from operator import add
from random import random, seed
from tqdm import tqdm
import numpy as np
import os

seed(100)
DRONE_FILE = 15
DATA_DIR = "P:/DASA/EDITTS Drone Tracking/GFI/GPS Tracking"
# DATA_DIR = "C:/Work/Drone_Tracking/EDITTS-Drone-Tracking/data/raw/"
SAVE_DIR = "C:/Work/Drone_Tracking/multi_model_results"
FIXED_WING = {"g2", "g4", "maja", "bixler", "x8", "kahu"}
ROTARY_WING = {"g6", "f550", "drdc"}

NUMBER_OF_PARTICLES = 300
rw_cv_noise_covariance = 2.5
fw_cv_noise_covariance = 0.03
rw_hover_noise_covariance = 0.01
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
location = location[1250:1500]

ax.plot3D(location[:, 0],
          location[:, 1],
          location[:, 2])
# plt.show()

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
                                                               ConstantVelocity(rw_cv_noise_covariance)))
                        ]

dynamic_model_list_FW = [
                       # Fixed Wing
                       CombinedLinearGaussianTransitionModel((ConstantVelocity(fw_cv_noise_covariance),
                                                              ConstantVelocity(fw_cv_noise_covariance),
                                                              ConstantVelocity(fw_cv_noise_covariance))),
                       CombinedLinearGaussianTransitionModel((ConstantAcceleration(fw_cv_noise_covariance),
                                                              ConstantAcceleration(fw_cv_noise_covariance),
                                                              ConstantAcceleration(fw_cv_noise_covariance)))
                       ]

dynamic_model_list = [*np.array(dynamic_model_list_RW), *np.array(dynamic_model_list_FW)]

detection_matrix_split = [len(dynamic_model_list_RW), len(dynamic_model_list_FW)]

model_mapping = [
                   # Rotary Wing
                   [0, 1, 3, 4, 6, 7],              # CV CV CV
                   [0, 1, 3, 4, 6, 7],              # H H CV

                   # Fixed Wing
                   [0, 1, 3, 4, 6, 7],              # CV CV CV
                   [0, 1, 2, 3, 4, 5, 6, 7, 8],     # CA CA CA
                  ]


# transition = form_detection_transition_matrix(detection_matrix_split, [0.05, 0.05])
transition = [[0.93, 0.05, 0.01, 0.01], [0.03, 0.95, 0.01, 0.01], [0.01, 0.01, 0.93, 0.5], [0.01, 0.01, 0.05, 0.93]]
measurement_model = LinearGaussian(
    ndim_state=9,  # Number of state dimensions (position, velocity and acceleration in 3D)
    mapping=(0, 3, 6),  # Locations of our position variables within the entire state space
    noise_covar=np.diag([1, 1, 1]))

multi_model = MultiModelPredictor(transition, model_mapping, transition_model=dynamic_model_list)
rao_multi_model = RaoBlackwellisedMultiModelPredictor(transition, model_mapping, transition_model=dynamic_model_list)

resampler = RaoBlackwellisedSystematicResampler()
updater = MultiModelParticleUpdater(measurement_model=measurement_model,
                                    resampler=resampler,
                                    transition_matrix=transition,
                                    position_mapping=model_mapping,
                                    transition_model=dynamic_model_list
                                    )

rao_updater = RaoBlackwellisedParticleUpdater(measurement_model=measurement_model,
                                              resampler=resampler
                                              )

x_0, v_x, a_x, y_0, v_y, a_y, z_0, v_z, a_z = create_prior(location)

samples = multivariate_normal.rvs(np.array([x_0, v_x, a_x, y_0, v_y, a_y, z_0, v_z, a_z]),
                                  np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), size=NUMBER_OF_PARTICLES)


def choose_model():
    rand = random()
    if rand < 0.5:
        return 0
    else:
        return detection_matrix_split[0]


particles = [RaoBlackwellisedParticle(sample.reshape(-1, 1), weight=Probability(1/NUMBER_OF_PARTICLES),
             dynamic_model=choose_model(), time_interval=start_time, model_probabilities=[0.5, 0, 0.5, 0])
             for sample in samples]

prior_state = ParticleState(particles, timestamp=start_time)

track = Track()
craft_probs = []
dynamic_model_split = []
model_probabilities = []
effective_sample_size = []
weighted_sum_per_model = []
probability_of_each_craft = []
for iteration, measurement in enumerate(tqdm(measurements)):

    prediction = rao_multi_model.predict(prior_state, timestamp=measurement.timestamp, multi_craft=True)

    weighted_sum_per_model.append([sum([p.weight for p in prediction.particles if p.dynamic_model == j])
                                   for j in range(len(transition))])

    particle_proportions = [p.dynamic_model for p in prediction.particles]
    print([particle_proportions.count(i) for i in range(len(transition))])

    model_probabilities.append([sum([p.model_probabilities[i] for p in prediction.particles]) / NUMBER_OF_PARTICLES
                                for i in range(len(transition))])
    print([sum([p.model_probabilities[i] for p in prediction.particles]) / NUMBER_OF_PARTICLES
           for i in range(len(transition))])
    dynamic_model_split.append([particle_proportions.count(i) for i in range(len(transition))])

    craft_sum = np.cumsum(detection_matrix_split)
    rw_prob = sum([weighted_sum_per_model[-1][i] for i in range(craft_sum[0])])
    fw_prob = sum([weighted_sum_per_model[-1][i] for i in range(craft_sum[0], craft_sum[1])])
    craft_probs.append([rw_prob, fw_prob])

    if iteration % 10 == 0 and iteration != 0:

        cumulative_rw_prob = sum([prob[0] for prob in craft_probs])
        cumulative_fw_prob = sum([prob[1] for prob in craft_probs])

        sum_of_probs = cumulative_fw_prob + cumulative_rw_prob

        print(f"Probability of Rotary Wing is : {cumulative_rw_prob / sum_of_probs}")
        print(f"Probability of Fixed Wing is : {cumulative_fw_prob / sum_of_probs}")

    hypothesis = SingleHypothesis(prediction, measurement)
    post, n_eff = rao_updater.update(hypothesis, iteration=iteration, predictor=rao_multi_model)

    effective_sample_size.append(n_eff)
    track.append(post)
    prior_state = track[-1]

data_plot = PlotData(truth, track, effective_sample_size, dynamic_model_split, model_probabilities,
                     weighted_sum_per_model, detection_matrix_split)

data_plot.pf_vs_truth()
data_plot.plot_neff()
data_plot.particles_per_model()
data_plot.rao_probabilities()
data_plot.craft_prob_plot()

print(f"Model Actually is : {model_type}")
