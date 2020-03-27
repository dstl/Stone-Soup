from functions_for_particle_filter import import_track_data, create_prior, read_synthetic_csv, \
    form_transition_matrix, form_detection_transition_matrix, PlotData, read_bird_csv
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity, ConstantAcceleration, ConstantTurn, ConstantPosition, LinearTurn
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.predictor.particle import MultiModelPredictor, RaoBlackwellisedMultiModelPredictor
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.resampler.particle import SystematicResampler, RaoBlackwellisedSystematicResampler, MultiResampler
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.updater.particle import ParticleUpdater, MultiModelParticleUpdater, RaoBlackwellisedParticleUpdater
from stonesoup.types.numeric import Probability
from stonesoup.types.state import ParticleState
from stonesoup.types.detection import Detection
from stonesoup.types.particle import RaoBlackwellisedParticle
from scipy.stats import multivariate_normal
from stonesoup.types.track import Track
from datetime import timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from random import random, seed
from tqdm import tqdm
import time
import numpy as np
import os

seed(100)

DRONE_FILE = np.random.randint(0, 15)
DIR = "C:/Work/editts_working/training_data/track_data/"
classes = os.listdir(DIR)
current_class = np.random.choice(classes)
print(current_class)
DATA_DIR = os.path.join(DIR, current_class)

# DATA_DIR = "C:/Work/Drone_Tracking/EDITTS-Drone-Tracking/scripts/BirdDynamicsSynthesiser/synthesisedtracks/"
# DATA_DIR = "C:/Work/Drone_Tracking/EDITTS-Drone-Tracking/data/raw/"
SAVE_FILE = f"C:/Work/pf_results/{current_class}.csv"

DATA_REDUCTION = 1  # (0, 1]

location = import_track_data(DRONE_FILE, DATA_REDUCTION, DATA_DIR)
# location = read_synthetic_csv(DATA_DIR + file_list[DRONE_FILE])
# location = read_bird_csv(DATA_DIR + file_list[DRONE_FILE])

for i, element in enumerate(location):
    location[i][:3] = np.random.normal(element[:3], 0)

ax = plt.axes(projection="3d")
ax.plot3D(location[:, 0],
          location[:, 1],
          location[:, 2])

# location = location[int(len(location) * 0): int(len(location) * 0.05)]
track_position = np.random.randint(0, len(location) - 50)
location = location[track_position: track_position + 50]

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

NUMBER_OF_PARTICLES = np.random.randint(2, 350)

bif_ca_nc = 0.5
bif_cp_nc = 0.5

fw_cv_nc = 0.2
fw_ca_nc = 0.2
fw_cp_nc = 0.1

rw_cv_nc = 0.05
rw_ca_nc = 0.5
rw_cp_nc = 0.00001

constant_turn_covariance = [0.1, 0.1]
turn_rate_left = 0.5
turn_rate_right = -0.5

dynamic_model_list_BIF = [CombinedLinearGaussianTransitionModel((ConstantAcceleration(bif_ca_nc),
                                                                 ConstantAcceleration(bif_ca_nc),
                                                                 ConstantPosition(bif_cp_nc))),

                          CombinedLinearGaussianTransitionModel((ConstantPosition(bif_cp_nc),
                                                                 ConstantPosition(bif_cp_nc),
                                                                 ConstantAcceleration(bif_ca_nc))),

                          # CombinedLinearGaussianTransitionModel((ConstantTurn(constant_turn_covariance,
                          #                                                     turn_rate_left),
                          #                                        ConstantPosition(fw_cv_nc))),

                          # CombinedLinearGaussianTransitionModel((ConstantTurn(constant_turn_covariance,
                          #                                                     turn_rate_left),
                          #                                        ConstantPosition(fw_cv_nc))),

                          # CombinedLinearGaussianTransitionModel((ConstantAcceleration(fw_cv_nc),
                          #                                        ConstantAcceleration(fw_cv_nc),
                          #                                        ConstantAcceleration(fw_cv_nc)))
                          ]

dynamic_model_list_FW = [
                       # Fixed Wing
                       CombinedLinearGaussianTransitionModel((ConstantVelocity(fw_cv_nc),
                                                              ConstantVelocity(fw_cv_nc),
                                                              ConstantPosition(fw_cp_nc))),

                       CombinedLinearGaussianTransitionModel((ConstantVelocity(fw_cv_nc),
                                                              ConstantVelocity(fw_cv_nc),
                                                              ConstantVelocity(fw_cv_nc))),

                       CombinedLinearGaussianTransitionModel((ConstantTurn(constant_turn_covariance,
                                                              turn_rate_left),
                                                              ConstantPosition(fw_cv_nc))),
                       ]

dynamic_model_list_RW = [
                        # Rotary Wing
                        CombinedLinearGaussianTransitionModel((ConstantVelocity(rw_cv_nc),
                                                               ConstantVelocity(rw_cv_nc),
                                                               ConstantPosition(rw_cv_nc))),

                        CombinedLinearGaussianTransitionModel((ConstantPosition(rw_cp_nc),
                                                               ConstantPosition(rw_cp_nc),
                                                               ConstantAcceleration(rw_ca_nc))),

                        CombinedLinearGaussianTransitionModel((ConstantPosition(rw_cp_nc),
                                                               ConstantPosition(rw_cp_nc),
                                                               ConstantVelocity(rw_cv_nc)))
                        ]

dynamic_model_list = [*np.array(dynamic_model_list_BIF),
                      *np.array(dynamic_model_list_FW),
                      *np.array(dynamic_model_list_RW)]

detection_matrix_split = [len(dynamic_model_list_BIF),
                          len(dynamic_model_list_FW),
                          len(dynamic_model_list_RW)]

model_mapping = [
                 # Birds In Flight
                 [0, 1, 2, 3, 4, 5, 6, 7],
                 [0, 1, 3, 4, 6, 7, 8],
                 # [0, 1, 3, 4, 6, 7],
                 # [0, 1, 2, 3, 4, 5, 6, 7, 8],

                 # Fixed Wing
                 [0, 1, 3, 4, 6, 7],  # CV CV CV
                 [0, 1, 3, 4, 6, 7],  # CA CA CA
                 [0, 1, 3, 4, 6, 7],

                 # Rotary Wing
                 [0, 1, 3, 4, 6, 7],              # CV CV CV
                 [0, 1, 3, 4, 6, 7, 8],              # H H CA
                 [0, 1, 3, 4, 6, 7],              # H H CA
                 ]

# transition = form_detection_transition_matrix(detection_matrix_split, [0.05, 0.05, 0.05])
transition = form_transition_matrix(dynamic_model_list, 0.001)
measurement_model = LinearGaussian(
    ndim_state=9,  # Number of state dimensions (position, velocity and acceleration in 3D)
    mapping=(0, 3, 6),  # Locations of our position variables within the entire state space
    noise_covar=np.diag([0.75, 0.75, 0.75]))

rao_multi_model = RaoBlackwellisedMultiModelPredictor(position_mapping=model_mapping,
                                                      transition_model=dynamic_model_list)

resampler = RaoBlackwellisedSystematicResampler()
updater = MultiModelParticleUpdater(measurement_model=measurement_model,
                                    resampler=resampler
                                    )

rao_updater = RaoBlackwellisedParticleUpdater(measurement_model=measurement_model,
                                              resampler=resampler
                                              )

x_0, v_x, a_x, y_0, v_y, a_y, z_0, v_z, a_z = create_prior(location)

samples = multivariate_normal.rvs(np.array([x_0, v_x, a_x, y_0, v_y, a_y, z_0, v_z, a_z]),
                                  np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                                  size=NUMBER_OF_PARTICLES)
if random() > 1.1:
    sample_index = len(transition)
else:
    sample_index = None


def choose_model():
    rand = random()
    if rand < 0.5:
        return 0
    else:
        return detection_matrix_split[0]


particles = [RaoBlackwellisedParticle(
    sample.reshape(-1, 1), weight=Probability(1/NUMBER_OF_PARTICLES),
    model_probabilities=[1/len(transition) for i in transition])
    for sample in samples]

prior_state = ParticleState(particles, timestamp=start_time)

time.sleep(1)

track = Track()
craft_probs = []
dynamic_model_split = []
model_probabilities = []
effective_sample_size = []
weighted_sum_per_model = []
probability_of_each_craft = []
for iteration, measurement in enumerate(tqdm(measurements)):

    prediction = rao_multi_model.predict(prior_state,
                                         sample_index,
                                         timestamp=measurement.timestamp)

    # weighted_sum_per_model.append([sum([p.weight for p in prediction.particles if p.dynamic_model == j])
    #                                for j in range(len(transition_block))])

    # particle_proportions = [p.dynamic_model for p in prediction.particles]

    # print([particle_proportions.count(i) for i in range(len(transition))])
    model_probabilities.append([sum([(p.model_probabilities[i] * p.weight)
                                     for p in prediction.particles])
                                for i in range(len(transition))])

    # dynamic_model_split.append([particle_proportions.count(i) for i in range(len(transition_block))])

    # craft_sum = np.cumsum(detection_matrix_split)
    # rw_prob = sum([weighted_sum_per_model[-1][i] for i in range(craft_sum[0])])
    # fw_prob = sum([weighted_sum_per_model[-1][i] for i in range(craft_sum[0], craft_sum[1])])
    # craft_probs.append([rw_prob, fw_prob])
#
    # if iteration % 10 == 0 and iteration != 0:
#
    #     cumulative_rw_prob = sum([prob[0] for prob in craft_probs])
    #     cumulative_fw_prob = sum([prob[1] for prob in craft_probs])
#
    #     sum_of_probs = cumulative_fw_prob + cumulative_rw_prob
#
    #     print("\n")
    #     print(f"Probability of Rotary Wing is : {cumulative_rw_prob / sum_of_probs}")
    #     print(f"Probability of Fixed Wing is : {cumulative_fw_prob / sum_of_probs}")

    hypothesis = SingleHypothesis(prediction, measurement)
    post = rao_updater.update(hypothesis, predictor=rao_multi_model, transition=transition,
                              prior_timestamp=prior_state.timestamp,
                              sampling_distribution_len=len(transition))

    track.append(post)
    prior_state = track[-1]

data_plot = PlotData(truth, track, effective_sample_size, dynamic_model_split, model_probabilities,
                     weighted_sum_per_model, detection_matrix_split)

# data_plot.pf_vs_truth()
# data_plot.plot_neff()
# data_plot.particles_per_model()
probabilities = data_plot.rao_probabilities()
# data_plot.craft_prob_plot()
metric = data_plot.plot_difference_metric()

predicted_index = np.argmax(probabilities)

row = []
row.append(
    os.listdir(os.path.join(DIR, current_class))[DRONE_FILE]
)
row.append(
    [track_position, track_position + 30]
)
if sample_index is None:
    row.append("Rao-Blackwellised")
else:
    row.append("Uniform")
row.append(NUMBER_OF_PARTICLES)
row.append(metric)
row.append(bif_ca_nc)
row.append(bif_cp_nc)
row.append(fw_cv_nc)
row.append(fw_ca_nc)
row.append(fw_cp_nc)
row.append(rw_cv_nc)
row.append(rw_ca_nc)
row.append(rw_cp_nc)
row.append(probabilities[0])
row.append(probabilities[1])
row.append(probabilities[2])
row.append(
    predicted_index == classes.index(current_class)
)
print(row[-5:])

"""import csv

with open(rf"{SAVE_FILE}", "a") as f:
    writer = csv.writer(f)
    writer.writerow(row)"""
