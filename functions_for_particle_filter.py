import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import sample
import csv
import pandas as pd
import scipy as sp


def create_geo_reference_point(lat, long):
    circum_polar = 40007863    # wikipedia
    circum_equator = 40075017
    degrees_to_metres_ns = circum_polar / 360.0
    degrees_to_metres_ew = circum_equator * np.cos(np.radians(lat)) / 360.0
    return [lat, long, degrees_to_metres_ns, degrees_to_metres_ew]


def convert_to_local_cartesian(lat, long, geo_ref_point):
    y = (lat - geo_ref_point[0]) * geo_ref_point[2]
    x = (long - geo_ref_point[1]) * geo_ref_point[3]
    return x, y


def read_csv_file(file_name):

    header_line_count = 5
    csv_file = open(file_name, 'r')
    csv_lines = list(csv.reader(csv_file, delimiter=','))
    csv_lines = np.array(csv_lines)
    # t lat long z
    track_data = np.array(csv_lines[header_line_count:, 1:5], dtype=np.float)

    # create array for the xy data for the track
    track_xy = np.zeros((track_data.shape[0], 2))

    # create a latlong reference point from the first point in the track
    ref_point = create_geo_reference_point(track_data[0, 1], track_data[0, 2])

    # populate the array of xy locations
    for i in range(0, track_data.shape[0]):
        x, y = convert_to_local_cartesian(track_data[i, 1], track_data[i, 2], ref_point)
        track_xy[i, 0] = x
        track_xy[i, 1] = y

    # merge it into the main array
    track_data_with_xy = np.hstack((track_data, track_xy))

    return track_data_with_xy


def plot_cartesian_data(location):
    # fig = plt.figure()
    ax = plt.axes(projection="3d")

    x_points = np.array(location)[:, 0:1]
    y_points = np.array(location)[:, 1:2]
    z_points = np.array(location)[:, 2:3]

    ax.plot3D(x_points.flatten(), y_points.flatten(), z_points.flatten())
    plt.show()


def clean_data(path):

    """Will remove rows that exhibit a large gap in between readings"""

    def clean_via_row(position):
        initial_norm = np.linalg.norm(position[0][:-1])
        cleaned_path = []
        differences = []
        for j, row in enumerate(position[1:]):
            row_norm = np.linalg.norm(row[:-1])
            if abs(row_norm - initial_norm) > 7 or abs(row_norm - initial_norm) < 0.01:
                differences.append(abs(row_norm - initial_norm))
                # print(f"Found discontinuity at row {j} position {row}")
            elif abs(row_norm - initial_norm) == 0.0 and differences[-1] == 0.0:
                differences.append(abs(row_norm - initial_norm))
                print(f"Found stationary target at row {j} position {row}")
            else:
                cleaned_path.append(row)
                differences.append(abs(row_norm - initial_norm))
            initial_norm = row_norm
        # cleaned_path = np.delete(cleaned_path, discontinuities, 0)
        # print(sorted(differences))
        return cleaned_path

    return clean_via_row(path)


def reduce_resolution(input_matrix, data_reduction):

    """This will randomly pick out rows within our track to create an artificial time and position jump"""

    random_indices = sample(range(len(input_matrix)), int(np.floor(data_reduction * len(input_matrix))))
    random_indices.sort()
    input_matrix = input_matrix[random_indices]
    return input_matrix


def import_track_data(file_number, data_reduction, data_dir):
    
    file_list = os.listdir(data_dir)
    print(file_list[file_number])
    # t lat long z x y
    data = read_csv_file(os.path.join(data_dir, file_list[file_number]))
    x_column_index = 4
    y_column_index = 5
    z_column_index = 3
    t_column_index = 0

    x = data[:len(data) - 1, x_column_index]
    y = data[:len(data) - 1, y_column_index]
    z = data[:len(data) - 1, z_column_index]
    t = data[:len(data) - 1, t_column_index]
    # x y z t
    location = np.vstack((x, y, z, t))
    # (many, 4)
    location = np.transpose(location)
    location = np.array(clean_data(location))
    location = reduce_resolution(location, data_reduction)
    return location


def create_prior(location):

    x_0, x_1, x_2 = location[0][0], location[2][0], location[4][0]
    y_0, y_1, y_2 = location[0][1], location[2][1], location[4][1]
    z_0, z_1, z_2 = location[0][2], location[2][2], location[4][2]
    delta_t = location[2][3] - location[0][3]

    v_x_0, v_x_1 = (x_1 - x_0) / delta_t, (x_2 - x_1) / delta_t
    v_y_0, v_y_1 = (y_1 - y_0) / delta_t, (y_2 - y_1) / delta_t
    v_z_0, v_z_1 = (z_1 - z_0) / delta_t, (z_2 - z_1) / delta_t

    a_x = (v_x_1 - v_x_0) / delta_t
    a_y = (v_y_1 - v_y_0) / delta_t
    a_z = (v_z_1 - v_z_0) / delta_t

    return x_0, v_x_0, a_x, y_0, v_y_0, a_y, z_0, v_z_0, a_z


def read_synthetic_csv(file_name):

    df = pd.read_csv(file_name)
    location = np.array(df, dtype=float)

    return location


def form_transition_matrix(dynamic_model_list, probability_for_change):
    transition = np.zeros((len(dynamic_model_list), len(dynamic_model_list)))
    for i in range(len(dynamic_model_list)):
        for j in range(len(dynamic_model_list)):
            if j == i:
                transition[i][i] = 1 - ((len(transition) - 1) * probability_for_change)
            else:
                transition[i][j] = probability_for_change
    return transition


def form_detection_transition_matrix(matrix_split, probability_for_change):

    model_matrix_1 = form_transition_matrix([i for i in range(matrix_split[0])], probability_for_change[0])
    model_matrix_2 = form_transition_matrix([i for i in range(matrix_split[1])], probability_for_change[1])

    return sp.linalg.block_diag(np.array(model_matrix_1), np.array(model_matrix_2))


class PlotData:

    def __init__(self, truth, track, effective_sample_size, dynamic_model_split, model_probabilities,
                 weighted_sum_per_model, detection_matrix_split):

        self.truth = truth
        self.track = track
        self.effective_sample_size = effective_sample_size
        self.dynamic_model_split = dynamic_model_split
        self.model_probabilities = model_probabilities
        self.weighted_sum_per_model = weighted_sum_per_model
        self.detection_matrix_split = detection_matrix_split

        self.dynamic_model_plot = [[element[j] for element in dynamic_model_split] for j in
                                   range(len(dynamic_model_split[0]))]
        self.model_probabilities_plot = [[element[j] for element in model_probabilities] for j in
                                         range(len(model_probabilities[0]))]
        self.sum_of_probs = sum([sum(self.craft_prob()[i]) for i in range(len(self.craft_prob()))])

    def calculate_sum_weighted_positions(self):

        weighted_positions = []
        for element in self.track:
            particle_container = []
            for particle in element.particles:
                particle_container.append(particle.state_vector * particle.weight)
            weighted_positions.append(particle_container)

        sum_weighted_positions = [sum(element) for element in weighted_positions]

        return sum_weighted_positions

    def pf_vs_truth(self):

        weighted_state = self.calculate_sum_weighted_positions()

        ax = plt.axes(projection="3d")

        ax.plot3D(np.array([state[0] for state in weighted_state]).flatten(),
                  np.array([state[3] for state in weighted_state]).flatten(),
                  np.array([state[6] for state in weighted_state]).flatten(), color='coral',
                  label='Weighted PF')

        ax.plot3D(np.array([state.state_vector[0] for state in self.truth]).flatten(),
                  np.array([state.state_vector[1] for state in self.truth]).flatten(),
                  np.array([state.state_vector[2] for state in self.truth]).flatten(), linestyle="--", color='c',
                  label='Truth')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend(loc="upper right")
        plt.title("True path and predicted PF path")
        plt.show()

    def plot_neff(self):

        plt.plot(range(len(self.effective_sample_size)), self.effective_sample_size)
        plt.xlabel("Timestep")
        plt.ylabel("Effective Sample Size")
        plt.title("Effective sample size at a given timestep")
        plt.show()

    def particles_per_model(self):

        for i, line in enumerate(self.dynamic_model_plot):
            plt.plot(range(len(self.dynamic_model_split)), line)
        plt.legend([f"Model {i}" for i in range(len(self.dynamic_model_plot))])
        plt.title("Number of Particles for each model")
        plt.xlabel('Timestep')
        plt.ylabel('Number of Particles')
        plt.show()

    def rao_probabilities(self):

        for i, line in enumerate(self.model_probabilities_plot):
            plt.plot(range(len(self.dynamic_model_split)), line)
        plt.legend([f"Model {i}" for i in range(len(self.model_probabilities_plot))])
        plt.title("Average Model Probability given previous track")
        plt.xlabel('Timestep')
        plt.ylabel('Probability')
        plt.show()

    def craft_prob(self):

        probability_of_each_craft = []
        for i in range(len(self.detection_matrix_split)):
            craft_sum = np.cumsum(self.detection_matrix_split)
            temp = []
            for entry in self.weighted_sum_per_model:
                if i == 0:
                    temp.append(
                        sum([entry[k] for k in range(self.detection_matrix_split[i])])
                    )
                else:
                    temp.append(
                        sum([entry[k] for k in range(craft_sum[i - 1], craft_sum[i])])
                    )
            probability_of_each_craft.append([*temp])

        return probability_of_each_craft

    def craft_prob_plot(self):

        for i, line in enumerate(self.craft_prob()):
            plt.plot(range(len(self.craft_prob()[0])), line)
        plt.legend([f"Model {i}" for i in range(len(self.craft_prob()[0]))])
        plt.title("Probability of each craft")
        plt.xlabel('Timestep')
        plt.ylabel('Probability')
        plt.show()

        print(f"Probability of Rotary Wing is : {sum(self.craft_prob()[0]) / self.sum_of_probs}")
        print(f"Probability of Fixed Wing is : {sum(self.craft_prob()[1]) / self.sum_of_probs}")

    def plot_difference_metric(self):

        weighted_state = self.calculate_sum_weighted_positions()

        difference = [np.linalg.norm(weighted_state[i][[0, 3, 6]] - self.truth[i].state_vector) for i in
                      range(len(self.track))]

        sum_of_difference = sum(difference)

        plt.plot(range(len(difference)), difference)
        plt.xlabel('Timestep')
        plt.ylabel('Difference')
        plt.title("Distance Metric")
        plt.show()
        print(f"Total distance : {sum_of_difference}")
