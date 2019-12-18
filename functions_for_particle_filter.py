from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.metricgenerator.ospametric import GOSPAMetric
from datetime import timedelta
from stonesoup.models.measurement.linear import LinearGaussian
from datetime import datetime
import numpy as np
import cupy as cp
import pandas as pd
import os
from stonesoup.types.state import State
from stonesoup.types.track import Track
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from random import sample
import csv

DATA_DIR = "C:/Users/i_jenkins/Documents/Python Scripts"
file_list = os.listdir(DATA_DIR)
NOISE_COVARIANCE = 0.1


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


def data_preprocessing_truth(location):
    path = GroundTruthPath()
    start_time = datetime.now()
    for t, element in enumerate(location):
        position = np.array([element[0], element[1], element[2]])
        position = position.reshape(3, 1)
        path.append(GroundTruthState(state_vector=position, timestamp=start_time+timedelta(seconds=t)))
    return path, start_time


def plot_cartesian_data(location):
    # fig = plt.figure()
    ax = plt.axes(projection="3d")

    x_points = np.array(location)[:, 0:1]
    y_points = np.array(location)[:, 1:2]
    z_points = np.array(location)[:, 2:3]

    ax.plot3D(x_points.flatten(), y_points.flatten(), z_points.flatten())
    plt.show()


def heading_direction(heading_radians, test_array):
    # Converting degrees to radians
    for i in range(len(data[:, 4:5])):
        heading_radians.append([float(data[i][4]) * (np.pi / 180)])
    plt.polar(heading_radians, test_array)
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


def create_truth_data(file_number, data_reduction):

    print(file_list[file_number])
    # t lat long z x y
    data = read_csv_file(os.path.join(DATA_DIR, file_list[file_number]))
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
    truth, start_time = data_preprocessing_truth(location)
    return truth, start_time, location


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


def compute_metric(track_path, truth_path, drone_dir, drone_file, title_parse, number_of_particles,
                   data_reduction, cutoff_distance, p):

    measurement_model_track = LinearGaussian(
        6,  # Number of state dimensions (position and velocity in 3D)
        (0, 2, 4),  # Mapping measurement dimensions to state dimensions
        np.diag([0.1, 0.1, 0.1]))  # Covariance matrix for Gaussian PDF
    measurement_model_truth = LinearGaussian(
        3,  # Number of state dimensions (position in 3D)
        (0, 1, 2),  # Mapping measurement dimensions to state dimensions
        np.diag([0.1, 0.1, 0.1]))  # Covariance matrix for Gaussian PDF

    gospa_generator = GOSPAMetric(c=cutoff_distance, p=p,
                                  measurement_model_truth=measurement_model_truth,
                                  measurement_model_track=measurement_model_track)

    # gospa_cost_matrix = gospa_generator.compute_cost_matrix(tracks.states, truths.states)
    # print(f"This is the gospa cost matrix: {gospa_cost_matrix[0:3]}")

    gospa_metric = gospa_generator.compute_over_time(track_path.states, truth_path.states)

    # gospa_timestamps = [gospa_metric.value[i].timestamp.timestamp() for i in range(len(gospa_metric.value))]
    # gospa_timestamps = [gospa_timestamps[i] - min(gospa_timestamps) / max(gospa_timestamps)
    #                     for i in range(len(gospa_timestamps))]

    gospa_distances = [gospa_metric.value[k].value['distance'] for k in range(len(gospa_metric.value))]
    try:
        gospa_distances = gospa_distances[:gospa_distances.index(cutoff_distance) + 1]
    except ValueError:
        print(f'{cutoff_distance} is not in the list of gospa distances')
    # gospa_timestamps = gospa_timestamps[:gospa_distances.index(cutoff_distance) + 2]

    print(f"This is the gospa metric: \n {gospa_distances}")

    plt.plot(range(0, len(gospa_distances)), gospa_distances)
    plt.xlabel("")
#     plt.savefig(f"{drone_dir}/{file_list[drone_file]}/{title_parse[3]} Drone with {number_of_particles}"
#                 f" particles with {data_reduction * 100}% Noise Covariance {NOISE_COVARIANCE} metric.png", dpi=2000)
    plt.show()

    # assignment_matrix = gospa_generator.compute_gospa_metric(tracks.states, truths.states)[1][0]
    # print(f"This is the assignment matrix: {sum(assignment_matrix)}")
