"""
Created on Wed Aug 23 18:25:56 2023
@author: 007
"""

import random
import numpy as np


class GP_Sensor():
    def distance(self, point1, point2):
        x_diff = point1[0] - point2[0]
        y_diff = point1[1] - point2[1]
        return np.sqrt(x_diff**2 + y_diff**2)

    def create_sensor_network_plot(
        self, num_sensors, range_value, min_distance, xrange, yrange, seed
    ):
        random.seed(seed)
        sensor_data = []

        for _ in range(num_sensors):
            while True:
                x_position = random.uniform(xrange[0], xrange[1])
                y_position = random.uniform(yrange[0], yrange[1])
                valid_position = True

                for existing_sensor in sensor_data:
                    if self.distance(
                        (x_position, y_position),
                        existing_sensor['position']
                    ) < min_distance:
                        valid_position = False
                        break

                if valid_position:
                    break

            sensor_data.append(
                {'position': (x_position, y_position), 'range': range_value}
            )
        return sensor_data

    def track_target(self, x_positions, y_positions, sensor_data):
        num_sensors = len(sensor_data)
        num_steps = len(x_positions)

        x_matrix = np.zeros((num_sensors, num_steps))
        y_matrix = np.zeros((num_sensors, num_steps))

        for t in range(num_steps):
            target_x = x_positions[t]
            target_y = y_positions[t]

            for sensor_id, sensor_info in enumerate(sensor_data):
                sensor_position = sensor_info['position']
                sensor_range = sensor_info['range']

                if self.distance(
                    (target_x, target_y), sensor_position
                ) <= sensor_range:
                    x_matrix[sensor_id, t] = target_x
                    y_matrix[sensor_id, t] = target_y

        return x_matrix, y_matrix

    def track_targetDGP(self, x_positions, y_positions, sensor_data):
        num_sensors = len(sensor_data)
        num_steps = len(x_positions)

        time_data = [[] for _ in range(num_sensors)]
        x_data = [[] for _ in range(num_sensors)]
        y_data = [[] for _ in range(num_sensors)]

        for t in range(num_steps):
            target_x = x_positions[t]
            target_y = y_positions[t]

            for sensor_id, sensor_info in enumerate(sensor_data):
                sensor_position = sensor_info['position']
                sensor_range = sensor_info['range']

                if self.distance(
                    (target_x, target_y), sensor_position
                ) <= sensor_range:
                    time_data[sensor_id].append(t)
                    x_data[sensor_id].append(target_x)
                    y_data[sensor_id].append(target_y)

        # Convert lists to NumPy arrays
        time_data = [np.array(data) for data in time_data]
        x_data = [np.array(data) for data in x_data]
        y_data = [np.array(data) for data in y_data]

        return time_data, x_data, y_data
