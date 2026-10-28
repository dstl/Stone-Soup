# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 13:11:08 2023

@author: 007
"""
from GP import GaussianProcess
import numpy as np


class GP_track():
    def sliding_window(self, t, window_size):
        if window_size <= 0:
            raise ValueError("Invalid window size1")
        if window_size > t:
            start_time = 0
        else:
            start_time = t-window_size + 1
        return start_time

    def update(self, Data_train, Time_train, test_t):
        G = GaussianProcess(kernel_type='SE')
        Time_test = np.array([test_t]).reshape(-1, 1)
        res = G.fit(Time_train, Data_train, flag='GP')
        l_opt, sigma_f_opt, sigma_no = res.x
        mu, cov = G.posterior(Time_test, Time_train, Data_train,
                              length_scale=l_opt, sigma_f=sigma_f_opt,
                              sigma_y=sigma_no)
        return mu, cov

    def update_DGP(self, Data_train, Time_train, test_t):
        G = GaussianProcess()
        Time_test = np.array([test_t]).reshape(-1, 1)

        res = G.fit(Time_train, Data_train, flag='DGP')
        l_opt_dgp, sigma_f_opt_dgp, sigma_no_dgp = res.x
        mu_sd, cov_sd = G.distributed_posterior(
            Time_test, Time_train, Data_train, length_scale=l_opt_dgp,
            sigma_f=sigma_f_opt_dgp, sigma_y=sigma_no_dgp
        )
        mu_sa, cov_sa = G.aggregation(mu_sd, cov_sd, Time_test)

        return mu_sa, cov_sa

    def tracking(self, measurements, window_size):
        Xm = []
        Ym = []
        x_data = []
        x_cov = []
        y_data = []
        y_cov = []

        for measurement in measurements:
            Xm.append(measurement.state_vector[0])

            Ym.append(measurement.state_vector[1])

        Xm = np.array(Xm).reshape(-1, 1)
        Ym = np.array(Ym).reshape(-1, 1)
        T = len(measurements)
        for i in range(2, T):
            SW = self.sliding_window(i, window_size)
            Time_train = np.arange(SW, i).reshape(i-SW, 1)
            X_train = Xm[self.sliding_window(i, window_size):i]

            mu_x, cov_x = self.update(X_train, Time_train, i)
            x_data.append(mu_x)
            x_cov.append(cov_x)

            Y_train = Ym[self.sliding_window(i, window_size):i]
            mu_y, cov_y = self.update(Y_train, Time_train, i)
            y_data.append(mu_y)
            y_cov.append(cov_y)
        x_data = np.array(x_data).reshape(-1)
        x_cov = np.array(x_cov).reshape(-1)
        y_data = np.array(y_data).reshape(-1)
        y_cov = np.array(y_cov).reshape(-1)
        return x_data, x_cov, y_data, y_cov

    def tracking_DGP(self, measurements, time_data, x_train, y_train):
        x_data = []
        x_cov = []
        y_data = []
        y_cov = []
        # List to keep track of sensor count at each time point
        sensors_count = []
        T = len(measurements)
        x_s = []
        y_s = []
        for i in range(3, T):
            X_test_dgp = i
            time_data_filtered = []
            x_data_filtered = []
            y_data_filtered = []

            sensor_counter = 0  # Initialize sensor counter for this time point

            for sensor_id in range(len(y_train)):
                indices_to_keep = np.where(
                    np.array(time_data[sensor_id]) < i+1)[0]

                # Filter the time data for the current sensor
                time_filtered = np.array(time_data[sensor_id])[
                    indices_to_keep].reshape(-1, 1)
                time_data_filtered.append(time_filtered)

                # Filter the x data for the current sensor
                x_filtered = np.array(x_train[sensor_id])[
                    indices_to_keep].reshape(-1, 1)
                # Replace mu and sigma with your chosen values
                x_noise = np.random.normal(0, 2.3, x_filtered.shape)
                x_filtered += x_noise
                x_data_filtered.append(x_filtered)

                # Filter the y data for the current sensor
                y_filtered = np.array(y_train[sensor_id])[
                    indices_to_keep].reshape(-1, 1)
                # Replace mu and sigma with your chosen values
                y_noise = np.random.normal(0, 2.3, y_filtered.shape)
                y_filtered += y_noise
                y_data_filtered.append(y_filtered)
                if len(time_filtered) > 0:
                    sensor_counter += 1

            # Filter to keep only time points where there are at least 3
            time_dataF = [
                data for data in time_data_filtered if len(data) >= 3]
            x_trainF = [data for data in x_data_filtered if len(data) >= 3]
            y_trainF = [data for data in y_data_filtered if len(data) >= 3]

            mu_x, cov_x = self.update_DGP(x_trainF, time_dataF, X_test_dgp)
            x_data.append(mu_x)
            x_cov.append(cov_x)

            mu_y, cov_y = self.update_DGP(y_trainF, time_dataF, X_test_dgp)
            y_data.append(mu_y)
            y_cov.append(cov_y)

            # Add the sensor count for this time point to the list
            sensors_count.append(sensor_counter)

        # Convert the collected data to arrays and reshape
        x_data = np.array(x_data).reshape(-1)
        x_cov = np.array(x_cov).reshape(-1)
        y_data = np.array(y_data).reshape(-1)
        y_cov = np.array(y_cov).reshape(-1)

        x_s = np.vstack(x_trainF)
        y_s = np.vstack(y_trainF)
        t_s = np.vstack(time_dataF)

        t_id = t_s[:, 0].argsort()

        t_s = t_s[t_id]
        x_s = x_s[t_id]
        y_s = y_s[t_id]
        # Return the additional list of sensor counts
        # along with the original outputs
        return x_data, x_cov, y_data, y_cov, x_s, y_s, t_s
