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
        G = GaussianProcess()
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
        T = len(measurements)
        for i in range(3, T):
            X_test_dgp = i
            time_data_filtered = []
            x_data_filtered = []
            y_data_filtered = []

            for sensor_id in range(len(y_train)):
                indices_to_keep = np.where(
                    np.array(time_data[sensor_id]) < i+1)[0]
                time_data_filtered.append(np.array(time_data[sensor_id])[
                                          indices_to_keep].reshape(-1, 1))
                x_data_filtered.append(np.array(x_train[sensor_id])[
                                       indices_to_keep].reshape(-1, 1))
                y_data_filtered.append(np.array(y_train[sensor_id])[
                                       indices_to_keep].reshape(-1, 1))

            # time_data = time_data_filtered
            # x_data = x_data_filtered
            # y_data = y_data_filtered
            time_dataF = [np.array(data)
                          for data in time_data_filtered if len(data) >= 3]
            x_trainF = [np.array(data)
                        for data in x_data_filtered if len(data) >= 3]
            y_trainF = [np.array(data)
                        for data in y_data_filtered if len(data) >= 3]

            mu_x, cov_x = self.update_DGP(x_trainF, time_dataF, X_test_dgp)
            # print(mu_x)

            x_data.append(mu_x)
            x_cov.append(cov_x)
            # x_data = np.vstack((x_data,mu_x))
            # x_cov.vstack(cov_x)

            mu_y, cov_y = self.update_DGP(y_trainF, time_dataF, X_test_dgp)
            y_data.append(mu_y)
            y_cov.append(cov_y)

        x_data = np.array(x_data).reshape(-1)
        x_cov = np.array(x_cov).reshape(-1)
        y_data = np.array(y_data).reshape(-1)
        y_cov = np.array(y_cov).reshape(-1)

        return x_data, x_cov, y_data, y_cov
