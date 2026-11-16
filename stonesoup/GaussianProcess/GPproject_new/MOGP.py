# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 03:55:56 2025

@author: 007
"""
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from datetime import datetime, timedelta
from stonesoup.types.groundtruth import GroundTruthState
from Generator import generate_groundtruth

class MultiOutputGP:
    def __init__(self, kernel):
        self.kernel = kernel
        self.hyperparameters = None
        self.is_trained = False
        self.timestamps = None
        self.states = None
        self.t_start = None
        self.t_scale = None

    def _normalize_timestamps(self, timestamps):
        """
        Normalize timestamps for numerical stability.
        """
        timestamps = np.array(timestamps)
        return (timestamps - self.t_start) / self.t_scale

    def _nll(self, theta, X_train, Y_train):
        """
        Negative log marginal likelihood for joint GP.
        """
        length_scale, sigma_f, sigma_y = theta
        K = self.kernel(X_train, X_train, length_scale=length_scale, sigma_f=sigma_f)
        K += sigma_y**2 * np.eye(K.shape[0])  # Add noise variance

        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return np.inf

        S1 = solve_triangular(L, Y_train, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)
        nll = np.sum(np.log(np.diagonal(L))) + 0.5 * Y_train.T @ S2 + 0.5 * len(X_train) * np.log(2 * np.pi)
        return nll

    def train(self, timestamps, states):
        """
        Train the Multi-output Gaussian Process using joint data.
        """
        self.t_start = timestamps[0]
        self.t_scale = 10.0  # Normalize timestamps to smaller range
        self.timestamps = self._normalize_timestamps(timestamps).reshape(-1, 1)
        self.states = states.reshape(-1, 1)
    
        # Initial guess for hyperparameters
        initial_guess = [1.0, 1.0, 0.1]
        bounds = [(1e-3, None), (1e-3, None), (1e-5, None)]
        max_attempts = 3  # Number of optimization attempts
        attempt = 0
    
        while attempt < max_attempts:
            result = minimize(
                self._nll,
                initial_guess,
                args=(self.timestamps, self.states),
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": 2000}
            )
    
            # If optimization succeeds, save hyperparameters and exit
            if result.success:
                self.hyperparameters = {"length_scale": result.x[0], "sigma_f": result.x[1], "sigma_y": result.x[2]}
                self.is_trained = True
                return
    
            # If optimization fails, adjust the initial guess and retry
            print(f"Optimization failed on attempt {attempt + 1}: {result.message}")
            initial_guess = [guess * 1.5 if i < 2 else guess * 2 for i, guess in enumerate(initial_guess)]
            attempt += 1
    
        # If all attempts fail, raise an error
        raise RuntimeError("Optimization failed after multiple attempts.")


    def posterior(self, test_timestamps):
        """
        Compute posterior mean and covariance for joint GP.
        """
        test_timestamps = self._normalize_timestamps(test_timestamps).reshape(-1, 1)
        params = self.hyperparameters

        K = self.kernel(self.timestamps, self.timestamps,
                        length_scale=params["length_scale"], sigma_f=params["sigma_f"])
        K += params["sigma_y"]**2 * np.eye(K.shape[0])
        K_s = self.kernel(self.timestamps, test_timestamps,
                          length_scale=params["length_scale"], sigma_f=params["sigma_f"])
        K_ss = self.kernel(test_timestamps, test_timestamps,
                           length_scale=params["length_scale"], sigma_f=params["sigma_f"])
        K_ss += 1e-5 * np.eye(K_ss.shape[0])  # Add jitter for numerical stability
        K_inv = np.linalg.inv(K)

        mu_s = K_s.T @ K_inv @ self.states
        cov_s = K_ss - K_s.T @ K_inv @ K_s

        return mu_s, cov_s