import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from datetime import datetime, timedelta

class GaussianProcess:
    def __init__(self, kernel):
        """
        Gaussian Process training class.

        Parameters
        ----------
        kernel : callable
            Kernel function to use for the Gaussian process.
        """
        self.kernel = kernel
        self.hyperparameters = None
        self.is_trained = False
        self.timestamps = None
        self.states = None
        self.dimensions = None
        self.t_start = None
        self.t_scale = None

    def _normalize_timestamps(self, timestamps):
        """
        Normalize timestamps for numerical stability.
        Ensure timestamps are converted to POSIX format if needed.
        """
        if isinstance(timestamps[0], datetime):  # Check if timestamps are datetime objects
            timestamps = np.array([ts.timestamp() for ts in timestamps])
        return (timestamps - self.t_start) / self.t_scale

    def _nll(self, theta, X_train, Y_train):
        """
        Negative log marginal likelihood for a single dimension.
        """
        length_scale, sigma_f, sigma_y = theta
        K = self.kernel(X_train, X_train, length_scale=length_scale, sigma_f=sigma_f)
        K += sigma_y**2 * np.eye(len(X_train))  # Add noise variance

        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return np.inf

        S1 = solve_triangular(L, Y_train, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)
        nll = np.sum(np.log(np.diagonal(L))) + 0.5 * Y_train.T @ S2 + 0.5 * len(X_train) * np.log(2 * np.pi)
        return nll

    def _train_single_dimension(self, timestamps, states):
        """
        Train a single dimension of the Gaussian Process.
        """
        # Initial guess for hyperparameters
        initial_guess = [1.0, 1.0, 0.1]
        bounds = [(1e-3, None), (1e-3, None), (1e-5, None)]
        max_attempts = 2  # Number of attempts allowed
        attempt = 0
    
        while attempt < max_attempts:
            # Try to optimize with the current initial guess
            result = minimize(
                self._nll,
                initial_guess,
                args=(timestamps, states),
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": 2000}
            )
    
            # If optimization succeeds, return the optimized parameters
            if result.success:
                return {
                    "length_scale": result.x[0],
                    "sigma_f": result.x[1],
                    "sigma_y": result.x[2]
                }
    
            # If optimization fails, adjust the initial guess
            initial_guess = [guess * 1.5 if i < 2 else guess * 2 for i, guess in enumerate(initial_guess)]
            attempt += 1
    
        # If all attempts fail, return fallback values
        return {"length_scale": 1.0, "sigma_f": 1.0, "sigma_y": 0.1}

    def train(self, groundtruth_states, window_seconds=None):
        """
        Train the Gaussian Process model using ground truth states.

        Parameters
        ----------
        groundtruth_states : list of GroundTruthState
            List of ground truth states.
        window_seconds : float, optional
            The sliding window size in seconds. If provided, only the most recent
            `window_seconds` of data will be used for training.
        """
        timestamps = np.array([state.timestamp.timestamp() for state in groundtruth_states])
        states = np.array([state.state_vector.flatten() for state in groundtruth_states])

        # Apply sliding window if specified
        if window_seconds is not None:
            current_time = timestamps[-1]
            mask = timestamps >= current_time - window_seconds
            timestamps = timestamps[mask]
            states = states[mask]

        # Normalize timestamps
        self.t_start = timestamps[0]
        self.t_scale = 10.0  # Normalize timestamps to smaller range
        self.timestamps = self._normalize_timestamps(timestamps).reshape(-1, 1)
        self.states = states

        # Determine dimensions and train each dimension
        self.dimensions = states.shape[1]
        self.hyperparameters = []
        for dim in range(self.dimensions):
            hyperparams = self._train_single_dimension(self.timestamps, states[:, dim])
            self.hyperparameters.append(hyperparams)

        self.is_trained = True
        # Removed print for optimized hyperparameters

    def posterior(self, test_timestamps):
        """
        Compute the posterior mean and covariance for all dimensions.

        Parameters
        ----------
        test_timestamps : list of datetime
            List of test timestamps.

        Returns
        -------
        dict
            Posterior mean and covariance for each dimension.
        """
        # Ensure test timestamps are POSIX time
        test_timestamps_normalized = self._normalize_timestamps(test_timestamps)
        X_s = test_timestamps_normalized.reshape(-1, 1)

        results = {}
        for dim in range(self.dimensions):
            params = self.hyperparameters[dim]

            K = self.kernel(self.timestamps, self.timestamps,
                            length_scale=params["length_scale"], sigma_f=params["sigma_f"])
            K += params["sigma_y"]**2 * np.eye(len(self.timestamps))
            K_s = self.kernel(self.timestamps, X_s,
                              length_scale=params["length_scale"], sigma_f=params["sigma_f"])
            K_ss = self.kernel(X_s, X_s, length_scale=params["length_scale"], sigma_f=params["sigma_f"])
            K_ss += 1e-5 * np.eye(len(X_s))  # Add jitter for numerical stability
            K_inv = np.linalg.inv(K)

            mu_s = K_s.T @ K_inv @ self.states[:, dim].reshape(-1, 1)
            cov_s = K_ss - K_s.T @ K_inv @ K_s

            results[dim] = {"mean": mu_s, "covariance": cov_s}

        return results
