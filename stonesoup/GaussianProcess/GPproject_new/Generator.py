import numpy as np
from datetime import datetime, timedelta
from stonesoup.kernel import GaussianKernel
import matplotlib.pyplot as plt



def generate_groundtruth(num_points, mode="S1", noise_std=0):
    """
    Generate advanced target tracking trajectories with enhancements.
    
    Parameters
    ----------
    num_points : int
        Number of trajectory points.
    mode : str
        Trajectory type: "S1", "S2", "S3", "S4".
    noise_std : float
        Standard deviation of Gaussian noise.
    
    Returns
    -------
    trajectory_x : np.ndarray
        x-coordinates of the trajectory.
    trajectory_y : np.ndarray
        y-coordinates of the trajectory.
    """
    t = np.linspace(0, 10, num_points)
    
    if mode == "S1":  # Wide-angle turn with small and large turns
        half_points = num_points // 2
        quarter_points = num_points // 4

        # Add a small turn first
        trajectory_x_1 = np.linspace(0, 300, quarter_points)
        trajectory_y_1 = np.linspace(0, 200, quarter_points)

        # Add the large turn
        trajectory_x_2 = np.linspace(300, 500, half_points) + 50 * np.sin(0.5 * np.linspace(0, np.pi, half_points))
        trajectory_y_2 = np.linspace(200, 700, half_points)

        # Add the return straight line
        trajectory_x_3 = np.linspace(500, 300, quarter_points)
        trajectory_y_3 = np.linspace(700, 1000, quarter_points)

        # Combine all segments
        trajectory_x = np.concatenate([trajectory_x_1, trajectory_x_2, trajectory_x_3])
        trajectory_y = np.concatenate([trajectory_y_1, trajectory_y_2, trajectory_y_3])

        # Ensure trajectory length matches num_points
        trajectory_x = np.interp(np.linspace(0, len(trajectory_x) - 1, num_points), np.arange(len(trajectory_x)), trajectory_x)
        trajectory_y = np.interp(np.linspace(0, len(trajectory_y) - 1, num_points), np.arange(len(trajectory_y)), trajectory_y)

    elif mode == "S2":  # Coordinated turn matched
        trajectory_x = 500 + 400 * np.cos(0.3 * t)
        trajectory_y = 500 + 400 * np.sin(0.3 * t)
    elif mode == "S3":  # Coordinated turn mismatched (shorter straight lines)
        trajectory_x = 500 + 200 * np.cos(0.5 * t) + 50 * np.sin(1.5 * t)
        trajectory_y = 500 + 200 * np.sin(0.5 * t) + 50 * np.cos(1.5 * t)
    elif mode == "S4":  # Singer matched with more complexity
        trajectory_x = np.cumsum(15 + 8 * np.sin(0.3 * np.pi * t) + 5 * np.cos(0.6 * np.pi * t))
        trajectory_y = np.cumsum(15 + 8 * np.cos(0.3 * np.pi * t) + 5 * np.sin(0.6 * np.pi * t))
    else:
        raise ValueError(f"Unsupported trajectory mode: {mode}")
    
    # Add Gaussian noise
    trajectory_x += np.random.normal(0, noise_std, num_points)
    trajectory_y += np.random.normal(0, noise_std, num_points)
    
    return trajectory_x, trajectory_y
