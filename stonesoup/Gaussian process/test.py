# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:52:45 2023

@author: 007
"""
import numpy as np
import matplotlib.pyplot as plt
from GP import GaussianProcess

# Set random seed for reproducibility
np.random.seed(0)

# Generate data points
X_train = np.linspace(0, 10, 20).reshape(-1, 1)
Y_train = np.sin(X_train) + np.random.randn(20, 1) * 0.5
# Sine data with noise

# Initialize Gaussian Process model for standard GP
gp = GaussianProcess(kernel_type='SE')

# Fit the standard GP model
gp.fit(X_train, Y_train, 'GP')

# New input points for predictions
X_s = np.linspace(0, 10, 100).reshape(-1, 1)

# Get posterior distribution for standard GP
mu_s, cov_s = gp.posterior(X_s, X_train, Y_train)

# Plot for standard GP
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
gp.plot_gp(mu_s, cov_s, X_s, X_train, Y_train)
plt.title("Standard Gaussian Process")

# Initialize Gaussian Process model for DGP
dgp = GaussianProcess(kernel_type='SE')

# Prepare distributed data (for simplicity, split the
# original data into two parts)
X_train_dgp = [X_train[:10], X_train[10:]]
Y_train_dgp = [Y_train[:10], Y_train[10:]]

# Fit the DGP model
dgp.fit(X_train_dgp, Y_train_dgp, 'DGP')

# Get posterior distribution for DGP
mu_set, cov_set = dgp.distributed_posterior(X_s, X_train_dgp, Y_train_dgp)

# Aggregate results for DGP
mu_agg, cov_agg = dgp.aggregation(mu_set, cov_set, X_s)

# Plot for DGP
plt.subplot(1, 2, 2)
dgp.plot_gp(mu_agg, cov_agg, X_s)
plt.title("Distributed Gaussian Process")

# Show plots
plt.tight_layout()
plt.show()
