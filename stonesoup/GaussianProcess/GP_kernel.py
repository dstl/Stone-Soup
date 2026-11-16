# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:49:27 2023

@author: 007
"""

import numpy as np


class KernelFunctions:
    def SE_kernel(self, X1, X2, length_scale=2.0, sigma_f=1.0):
        """
        Squared Exponential Kernel.
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) +\
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

    def ARD_kernel(self, X1, X2, length_scales, sigma_f=1.0):
        """
        Automatic Relevance Determination Kernel.
        """
        # Assuming length_scales is a numpy array of shape (d,)
        sqdist = np.sum(((X1 - X2.T)/length_scales)**2, axis=1)
        return sigma_f**2 * np.exp(-0.5 * sqdist)
