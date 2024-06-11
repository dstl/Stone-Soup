#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 08:53:07 2024

@author: matoujak
"""

from stonesoup.types.state import PointMassState
from stonesoup.functions import gridCreationFFT
import numpy as np
from numpy.linalg import inv
from stonesoup.types.array import StateVectors
from datetime import datetime
import numpy.matlib
start_time = datetime.now().replace(microsecond=0)


# Initial condition - Gaussian
nx = 2
meanX0 = np.array([20, 5]) # mean value
varX0 = np.array([[0.1, 0], [0, 0.1]]) # variance
Npa = np.array([21, 21]) # number of points per axis, for FFT must be ODD!!!!
N = np.prod(Npa) # number of points - total
sFactor = 4 # scaling factor (number of sigmas covered by the grid)


[predGrid, gridDimOld, predGridDelta] = gridCreationFFT(np.vstack(meanX0),varX0,sFactor,nx,Npa)

meanX0 = np.vstack(meanX0)
pom = predGrid-np.matlib.repmat(meanX0,1,N)
denominator = np.sqrt((2*np.pi)**nx)*np.linalg.det(varX0)
pompom = np.sum(-0.5*np.multiply(pom.T@inv(varX0),pom.T),1) #elementwise multiplication
pomexp = np.exp(pompom)
predDensityProb = pomexp/denominator # Adding probabilities to points
predDensityProb = predDensityProb/(sum(predDensityProb)*np.prod(predGridDelta))

np.hstack(predGrid @ predDensityProb * np.prod(predGridDelta))

prior = PointMassState(state_vector=StateVectors(predGrid),
                      weight=predDensityProb,
                      grid_delta = predGridDelta,
                      timestamp=start_time)

a = prior.covar