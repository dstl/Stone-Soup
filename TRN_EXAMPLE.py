# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time

from datetime import datetime
from datetime import timedelta

# Initialise Stone Soup ground-truth and transition models.
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.transition.linear import KnownTurnRate
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.nonlinear import TerrainAidedNavigation
from stonesoup.models.measurement.linear import LinearGaussian
from scipy.interpolate import RegularGridInterpolator
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import ESSResampler
from stonesoup.resampler.particle import MultinomialResampler
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.functions import gridCreation
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from numpy.linalg import inv
from stonesoup.types.state import PointMassState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.types.state import GaussianState

from stonesoup.predictor.pointmass import PointMassPredictor
from stonesoup.updater.pointmass import PointMassUpdater
from scipy.stats import multivariate_normal
from scipy.io import loadmat

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater

from stonesoup.types.numeric import Probability  # Similar to a float type
from stonesoup.types.state import ParticleState
from stonesoup.types.array import StateVectors
import json

#### Problem Setup ####
turnRate = np.deg2rad(30)
deltaT   = 1
nTime    = 20
X0       = np.array([36569,50,55581,50])
P0       = np.diag([90,5,160,5])
nS       = X0.shape[0]
MC       = 100

#### Initialise Arrays ####
errorGMF = np.zeros(shape = (nS,nTime,MC))
stateGMF = np.zeros(shape = (nS,nTime,MC))
covGMF   = np.zeros(shape = (nS,nS,nTime,MC))
neesGMF  = np.zeros(shape = (1,nTime,MC))

errorPMF = np.zeros(shape = (nS,nTime,MC)) 
statePMF = np.zeros(shape = (nS,nTime,MC))
covPMF   = np.zeros(shape = (nS,nS,nTime,MC))
neesPMF  = np.zeros(shape = (1,nTime,MC))

errorPF  = np.zeros(shape = (nS,nTime,MC))
statePF  = np.zeros(shape = (nS,nTime,MC))
covPF    = np.zeros(shape = (nS,nS,nTime,MC))
neesPF   = np.zeros(shape = (1,nTime,MC))

errorKF  = np.zeros(shape = (nS,nTime,MC))

#### Monte Carlo Runs ####
for mc in range(0,MC):
    
    #### MC Settings ####
    np.random.seed(mc)
    print(mc)

    #### Define Settings ####
    start_time       = datetime.now().replace(microsecond=0)
    transition_model = KnownTurnRate(turn_noise_diff_coeffs = [2,2], turn_rate = turnRate)
    # This needs to be done in other way
    time_difference  = timedelta(days = 0, hours = 0, minutes = 0, seconds = deltaT)
    timesteps        = [start_time]
    truth            = GroundTruthPath([GroundTruthState(np.random.multivariate_normal(X0,P0), timestamp = start_time)])
    # Create the truth path
    for k in range(1,nTime):
        timesteps.append(start_time + timedelta(seconds = k))
        truth.append(GroundTruthState(transition_model.function(truth[k - 1], noise = True, time_interval = timedelta(seconds = deltaT)),timestamp = timesteps[k]))
    
    #### Measurement Model: Map ####
    data              = loadmat('/Users/dopestmmac/Desktop/MapTAN.mat')
    map_x             = np.array(data['map_m'][0][0][0])
    map_y             = np.array(data['map_m'][0][0][1])
    map_z             = np.matrix(data['map_m'][0][0][2])
    interpolator      = RegularGridInterpolator((map_x[:,0],map_y[0,:]),map_z) 
    Rmap              = 0.1
    measurement_model = TerrainAidedNavigation(interpolator,noise_covar = Rmap, mapping=(0, 2))

    #### Measurement Model: Range and Bearing ####
    # sensor_x          = 36000
    # sensor_y          = 55000
    # RrangeBearing     = np.diag([np.radians(0.1),0.1])
    # measurement_model = CartesianToBearingRange(ndim_state = nS, mapping = (0,2), noise_covar = RrangeBearing, translation_offset = np.array([[sensor_x],[sensor_y]]))
    
    #### Measurement Model: Linear ####
    # matrix            = np.array([[1,0],[0,1],])
    # measurement_model = LinearGaussian(ndim_state = nS, mapping = (0,2), noise_covar = matrix)
    
    # Populate the measurement array
    measurements = []
    for state in truth:
        measurement = measurement_model.function(state, noise = True)
        measurements.append(Detection(measurement, timestamp = state.timestamp, measurement_model = measurement_model))

    #### Initialise Point Mass Filter - GSF ####
    predictorGMF    = PointMassPredictor(transition_model)
    updaterGMF      = PointMassUpdater(measurement_model)
    Npa             = np.array([7, 7, 7, 7]) # for FFT must be ODD!!!!
    N               = np.prod(Npa) # number of points - total
    sFactor         = 4 # scaling factor (number of sigmas covered by the grid)
    [predGrid, predGridDelta, gridDimOld, xOld, Ppold] = gridCreation(np.vstack(X0),P0,sFactor,nS,Npa)
    meanX0          = np.vstack(X0)
    pom             = predGrid - np.matlib.repmat(meanX0,1,N)
    denominator     = np.sqrt((2*np.pi)**nS)*np.linalg.det(P0)
    pompom          = np.sum(-0.5*np.multiply(pom.T@inv(P0),pom.T),1) #elementwise multiplication
    pomexp          = np.exp(pompom)
    predDensityProb = pomexp/denominator # Adding probabilities to points
    predDensityProb = predDensityProb/(sum(predDensityProb)*np.prod(predGridDelta))
    priorGMF        = PointMassState(state_vector = StateVectors(predGrid),
                                     weight       = predDensityProb,
                                     grid_delta   = predGridDelta,
                                     grid_dim     = gridDimOld,
                                     center       = xOld,
                                     eigVec       = Ppold,
                                     Npa          = Npa,
                                     timestamp    = start_time)

    #### Initialise Point Mass Filter - No GSF ####
    predictorPMF    = PointMassPredictor(transition_model)
    updaterPMF      = PointMassUpdater(measurement_model)
    Npa             = np.array([7, 7, 7, 7]) # for FFT must be ODD!!!!
    N               = np.prod(Npa) # number of points - total
    sFactor         = 4 # scaling factor (number of sigmas covered by the grid)
    [predGrid, predGridDelta, gridDimOld, xOld, Ppold] = gridCreation(np.vstack(X0),P0,sFactor,nS,Npa)
    meanX0          = np.vstack(X0)
    pom             = predGrid - np.matlib.repmat(meanX0,1,N)
    denominator     = np.sqrt((2*np.pi)**nS)*np.linalg.det(P0)
    pompom          = np.sum(-0.5*np.multiply(pom.T@inv(P0),pom.T),1) #elementwise multiplication
    pomexp          = np.exp(pompom)
    predDensityProb = pomexp/denominator # Adding probabilities to points
    predDensityProb = predDensityProb/(sum(predDensityProb)*np.prod(predGridDelta))
    priorPMF        = PointMassState(state_vector = StateVectors(predGrid),
                                     weight       = predDensityProb,
                                     grid_delta   = predGridDelta,
                                     grid_dim     = gridDimOld,
                                     center       = xOld,
                                     eigVec       = Ppold,
                                     Npa          = Npa,
                                     timestamp    = start_time)
    
    #### Initialise Particle Filter ####
    predictorPF = ParticlePredictor(transition_model)
    resamplerPF = MultinomialResampler()
    updaterPF   = ParticleUpdater(measurement_model, resamplerPF)
    nParticles  = N
    samplesPF   = multivariate_normal.rvs(X0, P0, size = nParticles)
    priorPF     = ParticleState(state_vector = StateVectors(samplesPF.T), weight = np.array([Probability(1/nParticles)]*nParticles), timestamp = start_time)
    
    # #### Initialise Kalman Filter ####
    # predictorKF = KalmanPredictor(transition_model)
    # updaterKF   = KalmanUpdater(measurement_model)
    # priorKF     = GaussianState(X0, P0, timestamp = start_time)
    
    #### Dynamics ####
    F = transition_model.matrix(prior = priorPF, time_interval = time_difference)
    Q = transition_model.covar(time_interval = time_difference)

    #### Run Point Mass Filter - GSF ####
    start_time = time.time()
    kTime      = 0
    for measurement in measurements:
        prediction           = predictorGMF.predict(priorGMF, timestamp = measurement.timestamp, runGSFversion = True, futureMeas = measurement, measModel = measurement_model)
        hypothesis           = SingleHypothesis(prediction, measurement)
        post                 = updaterGMF.update(hypothesis)
        priorGMF             = post
        errorGMF[:,kTime,mc] = np.array(truth.states[kTime].state_vector).T - post.mean
        stateGMF[:,kTime,mc] = post.mean
        covGMF[:,:,kTime,mc] = np.matrix(post.covar())
        neesGMF[:,kTime,mc]  = errorGMF[:,kTime,mc].reshape(1,nS) @ np.linalg.inv(covGMF[:,:,kTime,mc]) @ errorGMF[:,kTime,mc].reshape(nS,1)
        kTime               += 1
    end_time = time.time()
    
    #### Run Point Mass Filter - No GSF ####
    start_time = time.time()
    kTime      = 0
    for measurement in measurements:
        prediction           = predictorPMF.predict(priorPMF, timestamp = measurement.timestamp, runGSFversion = False, futureMeas = measurement, measModel = measurement_model)
        hypothesis           = SingleHypothesis(prediction, measurement)
        post                 = updaterPMF.update(hypothesis)
        priorPMF             = post
        errorPMF[:,kTime,mc] = np.array(truth.states[kTime].state_vector).T - post.mean
        statePMF[:,kTime,mc] = post.mean
        covPMF[:,:,kTime,mc] = np.matrix(post.covar())
        neesPMF[:,kTime,mc]  = errorPMF[:,kTime,mc].reshape(1,nS) @ np.linalg.inv(covPMF[:,:,kTime,mc]) @ errorPMF[:,kTime,mc].reshape(nS,1)
        kTime               += 1
    end_time = time.time()

    #### Run Particle Filter ####
    start_time = time.time()
    kTime      = 0
    for measurement in measurements:
        prediction          = predictorPF.predict(priorPF, timestamp = measurement.timestamp)
        hypothesis          = SingleHypothesis(prediction, measurement)
        post                = updaterPF.update(hypothesis)
        priorPF             = post
        errorPF[:,kTime,mc] = np.array(truth.states[kTime].state_vector).T - np.array(post.mean).T
        statePF[:,kTime,mc] = np.array(post.mean).T
        covPF[:,:,kTime,mc] = np.matrix(post.covar)
        neesPF[:,kTime,mc]  = errorPF[:,kTime,mc].reshape(1,nS) @ np.linalg.inv(covPF[:,:,kTime,mc]) @ errorPF[:,kTime,mc].reshape(nS,1)
        kTime              += 1
    end_time = time.time()

#### Plotting ####
plt.figure()
for iS in range(0,nS):
    plt.subplot(2,2,iS + 1)
    plt.plot(np.linspace(1,20,nTime),errorGMF[iS,:,:],'k',alpha = 0.1)
    plt.plot(np.linspace(1,20,nTime),np.mean(errorGMF[iS,:,:],1),'k')
    plt.plot(np.linspace(1,20,nTime),+3*np.sqrt(np.mean(covGMF[iS,iS,:,:],1)),'r')
    plt.plot(np.linspace(1,20,nTime),-3*np.sqrt(np.mean(covGMF[iS,iS,:,:],1)),'r')
    plt.plot(np.linspace(1,20,nTime),+3*np.std(errorGMF[iS,:,:],1),'b')
    plt.plot(np.linspace(1,20,nTime),-3*np.std(errorGMF[iS,:,:],1),'b')

plt.figure()
for iS in range(0,nS):
    plt.subplot(2,2,iS + 1)
    plt.plot(np.linspace(1,20,nTime),errorPMF[iS,:,:],'k',alpha = 0.1)
    plt.plot(np.linspace(1,20,nTime),np.mean(errorPMF[iS,:,:],1),'k')
    plt.plot(np.linspace(1,20,nTime),+3*np.sqrt(np.mean(covPMF[iS,iS,:,:],1)),'r')
    plt.plot(np.linspace(1,20,nTime),-3*np.sqrt(np.mean(covPMF[iS,iS,:,:],1)),'r')
    plt.plot(np.linspace(1,20,nTime),+3*np.std(errorPMF[iS,:,:],1),'b')
    plt.plot(np.linspace(1,20,nTime),-3*np.std(errorPMF[iS,:,:],1),'b')

plt.figure()
for iS in range(0,nS):
    plt.subplot(2,2,iS + 1)
    plt.plot(np.linspace(1,20,nTime),errorPF[iS,:,:],'k',alpha = 0.1)
    plt.plot(np.linspace(1,20,nTime),np.mean(errorPF[iS,:,:],1),'k')
    plt.plot(np.linspace(1,20,nTime),+3*np.sqrt(np.mean(covPF[iS,iS,:,:],1)),'r')
    plt.plot(np.linspace(1,20,nTime),-3*np.sqrt(np.mean(covPF[iS,iS,:,:],1)),'r')
    plt.plot(np.linspace(1,20,nTime),+3*np.std(errorPF[iS,:,:],1),'b')
    plt.plot(np.linspace(1,20,nTime),-3*np.std(errorPF[iS,:,:],1),'b')
plt.show()

#### Results ####

print('\n')

print('Position RMSE GMF:')
print(np.sqrt(np.mean(errorGMF[[0,2],:,:]**2,0)).mean())
print('Position RMSE PMF:') 
print(np.sqrt(np.mean(errorPMF[[0,2],:,:]**2,0)).mean())
print('Position RMSE PF:')     
print(np.sqrt(np.mean(errorPF[[0,2],:,:]**2,0)).mean())

print('\n')

print('Velocity RMSE GMF:') 
print(np.sqrt(np.mean(errorGMF[[1,3],:,:]**2,0)).mean())
print('Velocity RMSE PMF:') 
print(np.sqrt(np.mean(errorPMF[[1,3],:,:]**2,0)).mean())
print('Velocity RMSE PF:')     
print(np.sqrt(np.mean(errorPF[[1,3],:,:]**2,0)).mean())

print('\n')

print('SNEES GMF:') 
print(np.mean(neesGMF,1).mean()/nS)
print('SNEES PMF:') 
print(np.mean(neesPMF,1).mean()/nS)
print('SNEES PF:')     
print(np.mean(neesPF,1).mean()/nS)




