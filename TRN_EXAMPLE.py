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
turnRate = -np.deg2rad(0.25)
deltaT   = 2
nTime    = 100
X0       = np.array([80000,75,35000,0])
P0       = np.diag([120,20,120,20])
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

#### Measurement Model: Map ####
data              = loadmat('/Users/dopestmmac/Desktop/MapTAN.mat')
map_x             = np.array(data['map_m'][0][0][0])
map_y             = np.array(data['map_m'][0][0][1])
map_z             = np.matrix(data['map_m'][0][0][2])
interpolator      = RegularGridInterpolator((map_x[:,0],map_y[0,:]),map_z) 
Rmap              = 1
measurement_model = TerrainAidedNavigation(interpolator,noise_covar = Rmap, mapping=(0, 2))
# plt.figure()
# plt.contourf(map_x,map_y,map_z)
# plt.colorbar()

#### Monte Carlo Runs ####
for mc in range(0,MC):
    
    #### MC Settings ####
    np.random.seed(mc)
    print(mc)

    #### Define Settings ####
    start_time       = datetime.now().replace(microsecond=0)
    transition_model = KnownTurnRate(turn_noise_diff_coeffs = [0.001,0.001], turn_rate = turnRate)
    timesteps        = [start_time]
    truth            = GroundTruthPath([GroundTruthState(np.random.multivariate_normal(X0,P0), timestamp = start_time)])
    # Create the truth path
    for k in range(1,nTime):
        timesteps.append(start_time + deltaT*timedelta(seconds = k))
        truth.append(GroundTruthState(transition_model.function(truth[k - 1], noise = True, time_interval = timedelta(seconds = deltaT)),timestamp = timesteps[k]))
    
    # Populate the measurement array
    measurements = []
    for state in truth:
        measurement = measurement_model.function(state, noise = True)
        measurements.append(Detection(measurement, timestamp = state.timestamp, measurement_model = measurement_model))
    #     plt.scatter(state.state_vector[0],state.state_vector[2])

    # plt.show()

    #### Initialise Point Mass Filter - GSF ####
    predictorGMF    = PointMassPredictor(transition_model)
    updaterGMF      = PointMassUpdater(measurement_model)
    Npa             = np.array([7, 5, 7, 5]) # for FFT must be ODD!!!!
    N               = np.prod(Npa) # number of points - total
    sFactor         = 6 # scaling factor (number of sigmas covered by the grid)
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
    Npa             = np.array([7, 5, 7, 5]) # for FFT must be ODD!!!!
    N               = np.prod(Npa) # number of points - total
    sFactor         = 6 # scaling factor (number of sigmas covered by the grid)
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
        neesGMF[:,kTime,mc]  = errorGMF[:,kTime,mc].reshape(1,nS) @ np.linalg.pinv(covGMF[:,:,kTime,mc]) @ errorGMF[:,kTime,mc].reshape(nS,1)
        kTime               += 1
    end_time = time.time()

    del prediction, hypothesis, post, priorGMF
    
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
        neesPMF[:,kTime,mc]  = errorPMF[:,kTime,mc].reshape(1,nS) @ np.linalg.pinv(covPMF[:,:,kTime,mc]) @ errorPMF[:,kTime,mc].reshape(nS,1)
        kTime               += 1
    end_time = time.time()

    del prediction, hypothesis, post, priorPMF

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
        neesPF[:,kTime,mc]  = errorPF[:,kTime,mc].reshape(1,nS) @ np.linalg.pinv(covPF[:,:,kTime,mc]) @ errorPF[:,kTime,mc].reshape(nS,1)
        kTime              += 1
    end_time = time.time()

    del prediction, hypothesis, post, priorPF

#### Plotting ####
plt.rc('font', family='serif', serif=['Computer Modern'])
plt.rc('text', usetex=True)
plt.rcParams['axes.linewidth']   = 2 # Thicker axes
plt.rcParams['lines.linewidth']  = 2 # Thicker lines
plt.rcParams['xtick.major.size'] = 7 # Major tick length
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.minor.size'] = 4 # Minor tick length
plt.rcParams['ytick.minor.size'] = 4
x_vals                           = np.linspace(1,nTime,nTime)
translucent_blue                 = (0/255,114/255,178/255,0.3)
y_labels                         = [r'$\tilde{r}_x$ (m)',r'$\tilde{v}_x$ (m/s)',r'$\tilde{r}_y$ (m)',r'$\tilde{v}_y$ (m/s)']
cb_colors = {
    'neutral': '#000000',  # black/neutral
    'mean':    '#000000',  # black
    'upper':   '#D55E00',  # red
    'lower':   '#D55E00',  # same for lower bound
    'std':     '#0072B2'   # blue
}

# GMF plots
plt.figure()
for iS in range(nS):
    ax = plt.subplot(2,2,iS + 1)
    ax.plot(x_vals,errorGMF[iS,:,:],color = cb_colors['neutral'],alpha = 0.1)
    ax.plot(x_vals,np.mean(errorGMF[iS,:,:],axis = 1),color = cb_colors['mean'])
    ax.plot(x_vals,+3*np.sqrt(np.mean(covGMF[iS,iS,:,:],axis = 1)),color = cb_colors['upper'])
    ax.plot(x_vals,-3*np.sqrt(np.mean(covGMF[iS,iS,:,:],axis = 1)),color = cb_colors['upper'])
    ax.plot(x_vals,+3*np.std(errorGMF[iS,:,:],axis = 1),color = cb_colors['std'])
    ax.plot(x_vals,-3*np.std(errorGMF[iS,:,:],axis = 1),color = cb_colors['std'])
    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(y_labels[iS])
    ax.minorticks_on()
    ax.tick_params(which = 'both',width = 2)
    ax.tick_params(which = 'major',length = 7)
    ax.tick_params(which = 'minor',length = 4)
plt.tight_layout(pad = 0.5,w_pad = 0.5,h_pad = 0.5)

# PMF plots
plt.figure()
for iS in range(nS):
    ax = plt.subplot(2,2,iS + 1)
    ax.plot(x_vals,errorPMF[iS,:,:],color = cb_colors['neutral'],alpha = 0.1)
    ax.plot(x_vals,np.mean(errorPMF[iS,:,:],axis = 1),color = cb_colors['mean'])
    ax.plot(x_vals,+3*np.sqrt(np.mean(covPMF[iS,iS,:,:],axis = 1)),color = cb_colors['upper'])
    ax.plot(x_vals,-3*np.sqrt(np.mean(covPMF[iS,iS,:,:],axis = 1)),color = cb_colors['upper'])
    ax.plot(x_vals,+3*np.std(errorPMF[iS,:,:],axis = 1),color = cb_colors['std'])
    ax.plot(x_vals,-3*np.std(errorPMF[iS,:,:],axis = 1),color = cb_colors['std'])
    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(y_labels[iS])
    ax.minorticks_on()
    ax.tick_params(which = 'both',width = 2)
    ax.tick_params(which = 'major',length = 7)
    ax.tick_params(which = 'minor',length = 4)
plt.tight_layout(pad = 0.5,w_pad = 0.5,h_pad = 0.5)

# PF plots
plt.figure()
for iS in range(nS):
    ax = plt.subplot(2,2,iS + 1)
    ax.plot(x_vals,errorPF[iS,:,:],color = cb_colors['neutral'],alpha = 0.1)
    ax.plot(x_vals,np.mean(errorPF[iS,:,:],axis = 1),color = cb_colors['mean'])
    ax.plot(x_vals,+3*np.sqrt(np.mean(covPF[iS,iS,:,:],axis = 1)),color = cb_colors['upper'])
    ax.plot(x_vals,-3*np.sqrt(np.mean(covPF[iS,iS,:,:],axis = 1)),color = cb_colors['upper'])
    ax.plot(x_vals,+3*np.std(errorPF[iS,:,:],axis = 1),color = cb_colors['std'])
    ax.plot(x_vals,-3*np.std(errorPF[iS,:,:],axis = 1),color = cb_colors['std'])
    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(y_labels[iS])
    ax.minorticks_on()
    ax.tick_params(which = 'both',width = 2)
    ax.tick_params(which = 'major',length = 7)
    ax.tick_params(which = 'minor',length = 4)
plt.tight_layout(pad = 0.5,w_pad = 0.5,h_pad = 0.5)

fig, axs = plt.subplots(1,3)
# RMSE Position
data_1 =  np.mean(np.sqrt(np.mean(errorGMF[[0,2],:,:]**2,axis = 0)),axis  =  0)
data_2 =  np.mean(np.sqrt(np.mean(errorPMF[[0,2],:,:]**2,axis = 0)),axis = 0)
data_3 =  np.mean(np.sqrt(np.mean(errorPF[[0,2],:,:]**2,axis = 0)),axis = 0)
data   =  [data_1,data_2,data_3]
bp     =  axs[0].boxplot(data,patch_artist = True,
                    boxprops = dict(facecolor = translucent_blue,color = cb_colors['neutral'],linewidth = 2),
                    medianprops = dict(color = cb_colors['upper'],linewidth = 2))
axs[0].minorticks_on()
axs[0].tick_params(which = 'both',width = 2)
axs[0].tick_params(which = 'major',length = 7)
axs[0].tick_params(which = 'minor',length = 4)
axs[0].set_xticks([1,2,3])
axs[0].set_xticklabels(['LbPMF+GSF','LbPMF','PF'])
axs[0].set_ylabel(r'\textbf{RMSE} Position (m)')
axs[0].set_ylim([0,50])

# RMSE Velocity
data_1 =  np.mean(np.sqrt(np.mean(errorGMF[[1,3],:,:]**2,axis = 0)),axis = 0)
data_2 =  np.mean(np.sqrt(np.mean(errorPMF[[1,3],:,:]**2,axis = 0)),axis = 0)
data_3 =  np.mean(np.sqrt(np.mean(errorPF[[1,3],:,:]**2,axis = 0)),axis = 0)
data   =  [data_1,data_2,data_3]
bp     =  axs[1].boxplot(data,patch_artist = True,
                    boxprops = dict(facecolor = translucent_blue,color = cb_colors['neutral'],linewidth = 2),
                    medianprops = dict(color = cb_colors['upper'],linewidth = 2))
axs[1].minorticks_on()
axs[1].tick_params(which = 'both',width = 2)
axs[1].tick_params(which = 'major',length = 7)
axs[1].tick_params(which = 'minor',length = 4)
axs[1].set_xticks([1,2,3])
axs[1].set_xticklabels(['LbPMF+GSF','LbPMF','PF'])
axs[1].set_ylabel(r'\textbf{RMSE} Velocity (m/s)')
axs[1].set_ylim([0,1.5])

# SNEES
data_1 =  np.median(neesGMF,axis = 1)[0]/nS
data_2 =  np.median(neesPMF,axis = 1)[0]/nS
data_3 =  np.median(neesPF,axis = 1)[0]/nS
data   =  [data_1,data_2,data_3]
bp     =  axs[2].boxplot(data,patch_artist = True,
                boxprops = dict(facecolor = translucent_blue,color = cb_colors['neutral'],linewidth = 2),
                medianprops = dict(color = cb_colors['upper'],linewidth = 2))
axs[2].minorticks_on()
axs[2].tick_params(which = 'both',width = 2)
axs[2].tick_params(which = 'major',length = 7)
axs[2].tick_params(which = 'minor',length = 4)
axs[2].set_xticks([1,2,3])
axs[2].set_xticklabels(['LbPMF+GSF','LbPMF','PF'])
axs[2].set_ylabel(r'\textbf{SNEES} (a.u)')
axs[2].set_ylim([0,10])
plt.tight_layout(pad = 0.5,w_pad = 0.5,h_pad = 0.5)

plt.show()