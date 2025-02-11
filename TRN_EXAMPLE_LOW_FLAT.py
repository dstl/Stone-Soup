# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time

from datetime import datetime
from datetime import timedelta

# Initialise Stone Soup ground-truth and transition models.
from stonesoup.models.transition.linear import KnownTurnRate
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.nonlinear import TerrainAidedNavigation
from scipy.interpolate import RegularGridInterpolator
from stonesoup.predictor.particle import ParticlePredictor

from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater

from stonesoup.resampler.particle import SystematicResampler
from stonesoup.resampler.particle import ResidualResampler
from stonesoup.resampler.particle import StratifiedResampler
from stonesoup.resampler.particle import ESSResampler

from stonesoup.types.state import GaussianState

from stonesoup.updater.particle import ParticleUpdater
from stonesoup.functions import gridCreation
from numpy.linalg import inv
from stonesoup.types.state import PointMassState
from stonesoup.types.hypothesis import SingleHypothesis

from stonesoup.predictor.pointmass import PointMassPredictor
from stonesoup.updater.pointmass import PointMassUpdater
from scipy.stats import multivariate_normal
from scipy.io import loadmat

from stonesoup.types.numeric import Probability  # Similar to a float type
from stonesoup.types.state import ParticleState
from stonesoup.types.array import StateVectors

#### Problem Setup ####
turnRate = -np.deg2rad(0.25)
deltaT   = 2
nTime    = 100
X0       = np.array([80000,75,35000,0])
P0       = np.diag([120,20,120,20])
nS       = X0.shape[0]
MC       = 100

# Preallocate the timing arrays for each method
end_time_GMF = np.zeros(MC)
end_time_PMF = np.zeros(MC)
end_time_Strat = np.zeros(MC)
end_time_Residual = np.zeros(MC)
end_time_Systematic = np.zeros(MC)
end_time_UKF = np.zeros(MC)

start_time_GMF = np.zeros(MC)
start_time_PMF = np.zeros(MC)
start_time_Strat = np.zeros(MC)
start_time_Residual = np.zeros(MC)
start_time_Systematic = np.zeros(MC)
start_time_UKF = np.zeros(MC)

#### Initialise Arrays ####
errorGMF = np.zeros(shape = (nS, nTime, MC))
stateGMF = np.zeros(shape = (nS, nTime, MC))
covGMF   = np.zeros(shape = (nS, nS, nTime, MC))
neesGMF  = np.zeros(shape = (1, nTime, MC))


errorPMF = np.zeros(shape = (nS, nTime, MC)) 
statePMF = np.zeros(shape = (nS, nTime, MC))
covPMF   = np.zeros(shape = (nS, nS, nTime, MC))
neesPMF  = np.zeros(shape = (1, nTime, MC))

errorUKF = np.zeros(shape = (nS, nTime, MC)) 
stateUKF = np.zeros(shape = (nS, nTime, MC))
covUKF   = np.zeros(shape = (nS, nS, nTime, MC))
neesUKF  = np.zeros(shape = (1, nTime, MC))

# Initialize result variables for Stratified Particle Filter
errorPF_Strat = np.zeros((nS, nTime, MC))
statePF_Strat = np.zeros((nS, nTime, MC))
covPF_Strat = np.zeros((nS, nS, nTime, MC))
neesPF_Strat = np.zeros((1, nTime, MC))

# Initialize result variables for Residual Particle Filter
errorPF_Residual = np.zeros((nS, nTime, MC))
statePF_Residual = np.zeros((nS, nTime, MC))
covPF_Residual = np.zeros((nS, nS, nTime, MC))
neesPF_Residual = np.zeros((1, nTime, MC))

# Initialize result variables for Systematic Particle Filter
errorPF_Systematic = np.zeros((nS, nTime, MC))
statePF_Systematic = np.zeros((nS, nTime, MC))
covPF_Systematic = np.zeros((nS, nS, nTime, MC))
neesPF_Systematic = np.zeros((1, nTime, MC))


#### Measurement Model: Map ####
data              = loadmat('/Users/matoujak/Desktop/MapTAN.mat')
map_x             = np.array(data['map_m'][0][0][0])
map_y             = np.array(data['map_m'][0][0][1])
map_z             = np.matrix(data['map_m'][0][0][2])
interpolator      = RegularGridInterpolator((map_x[:,0],map_y[0,:]),map_z) 
Rmap              = 1
measurement_model = TerrainAidedNavigation(interpolator,noise_covar = Rmap, mapping=(0, 2))


plt.figure()
plt.contourf(map_x,map_y,map_z)
plt.colorbar()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Create the figure and 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot filled contours
# contour = ax.contourf(map_x, map_y, map_z, 100, cmap='viridis')  # Adjust '20' for more/less contour levels

# # Add a colorbar to the plot
# fig.colorbar(contour, ax=ax)

# # Show the plot
# plt.show()


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
        plt.scatter(state.state_vector[0],state.state_vector[2])

    plt.show()
    
    #### Initialise UKF 
    predictorUKF = UnscentedKalmanPredictor(transition_model)####
    updaterUKF = UnscentedKalmanUpdater(measurement_model)
    priorUKF = GaussianState([ [80000],[75],[35000],[0]], np.diag([120, 20, 120, 20]), timestamp=start_time)
    

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
    Npa             = np.array([9, 5, 9, 5]) # for FFT must be ODD!!!!
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
    
    nParticles_Strat = np.round(N*4.5).astype(int)
    nParticles_Res = np.round(N*3).astype(int)
    nParticles_Sys =  np.round(N*4.5).astype(int)


    
    #### Initialise Particle Filter - Stratified ####
    predictorPF_Strat = ParticlePredictor(transition_model)
    resamplerPF_Strat = StratifiedResampler()
    resamplerPF_Strat = ESSResampler(threshold = nParticles_Strat*0.8, resampler = resamplerPF_Strat)
    updaterPF_Strat   = ParticleUpdater(measurement_model, resamplerPF_Strat)
    samplesPF_Strat   = multivariate_normal.rvs(X0, P0, size = nParticles_Strat)
    priorPF_Strat     = ParticleState(state_vector = StateVectors(samplesPF_Strat.T), weight = np.array([Probability(1/nParticles_Strat)]*nParticles_Strat), timestamp = start_time)
    
    #### Initialise Particle Filter - Residual ####
    predictorPF_Residual = ParticlePredictor(transition_model)
    resamplerPF_Residual = ResidualResampler()
    resamplerPF_Residual = ESSResampler(threshold = nParticles_Res*0.8, resampler = resamplerPF_Residual)
    updaterPF_Residual   = ParticleUpdater(measurement_model, resamplerPF_Residual)
    samplesPF_Residual   = multivariate_normal.rvs(X0, P0, size = nParticles_Res)
    priorPF_Residual     = ParticleState(state_vector = StateVectors(samplesPF_Residual.T), weight = np.array([Probability(1/nParticles_Res)]*nParticles_Res), timestamp = start_time)
    
    #### Initialise Particle Filter - Systematic ####
    predictorPF_Systematic = ParticlePredictor(transition_model)
    resamplerPF_Systematic = SystematicResampler()
    resamplerPF_Systematic = ESSResampler(threshold = nParticles_Sys*0.8, resampler = resamplerPF_Systematic)
    updaterPF_Systematic   = ParticleUpdater(measurement_model, resamplerPF_Systematic)
    samplesPF_Systematic   = multivariate_normal.rvs(X0, P0, size = nParticles_Sys)
    priorPF_Systematic     = ParticleState(state_vector = StateVectors(samplesPF_Systematic.T), weight = np.array([Probability(1/nParticles_Sys)]*nParticles_Sys), timestamp = start_time)

    
    # Pick best and use with and without ESS
    
    
    #### Run UKF ####
    start_time_UKF[mc] = time.time()
    kTime      = 0
    for measurement in measurements:
        prediction           = predictorUKF.predict(priorUKF, timestamp = measurement.timestamp)
        hypothesis           = SingleHypothesis(prediction, measurement)
        post                 = updaterUKF.update(hypothesis)
        priorUKF            = post
        errorUKF[:,kTime,mc] = np.array(truth.states[kTime].state_vector).T - post.mean.T
        stateUKF[:,kTime,mc] = post.mean.T
        covUKF[:,:,kTime,mc] = np.matrix(post.covar)
        neesUKF[:,kTime,mc]  = errorUKF[:,kTime,mc].reshape(1,nS) @ np.linalg.pinv(covUKF[:,:,kTime,mc]) @ errorUKF[:,kTime,mc].reshape(nS,1)
        kTime               += 1
    end_time_UKF[mc] = time.time()

    del prediction, hypothesis, post
    

    #### Run Point Mass Filter - GSF ####
    start_time_GMF[mc] = time.time()
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
    end_time_GMF[mc] = time.time()

    del prediction, hypothesis, post, priorGMF
    
    #### Run Point Mass Filter - No GSF ####
    start_time_PMF[mc] = time.time()
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
    end_time_PMF[mc] = time.time()

    del prediction, hypothesis, post, priorPMF

    #### Run Particle Filter - Stratified ####
    start_time_Strat[mc] = time.time()
    kTime      = 0
    for measurement in measurements:
        prediction_Strat     = predictorPF_Strat.predict(priorPF_Strat, timestamp = measurement.timestamp)
        hypothesis_Strat     = SingleHypothesis(prediction_Strat, measurement)
        post_Strat           = updaterPF_Strat.update(hypothesis_Strat)
        priorPF_Strat        = post_Strat
        errorPF_Strat[:,kTime,mc] = np.array(truth.states[kTime].state_vector).T - np.array(post_Strat.mean).T
        statePF_Strat[:,kTime,mc] = np.array(post_Strat.mean).T
        covPF_Strat[:,:,kTime,mc] = np.matrix(post_Strat.covar)
        neesPF_Strat[:,kTime,mc]  = errorPF_Strat[:,kTime,mc].reshape(1,nS) @ np.linalg.pinv(covPF_Strat[:,:,kTime,mc]) @ errorPF_Strat[:,kTime,mc].reshape(nS,1)
        kTime                += 1
    end_time_Strat[mc] = time.time()
    del prediction_Strat, hypothesis_Strat, post_Strat, priorPF_Strat
    
    #### Run Particle Filter - Residual ####
    start_time_Residual[mc] = time.time()
    kTime      = 0
    for measurement in measurements:
        prediction_Residual     = predictorPF_Residual.predict(priorPF_Residual, timestamp = measurement.timestamp)
        hypothesis_Residual     = SingleHypothesis(prediction_Residual, measurement)
        post_Residual           = updaterPF_Residual.update(hypothesis_Residual)
        priorPF_Residual        = post_Residual
        errorPF_Residual[:,kTime,mc] = np.array(truth.states[kTime].state_vector).T - np.array(post_Residual.mean).T
        statePF_Residual[:,kTime,mc] = np.array(post_Residual.mean).T
        covPF_Residual[:,:,kTime,mc] = np.matrix(post_Residual.covar)
        neesPF_Residual[:,kTime,mc]  = errorPF_Residual[:,kTime,mc].reshape(1,nS) @ np.linalg.pinv(covPF_Residual[:,:,kTime,mc]) @ errorPF_Residual[:,kTime,mc].reshape(nS,1)
        kTime                    += 1
    end_time_Residual[mc] = time.time()
    del prediction_Residual, hypothesis_Residual, post_Residual, priorPF_Residual
    
    #### Run Particle Filter - Systematic ####
    start_time_Systematic[mc] = time.time()
    kTime      = 0
    for measurement in measurements:
        prediction_Systematic    = predictorPF_Systematic.predict(priorPF_Systematic, timestamp = measurement.timestamp)
        hypothesis_Systematic    = SingleHypothesis(prediction_Systematic, measurement)
        post_Systematic          = updaterPF_Systematic.update(hypothesis_Systematic)
        priorPF_Systematic       = post_Systematic
        errorPF_Systematic[:,kTime,mc] = np.array(truth.states[kTime].state_vector).T - np.array(post_Systematic.mean).T
        statePF_Systematic[:,kTime,mc] = np.array(post_Systematic.mean).T
        covPF_Systematic[:,:,kTime,mc] = np.matrix(post_Systematic.covar)
        neesPF_Systematic[:,kTime,mc]  = errorPF_Systematic[:,kTime,mc].reshape(1,nS) @ np.linalg.pinv(covPF_Systematic[:,:,kTime,mc]) @ errorPF_Systematic[:,kTime,mc].reshape(nS,1)
        kTime                    += 1
    end_time_Systematic[mc] = time.time()
    del prediction_Systematic, hypothesis_Systematic, post_Systematic, priorPF_Systematic


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

# Boxplot for RMSE and SNEES comparison
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# RMSE Position
data_1 = np.mean(np.sqrt(np.mean(errorGMF[[0, 2], :, :]**2, axis=0)), axis=0)
data_2 = np.mean(np.sqrt(np.mean(errorPMF[[0, 2], :, :]**2, axis=0)), axis=0)
data_3 = np.mean(np.sqrt(np.mean(errorPF_Systematic[[0, 2], :, :]**2, axis=0)), axis=0)
data_4 = np.mean(np.sqrt(np.mean(errorPF_Residual[[0, 2], :, :]**2, axis=0)), axis=0)
data_5 = np.mean(np.sqrt(np.mean(errorPF_Strat[[0, 2], :, :]**2, axis=0)), axis=0)
data_6 = np.mean(np.sqrt(np.mean(errorUKF[[0, 2], :, :]**2, axis=0)), axis=0)
data = [data_1, data_2, data_3, data_4, data_5, data_6]
bp     =  axs[0].boxplot(data,patch_artist = True,
                    boxprops = dict(facecolor = translucent_blue,color = cb_colors['neutral'],linewidth = 2),
                    medianprops = dict(color = cb_colors['upper'],linewidth = 2))
axs[0].minorticks_on()
axs[0].tick_params(which='both', width=2)
axs[0].tick_params(which='major', length=7)
axs[0].tick_params(which='minor', length=4)
axs[0].set_xticks([1, 2, 3, 4, 5, 6])
axs[0].set_xticklabels(['GMF', 'PMF', 'Sys', 'Res', 'Strat','UKF'])
axs[0].set_ylabel(r'\textbf{RMSE} Position (m)')
axs[0].set_ylim([0, 20])

# RMSE Velocity
data_1 = np.mean(np.sqrt(np.mean(errorGMF[[1, 3], :, :]**2, axis=0)), axis=0)
data_2 = np.mean(np.sqrt(np.mean(errorPMF[[1, 3], :, :]**2, axis=0)), axis=0)
data_3 = np.mean(np.sqrt(np.mean(errorPF_Systematic[[1, 3], :, :]**2, axis=0)), axis=0)
data_4 = np.mean(np.sqrt(np.mean(errorPF_Residual[[1, 3], :, :]**2, axis=0)), axis=0)
data_5 = np.mean(np.sqrt(np.mean(errorPF_Strat[[1, 3], :, :]**2, axis=0)), axis=0)
data_6 = np.mean(np.sqrt(np.mean(errorUKF[[1, 3], :, :]**2, axis=0)), axis=0)
data = [data_1, data_2, data_3, data_4, data_5, data_6]
bp     =  axs[1].boxplot(data,patch_artist = True,
                    boxprops = dict(facecolor = translucent_blue,color = cb_colors['neutral'],linewidth = 2),
                    medianprops = dict(color = cb_colors['upper'],linewidth = 2))
axs[1].minorticks_on()
axs[1].tick_params(which='both', width=2)
axs[1].tick_params(which='major', length=7)
axs[1].tick_params(which='minor', length=4)
axs[1].set_xticks([1, 2, 3, 4, 5, 6])
axs[1].set_xticklabels(['GMF', 'PMF', 'Sys', 'Res', 'Strat','UKF'])
axs[1].set_ylabel(r'\textbf{RMSE} Velocity (m/s)')
axs[1].set_ylim([0, 2])

# SNEES Position
data_1 =  np.median(neesGMF,axis = 1)[0]/nS
data_2 =  np.median(neesPMF,axis = 1)[0]/nS
data_3 =  np.median(neesPF_Systematic,axis = 1)[0]/nS
data_4 =  np.median(neesPF_Residual,axis = 1)[0]/nS
data_5 =  np.median(neesPF_Strat,axis = 1)[0]/nS
data_6 =  np.median(neesPF_Strat,axis = 1)[0]/nS
data = [data_1, data_2, data_3, data_4, data_5, data_6]
bp     =  axs[2].boxplot(data,patch_artist = True,
                    boxprops = dict(facecolor = translucent_blue,color = cb_colors['neutral'],linewidth = 2),
                    medianprops = dict(color = cb_colors['upper'],linewidth = 2))
axs[2].minorticks_on()
axs[2].tick_params(which='both', width=2)
axs[2].tick_params(which='major', length=7)
axs[2].tick_params(which='minor', length=4)
axs[2].set_xticks([1, 2, 3, 4, 5, 6])
axs[2].set_xticklabels(['GMF', 'PMF', 'Sys', 'Res', 'Strat','UKF'])
axs[2].set_ylabel(r'\textbf{SNEES} Position')
axs[2].set_ylim([0, 3])

fig.savefig("STATS_LOW_FLAT.pdf", format='pdf', dpi=1000, bbox_inches='tight')

# Calculate the mean times for each filter method
mean_time_GMF = np.mean(end_time_GMF - start_time_GMF)/nTime
mean_time_PMF = np.mean(end_time_PMF - start_time_PMF)/nTime
mean_time_Strat = np.mean(end_time_Strat - start_time_Strat)/nTime
mean_time_Residual = np.mean(end_time_Residual - start_time_Residual)/nTime
mean_time_Systematic = np.mean(end_time_Systematic - start_time_Systematic)/nTime
mean_time_UKF = np.mean(end_time_UKF - start_time_UKF)/nTime

# Create a list of the mean times and filter labels
mean_times = [mean_time_GMF, mean_time_PMF, mean_time_Strat, mean_time_Residual, mean_time_Systematic, mean_time_UKF]
filters = ['GMF', 'PMF', 'Sys', 'Res', 'Strat','UKF']

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(filters, mean_times, color='skyblue', edgecolor='black')

# Set labels and title
ax.set_xlabel('Filter Method', fontsize=14)
ax.set_ylabel('Mean Time (s)', fontsize=14)
ax.set_title('Mean Time for Each Filter Method', fontsize=16)

fig.savefig("TIME_LOW_FLAT.pdf", format='pdf', dpi=1000, bbox_inches='tight')

# Show the plot
plt.tight_layout()
plt.show()
