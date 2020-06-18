#!/usr/bin/env python
# coding: utf-8

"""
UAV Tracking Tutorial
===========================================
"""

# %%
# Overview
# --------
# Starting with GPS data from an instrumented UAV, we will generate range, bearing, and
# elevation measurements (from a given radar position). We will use Stone Soup's simple
# :class:`~.SingleTargetTracker` to perform the tracking. At this point we are primarly interested
# in the necessary motion models that may be needed so the example is fairly simple, but
# we want to be able to easily expand the simulation to handle more complex scenarios.
#
# Items to note:
#
# - Assumes a single target track, which simplifies track management
# - There is no clutter, and no missed detections -> No Data Association
# - Need an initiator and deleter for the
# - GPS updates are 1 sec., we assumes radar revisit is the same (little unrealistic)
#
# We are assuming a ground based radar:
#
# - No need to use a platform or mount sensors on it.
# - Radar has course elevation resolution and fine bearing resolution.
# - Use range standard deviation of 3.14 m as a replacement for range resolution.



# %%
# Setup: transition model, measurement model, updater and predictor
# -----------------------------------------------------------------

import numpy as np
from stonesoup.models.transition.linear import (
    ConstantVelocity,
    CombinedLinearGaussianTransitionModel
    )
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.models.measurement.nonlinear import (
    CartesianToElevationBearingRange
    )
from stonesoup.types.array import CovarianceMatrix


transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(1.0),
     ConstantVelocity(1.0),
     ConstantVelocity(1.0)])

# Model coords = elev, bearing, range. Angles in radians
meas_covar = np.diag([np.radians(np.sqrt(10.0))**2,
                      np.radians(0.6)**2,
                      3.14**2])

meas_covar_trk = CovarianceMatrix(1.0*meas_covar)
meas_model = CartesianToElevationBearingRange(
            ndim_state=6,
            mapping=np.array([0, 2, 4]),
            noise_covar=meas_covar_trk)
predictor = ExtendedKalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model=meas_model)

# %%
# Define Function
# ---------------
# Reads in our pickle file, which contains a pandas dataframe
import pandas as pd
import pickle as pkl

def readPkl(fdir, fname):
    # Reads pkl data file and changes column headings as appropriate
    data = pd.read_pickle(fdir+fname)
    data.rename(columns={'raw_lon' : 'longitude',
                         'raw_lat' : 'latitude',
                         'alt' : 'altitude (m)'},
                inplace = True)
    data['Vx m/s'] = (np.sin(np.deg2rad(data['heading'])) *
                      data['speed'])
    data['Vy m/s'] = (np.cos(np.deg2rad(data['heading'])) *
                      data['speed'])
    return data

# %%
# Define Detector 
# ---------------
# Detector reads in the ground truth, generates range, bearing
# and elevation measurements and adds noise to these based on the measurement
# covariance. More involved detector could:
#
# - Add clutter
# - Handle :math:`P_d` -> Could be based on radial velocity
# - Handle radar revisit times
# - Add unknown & multiple targets

#import datetime
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader import DetectionReader
from stonesoup.types.detection import Detection
from numpy.random import multivariate_normal
import pymap3d as pm
from stonesoup.types.angle import Bearing, Elevation

class Detector(DetectionReader):
        def init_vals(self, df, C0, obs_pt=(50.297311666, -110.945686, 0)):
            '''Initializes variables inside the Detector.

            Parameters
            ----------
            df : Pandas dataframe.
                Data frame with lat/long/altitude info.
            C0 : numpy.array
                Covariance matrix for noisy measurements.
            obs_pt : TYPE, optional
                DESCRIPTION. The default is (50.297311666, -110.945686, 0).

            Returns
            -------
            None.

            '''
            self.df=df
            self.obs_pt = obs_pt
            truth, measurements, aer = self.genRBE(C0)
            self.truth = truth
            self.measurements = measurements
            self.aer = aer
        
        def genRBW():
            '''
            

            Returns
            -------
            None.

            '''
        def genRBE(self, C0):
            ''' Generates Range, Bearing and Elevation data based on the 
            observation point. Noisy measurement are generated based on
            the covariance matrix passed in.
            
            Input
            -----
            C0 = Covariance matrix for noisy measurements
            
            Returns
            -------
            truth - list of truth range (m), bearing (rad) measurements
            meas_pol_list - list polar measurements: range (m), bearing (rad)
            aer = list of (azimuth (deg), elevation (deg), range (m)) tuples.
            '''
        
            alt = self.df['altitude (m)'].tolist()
        
            longs = self.df['longitude'].tolist()
            lats = self.df['latitude'].tolist()
            timestamp = self.df.index
        
            lla = list(zip(lats, longs, alt))
            aer=[]
            
            for x in lla:
                a, e, r = pm.geodetic2aer(*x, *self.obs_pt)
                # Convert heading from North to StoneSoup Polar heading
                # i.e. angle off horiz x axis.
                a = 90.-a
                if a > 180.0:
                    a = a - 360
                elif a < -180:
                    a = a + 360
                aer.append((a, e, r))
            aer2 = list(zip(*aer))
        
            measurements = []
            truth = []
            Nsamp = len(self.df)
            noise = multivariate_normal([0]*C0.shape[0], C0, Nsamp)
            for count, item in enumerate(aer):
        
                meas = np.array([[Elevation(np.deg2rad(item[1])),
                                  Bearing(np.deg2rad(item[0])),
                                  float(item[2])]]).T
                noisy_meas = meas + noise[[count],:].T
                truth.append(Detection(meas, timestamp=timestamp[count]))
                measurements.append(Detection(noisy_meas,
                                              timestamp=timestamp[count]))
        
            return truth, measurements, aer

        @BufferedGenerator.generator_method
        def detections_gen(self):
            for meas in self.measurements:
                yield meas.timestamp, {meas}

# %%
# Create Detector
# -----------------
# Create the detector and initialize it.

#import UAV_Tut_Aux as uta

fdir = ".\\"
fname = "UAV_Rot.pkl"
df = readPkl(fdir, fname)
obs_pos = (50.297311666, -30.948, 0) # Radar position
detector = Detector()
detector.init_vals(df, meas_covar, obs_pt=obs_pos)

# %%
# Setup Initiator and Deletor classes for the Tracker
# ---------------------------------------------------
# This is just an heuristic initiation:
# Assume most of the deviation is caused by the Bearing measurement error.
# This is then converted into x, y components using the target bearing. For z,
# we simply use range*elev_std_dev (ignore any bearing or range related components).
# Velocity covariances are just based on expected velocity range of targets.
#
# **NOTE** - The Extended Kalman filter can be very sensitive to the state
# initiation. I tried using the default :class:`~.SimpleMeasurementInitiator` but it tended
# diverge over the course of the track, especially when large bearing measurement
# covariances.
from stonesoup.types.state import GaussianState, State
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis

class Initiator(SimpleMeasurementInitiator):
    def initiate(self, detections, **kwargs):
        MAX_DEV = 400.
        tracks = set()
        measurement_model = self.measurement_model
        for detection in detections:
            state_vector = measurement_model.inverse_function(
                            detection)
            model_covar = measurement_model.covar()
            
            el_az_range = np.sqrt(np.diag(model_covar)) #elev, az, range
            
            std_pos = detection.state_vector[2, 0]*el_az_range[1]
            stdx = np.abs(std_pos*el_az_range[1]*np.sin(el_az_range[1]))
            stdy = np.abs(std_pos*el_az_range[1]*np.cos(el_az_range[1]))
            stdx = np.abs(std_pos*np.sin(el_az_range[1]))
            stdy = np.abs(std_pos*np.cos(el_az_range[1]))
            stdz = np.abs(detection.state_vector[2, 0]*el_az_range[0])
            if stdx > MAX_DEV:
                print('Warning - X Deviation exceeds limit!!')
            if stdy > MAX_DEV:
                print('Warning - Y Deviation exceeds limit!!')  
            if stdz > MAX_DEV:
                print('Warning - Z Deviation exceeds limit!!')
            C0 = np.diag(np.array([stdx, 30.0, stdy, 30.0, stdz, 30.0])**2)
        
            tracks.add(Track([GaussianStateUpdate(
                state_vector,
                C0,
                SingleHypothesis(None, detection),
                timestamp=detection.timestamp)
            ]))
        return tracks
            
   
prior_state = GaussianState(
        np.array([[0], [0], [0], [0], [0], [0]]),
        np.diag([0, 30.0, 0, 30.0, 0, 30.0])**2)
initiator = Initiator(prior_state, meas_model)
#initiator = SimpleMeasurementInitiator(prior_state, meas_model)

class MyDeleter:
    def delete_tracks(self, tracks):
        return {}
 
deleter = MyDeleter()

# %%
# Setup Hypothesiser and Associator
# ---------------------------------
# Since we know there is only one measurement per scan, we can just use the
# :class:`~.NearestNeigbour` associator to achieve our desired result.
from stonesoup.measures import Euclidean
from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.tracker.simple import SingleTargetTracker
meas = Euclidean()
hypothesiser = DistanceHypothesiser(predictor, updater, meas)
associator = NearestNeighbour(hypothesiser)


tracker = SingleTargetTracker(initiator,
                              deleter,
                              detector,
                              associator,
                              updater)

# %%
# Run the Tracker
# ---------------------------------
# We extract the ground truth from the detector and then run the tracker.
# While running the tracker we:
#
# - Extract the measurement that is associated with it.
# - Extract the position components of the estimated state vector.
#
# This allows us to plot the measurements, ground truth, and state estimates.
#
# **Note:** The meas_model.inverse_function() returns a state vector, which
# for our CV model consists of [x, vx, y, vy, z, vz].
from matplotlib import pyplot as plt
est_X=[]
est_Y=[]
meas_X=[]
meas_Y=[]
true_X = []
true_Y = []
Range=[]
Bearing=[]
times = []
MRange=[]
MBearing=[]
track_list = []
for item in detector.truth:
    xyz = meas_model.inverse_function(State(item.state_vector,item.timestamp))
    
    true_X.append(xyz[0])
    true_Y.append(xyz[2])
for time, tracks in tracker:
    times.append(time)
    # Because this is a single target tracker, I know there is only 1 track.
    track_list.append(list(tracks)[0])
    for track in tracks:

        #Get the corresponding measurement
        Tmeas = track.states[-1].hypothesis.measurement.state_vector
        MRange.append(Tmeas[2])
        MBearing.append(Tmeas[1])
        # Convert measurement into xy
        xyz = meas_model.inverse_function(State(Tmeas,time))
        meas_X.append(xyz[0])
        meas_Y.append(xyz[2])
        
        vec = track.states[-1].state_vector
        est_X.append(vec[0])
        est_Y.append(vec[2])
        sv = State(track.states[-1].state_vector, time)
        tmp = meas_model.function(sv)
        Range.append(tmp[2])
        Bearing.append(tmp[1])

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(1, 1, 1)
plt.plot(meas_X, meas_Y, 'xb', label='Measurements')
ax1.plot(true_X, true_Y, 'd-k', label='Truth', markerfacecolor='None')
ax1.legend()
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')

fig = plt.figure(figsize=(10, 6))
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(true_X, true_Y, 'd-k', label='Truth', markerfacecolor='None')
ax2.plot(est_X, est_Y, 'r.', label='Estimates')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.legend()