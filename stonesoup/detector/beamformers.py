# -*- coding: utf-8 -*-
"""Beamforming algorithms for direction of arrival estimation.

These algorithms take sensor data measured by an array with specified geometry and return bearings
in the form of target detections.

The data from the array is passed to the algorithms in a csv file, with each column storing a time
series from a single sensor.

The relative sensor locations is passed to each algorithm in a separate csv file, which lists the
lists the Cartesian coordinates for each time window.

"""

import csv
import copy
import math
from scipy.stats import norm, uniform
from itertools import islice
from pathlib import Path
from datetime import datetime, timedelta
from itertools import zip_longest
import numpy as np
from typing import Sequence
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..models.measurement.linear import LinearGaussian
from ..types.array import StateVector, CovarianceMatrix
from ..types.detection import Detection
from ..types.angle import Elevation, Bearing
from ..reader import DetectionReader


def grouper(iterable, n):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=None)


class CaponBeamformer(DetectionReader):
    """An adaptive beamformer method designed to reduce the influence of side lobes in the case of
    multiple signals with different directions of arrival. The beamformer uses the Capon algorithm.

    J. Capon, High-Resolution Frequency-Wavenumber Spectrum Analysis, Proc. IEEE 57(8):1408-1418
    (1969)

    """

    path: Path = Property(doc='The path to the csv file containing the raw data')
    fs: float = Property(doc='Sampling frequency (Hz)')
    loc_path: Path = Property(doc='The path to the csv file containing the Cartesian coordinates\
                              of the sensors in the format "X1, Y1, Z1\n X2, Y2, Z2\n...."')
    num_sensors: int = Property(doc='Number of sensors')
    omega: float = Property(doc='Signal frequency (Hz)')
    wave_speed: float = Property(doc='Speed of wave in the medium')
    window_size: int = Property(doc='Window size', default=750)
    start_time: datetime = Property(doc='Time first sample was recorded', default=datetime.now())

    def __init__(self, path, *args, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)
        super().__init__(path, *args, **kwargs)

    @BufferedGenerator.generator_method
    def detections_gen(self):
        with self.path.open(newline='') as csv_file, open(self.loc_path, newline='') as loc_file:
            thetavals = np.linspace(0, 2*np.pi, num=400)
            phivals = np.linspace(0, np.pi/2, num=100)

            # Use a csv reader to read the files
            reader = csv.reader(csv_file, delimiter=',')
            reader_loc = csv.reader(loc_file, delimiter=',')

            current_time = self.start_time

            # Use itertools to grab windows (in case of streaming data)
            windows = grouper(reader, self.window_size)
            sensor_loc = grouper(reader_loc, self.num_sensors)

            for (window, sensor_pos) in zip(windows, sensor_loc):
                # Grab the next `window_size` lines from the reader and read it into y (also
                # convert to float)
                y = np.array(window).astype(float)
                # spatial locations of hydrophones
                z = np.reshape(np.asarray(sensor_pos, dtype=float), [self.num_sensors, 3])

                c = self.wave_speed/(2*self.omega*np.pi)

                # calculate covariance estimate
                R = y.T @ y
                R_inv = np.linalg.inv(R)

                maxF = 0
                maxtheta = 0

                for theta in thetavals:
                    for phi in phivals:
                        # directional unit vector
                        # convert from spherical polar coordinates to cartesian
                        a = np.array([np.cos(theta) * np.sin(phi),
                                      np.sin(theta) * np.sin(phi),
                                      np.cos(phi)])
                        a /= np.sqrt(np.sum(a ** 2))

                        phases = np.sum(a * z, 1) / c

                        # steering vector
                        v = np.cos(phases) - np.sin(phases) * 1j

                        F = 1 / ((self.window_size - self.num_sensors) * np.conj(v).T @ R_inv @ v)
                        if F > maxF:
                            maxF = F
                            maxtheta = theta
                            maxphi = phi

                # Defining a detection
                state_vector = StateVector([maxtheta, maxphi])  # [Azimuth, Elevation]
                covar = CovarianceMatrix(np.array([[1, 0], [0, 1]]))
                measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                                   noise_covar=covar)
                current_time = current_time + timedelta(milliseconds=1000*self.window_size/self.fs)
                detection = Detection(state_vector, timestamp=current_time,
                                      measurement_model=measurement_model)
                yield current_time, {detection}


class RJMCMCBeamformer(DetectionReader):
    """A parameter estimation algorithm for a sensor array measuring passive signals. Given the
    input signals from the array, the algorithm uses reversible-jump Markov chain Monte Carlo [1]
    to sample from the posterior probability for a model where the number of targets and directions
    of arrival are the unknown parameters. The algorithm is based on the work in [2] and [3].

    [1] P. J. Green, Reversible jump Markov chain Monte Carlo computation and Bayesian
    model determination, Biometrika 82(4):711-732 (1995)
    [2] C. Andrieu and A. Doucet, Joint Bayesian Model Selection and Estimation of
    Noisy Sinusoids via Reversible Jump MCMC, IEEE Trans. Signal Process. 47(10):2667-2676 (1999)
    [3] C. Andrieu, N. de Freitas and A. Doucet, Robust Full Bayesian Learning for Radial Basis
    Networks, Neural Computation 13:2359–2407 (2001)

    """

    path: str = Property(doc='The path to the csv file, containing the raw data')
    fs: float = Property(doc='Sampling frequency (Hz)')
    loc_path: Path = Property(doc='The path to the csv file containing the Cartesian coordinates\
                              of the sensors in the format "X1, Y1, Z1\n X2, Y2, Z2\n...."')
    num_sensors: int = Property(doc='Number of sensors')
    omega: float = Property(doc='Signal frequency (Hz)')
    wave_speed: float = Property(doc='Speed of wave in the medium')
    max_targets: int = Property(default=5, doc='Maximum number of targets')
    seed: int = Property(doc='Random number generator seed for reproducible output. Set to 0 for\
                         non-reproducible output')
    window_size: int = Property(doc='Window size', default=750)
    num_samps: int = Property(default=20000, doc='Number of samples generated by MCMC\
                              algorithm (higher samples = more accurate estimates)')
    Lambda: int = Property(default=1, doc='Expected number of targets')
    nbins: int = Property(default=128, doc='Number of bins used in histogram of samples over\
                          azimuth and elevation (configurable but should generally scale with\
                          the number of samples)')
    start_time: datetime = Property(doc='Time first sample was recorded', default=datetime.now())

    def __init__(self, path, *args, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)
        super().__init__(path, *args, **kwargs)
        if self.seed != 0:
            self.random_state = np.random.RandomState(self.seed)

    @BufferedGenerator.generator_method
    def detections_gen(self):
        with self.path.open(newline='') as csv_file, open(self.loc_path, newline='') as loc_file:
            csv_file.seek(0)  # Reset file read position

            # Use a csv reader to read the file
            reader = csv.reader(csv_file, delimiter=',')
            reader_loc = csv.reader(loc_file, delimiter=',')

            current_time = self.start_time

            # Use itertools to grab windows (in case of streaming data)
            windows = grouper(reader, self.window_size)
            sensor_loc = grouper(reader_loc, self.num_sensors)

            bin_steps = [math.pi/(2*self.nbins), 2*math.pi/self.nbins]

            for (window, sensor_pos) in zip(windows, sensor_loc):
                # Grab the next `window_size` lines from the reader and read it into y (also
                # convert to float)
                y = np.array([row for row in islice(reader, self.window_size)]).astype(float)

                self.sensor_pos = np.reshape(np.asarray(sensor_pos, dtype=float),
                                             [self.num_sensors, 3])

                N = self.num_sensors*self.window_size

                # initialise histograms
                param_hist = np.zeros([self.max_targets, self.nbins, self.nbins])
                order_hist = np.zeros([self.max_targets])

                # initialise params
                angle_params = []
                for n in range(0, self.max_targets):
                    angle_params.append(StateVector([Elevation(0), Bearing(0)]))
                p_params: Sequence[StateVector] = angle_params
                [params, K] = self.proposal([], 0, p_params)

                # calculate sinTy and cosTy
                sinTy = np.zeros([self.num_sensors])
                cosTy = np.zeros([self.num_sensors])

                yTy = 0

                for k in range(0, self.num_sensors):
                    for t in range(0, self.window_size):
                        sinTy[k] = sinTy[k] + math.sin(2*math.pi*t*self.omega/self.fs)*y[t, k]
                        cosTy[k] = cosTy[k] + math.cos(2*math.pi*t*self.omega/self.fs)*y[t, k]
                        yTy = yTy + y[t, k]*y[t, k]

                sumsinsq = 0
                sumcossq = 0
                sumsincos = 0

                for t in range(0, self.window_size):
                    sumsinsq = sumsinsq \
                        + math.sin(2*math.pi*t*self.omega/self.fs) \
                        * math.sin(2*math.pi*t*self.omega/self.fs)
                    sumcossq = sumcossq \
                        + math.cos(2*math.pi*t*self.omega/self.fs) \
                        * math.cos(2*math.pi*t*self.omega/self.fs)
                    sumsincos = sumsincos \
                        + math.sin(2*math.pi*t*self.omega/self.fs) \
                        * math.cos(2*math.pi*t*self.omega/self.fs)
                sumsincos = 0
                old_logp = self.log_prob(params, K, sinTy, cosTy, yTy,
                                         sumsinsq, sumcossq, sumsincos, N)
                n = 0

                while n < self.num_samps:
                    [p_params, p_K, Qratio] = self.proposal_func(params, K, p_params,
                                                                 self.max_targets)
                    if p_K != 0:
                        new_logp = self.log_prob(p_params, p_K, sinTy, cosTy, yTy,
                                                 sumsinsq, sumcossq, sumsincos, N)
                        logA = new_logp - old_logp + np.log(Qratio)

                        # do a Metropolis-Hastings step
                        if logA > np.log(uniform.rvs(random_state=self.random_state)):
                            old_logp = new_logp
                            params = copy.deepcopy(p_params)
                            K = copy.deepcopy(p_K)
                            for k in range(0, K):
                                # correct for mirrored DOAs in elevation
                                if ((params[k][0] > math.pi/2) & (params[k][0] <= math.pi)):
                                    params[k][0] = math.pi - params[k][0]
                                elif ((params[k][0] > math.pi) & (params[k][0] <= 3*math.pi/2)):
                                    params[k][0] = params[k][0] - math.pi
                                    params[k][1] = params[k][1] - math.pi
                                elif ((params[k][0] > 3*math.pi/2) & (params[k][0] <= 2*math.pi)):
                                    params[k][0] = 2*math.pi - params[k][0]
                                    params[k][1] = params[k][1] - math.pi
                                if (params[k][1] < 0):
                                    params[k][1] += 2*math.pi
                                elif (params[k][1] > 2*math.pi):
                                    params[k][1] -= 2*math.pi
                        for k in range(0, K):
                            bin_ind = [0, 0]
                            for ind in range(0, 2):
                                edge = bin_steps[ind]
                                while edge < params[k][ind]:
                                    edge += bin_steps[ind]
                                    bin_ind[ind] += 1
                                    if bin_ind[ind] == self.nbins-1:
                                        break
                            param_hist[K-1, bin_ind[0], bin_ind[1]] += 1
                            order_hist[K-1] += 1
                        n += 1

                # look for peaks in histograms
                max_peak = 0
                max_ind = 0
                for ind in range(0, self.max_targets):
                    if order_hist[ind] > max_peak:
                        max_peak = order_hist[ind]
                        max_ind = ind

                # look for largest N peaks, where N corresponds to peak in the order histogram
                # use divide-and-conquer quadrant-based approach
                if max_ind == 0:
                    # only one target
                    [unique_peak_inds1, unique_peak_inds2] = np.unravel_index(
                                                             param_hist[0, :, :].argmax(),
                                                             param_hist[0, :, :].shape)
                else:
                    # multiple targets
                    order_ind = max_ind - 1
                    quadrant_factor = 2
                    nstart = 0
                    mstart = 0
                    nend = quadrant_factor
                    mend = quadrant_factor
                    peak_inds1 = [None] * 16
                    peak_inds2 = [None] * 16
                    k = 0
                    while quadrant_factor < 32:
                        max_quadrant = 0
                        quadrant_size = self.nbins/quadrant_factor
                        for n in range(nstart, nend):
                            for m in range(mstart, mend):
                                [ind1, ind2] = np.unravel_index(
                                               param_hist[order_ind,
                                                          int(n*quadrant_size):
                                                          int((n+1)*quadrant_size-1),
                                                          int(m*quadrant_size):
                                                          int((m+1)*quadrant_size-1)].argmax(),
                                               param_hist[order_ind, int(n*quadrant_size):
                                                          int((n+1)*quadrant_size-1),
                                                          int(m*quadrant_size):
                                                          int((m+1)*quadrant_size-1)].shape)
                                peak_inds1[k] = int(ind1 + n*quadrant_size)
                                peak_inds2[k] = int(ind2 + m*quadrant_size)
                                if param_hist[order_ind, peak_inds1[k], peak_inds2[k]]\
                                   > max_quadrant:
                                    max_quadrant = param_hist[order_ind, peak_inds1[k],
                                                              peak_inds2[k]]
                                    max_ind1 = n
                                    max_ind2 = m
                                k += 1
                        quadrant_factor = 2*quadrant_factor
                        # on next loop look for other peaks in the quadrant containing the highest
                        # peak
                        nstart = 2*max_ind1
                        mstart = 2*max_ind2
                        nend = 2*(max_ind1+1)
                        mend = 2*(max_ind2+1)

                    # determine unique peaks
                    unique_peak_inds1 = [None] * 16
                    unique_peak_inds2 = [None] * 16
                    unique_peak_inds1[0] = peak_inds1[0]
                    unique_peak_inds2[0] = peak_inds2[0]
                    num_peaks = 1
                    for n in range(0, 16):
                        flag_unique = 1
                        for k in range(0, num_peaks):
                            # check if peak is close to any other known peaks
                            if (unique_peak_inds1[k] - peak_inds1[n]) < 2:
                                if (unique_peak_inds2[k] - peak_inds2[n]) < 2:
                                    # part of same peak (check if bin is taller)
                                    if (param_hist[order_ind, peak_inds1[n], peak_inds2[n]]
                                       > param_hist[order_ind, unique_peak_inds1[k],
                                                    unique_peak_inds2[k]]):
                                        unique_peak_inds1 = peak_inds1[n]
                                        unique_peak_inds2 = peak_inds2[n]
                                    flag_unique = 0
                                    break
                        if flag_unique == 1:
                            unique_peak_inds1[num_peaks] = peak_inds1[n]
                            unique_peak_inds2[num_peaks] = peak_inds2[n]
                            num_peaks += 1

                # Defining a detection [Azimuth, Elevation]
                state_vector = StateVector([unique_peak_inds2*bin_steps[1],
                                            unique_peak_inds1*bin_steps[0]])
                covar = CovarianceMatrix(np.array([[1, 0], [0, 1]]))
                measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                                   noise_covar=covar)
                current_time = current_time + timedelta(milliseconds=1000*self.window_size/self.fs)
                detection = Detection(state_vector, timestamp=current_time,
                                      measurement_model=measurement_model)

                yield current_time, {detection}

    def log_prob(self, p_params, p_K, sinTy, cosTy, yTy, sumsinsq, sumcossq,
                 sumsincos, N):
        """Calculates the log probability of the unnormalised posterior distribution for a given
        set of parameters and data.

        Parameters
        ==========
        p_params : :class:`Sequence[StateVector]`
            A sequence of the proposed bearings and elevations for each target.
        p_K : int
            The proposed number of targets.
        sinTy: numpy.ndarray
            Array of dot products between the sine wave at the given frequency and the sensor data.
        cosTy: numpy.ndarray
            Array of dot products between the cosine wave at the given frequency and the sensor
            data.
        yTy: numpy.ndarray
            Array of dot products between the data for each sensor.
        sumsinsq: float
            Sum over the square of the sine wave at the given frequency.
        sumcossq: float
            Sum over the square of the cosine wave at the given frequency.
        sumsincos: float
            Sum over the sine wave multiplied by the cos wave at the given frequency.
        N: int
            Number of data points summed over all of the sensors.

        Returns
        =======
        : float
            The value of the log joint probability distribution for the given parameters.

        """

        DTy = np.zeros(p_K)
        DTD = np.zeros((p_K, p_K))
        sinalpha = np.zeros((p_K, self.num_sensors))
        cosalpha = np.zeros((p_K, self.num_sensors))

        for k in range(0, p_K):
            # calculate phase offsets relative to first sensor in the array
            for sensor_ind in range(0, self.num_sensors):
                alpha = 2*math.pi*self.omega*((self.sensor_pos[sensor_ind, 1]
                                               - self.sensor_pos[0, 1]) * math.sin(p_params[k][1])
                                              * math.sin(p_params[k][0])
                                              + (self.sensor_pos[sensor_ind, 0]
                                                 - self.sensor_pos[0, 0])
                                              * math.cos(p_params[k][1])
                                              * math.sin(p_params[k][0])
                                              + (self.sensor_pos[sensor_ind, 2]
                                                 - self.sensor_pos[0, 2])
                                              * math.sin(p_params[k][0])) / self.wave_speed
                DTy[k] = DTy[k] + math.cos(alpha) * sinTy[sensor_ind] \
                    + math.sin(alpha) * cosTy[sensor_ind]
                sinalpha[k, sensor_ind] = math.sin(alpha)
                cosalpha[k, sensor_ind] = math.cos(alpha)

        for k1 in range(0, p_K):
            DTD[k1, k1] = N/2

        if (p_K > 1):
            for sensor_ind in range(0, 9):
                for k1 in range(0, p_K):
                    for k2 in range(k1+1, p_K):
                        DTD[k1, k2] = DTD[k1, k2] \
                            + cosalpha[k1, sensor_ind] * cosalpha[k2, sensor_ind] * sumsinsq \
                            + (cosalpha[k1, sensor_ind] * sinalpha[k2, sensor_ind]
                               + cosalpha[k2, sensor_ind] * sinalpha[k1, sensor_ind]) * sumsincos \
                            + sinalpha[k1, sensor_ind]*sinalpha[k2, sensor_ind] * sumcossq
                        DTD[k2, k1] = DTD[k1, k2]

        Dterm = np.matmul(np.linalg.solve(1001*DTD, DTy), np.transpose(DTy))
        log_posterior = - (p_K * np.log(1.001) / 2) - (N / 2) * np.log((yTy - Dterm) / 2) \
            + p_K * np.log(self.Lambda) - np.log(np.math.factorial(p_K)) \
            - p_K*np.log(math.pi * math.pi)
        # note: math.pi*math.pi comes from area of parameter space in one dimension (i.e. range of
        # azimuth * range of elevation)

        return log_posterior

    def elevation_mode_coin_toss(self, p_params, p_K):
        """The state space is reflected across different elevation angle sectors resulting in
        duplicate modes across the posterior probability distribution. This function uses a coin
        toss to jump to another mode to ensure full and unbiased exploration of the Markov chain.

        Parameters
        ==========
        p_params : :class:`Sequence[StateVector]`
            A sequence of the current bearings and elevations for each target.
        p_K : int
            Current number of targets.

        Returns
        =======
        : :class:`Sequence[StateVector]`
            A sequence of the bearings and elevations for each target.

        """

        for k in range(0, p_K):
            # transform to first mode
            if p_params[k][0] > 3*math.pi/2:
                p_params[k][0] = 2*math.pi - p_params[k][0]
                p_params[k][1] = p_params[k][1] - math.pi
            elif p_params[k][0] > math.pi:
                p_params[k][0] = p_params[k][0] - math.pi
                p_params[k][1] = p_params[k][1] - math.pi
            elif p_params[k][0] > math.pi/2:
                p_params[k][0] = math.pi - p_params[k][0]

            # do coin toss to decide mode to jump to
            toss = uniform.rvs(random_state=self.random_state)
            if toss < 0.25:
                # first mode
                pass
            elif toss < 0.5:
                # second mode
                p_params[k][0] = math.pi - p_params[k][0]
            elif toss < 0.75:
                # third mode
                p_params[k][0] = p_params[k][0] + math.pi
                p_params[k][1] = p_params[k][1] + math.pi
            else:
                # fourth mode
                p_params[k][0] = 2*math.pi - p_params[k][0]
                p_params[k][1] = p_params[k][1] + math.pi

            # wrap angles
            if p_params[k][0] < 0:
                p_params[k][0] = p_params[k][0] + 2*math.pi
            elif p_params[k][0] > 2*math.pi:
                p_params[k][0] = p_params[k][0] - 2*math.pi
            if p_params[k][1] < 0:
                p_params[k][1] = p_params[k][1] + 2*math.pi
            elif p_params[k][1] > 2*math.pi:
                p_params[k][1] = p_params[k][1] - 2*math.pi

        return p_params

    def proposal(self, params, K, p_params):
        """The proposal function for direction of arrival parameters. The parameters are initially
        sampled from a uniform distribution. Gaussian probability densities are used to make new
        proposals centred on the current parameter values.

        Parameters
        ==========
        params : :class:`Sequence[StateVector]`
            A sequence of the current bearings and elevations for each target.
        K : int
            Current number of targets.
        p_params: :class:`Sequence[StateVector]`
            A sequence to store the proposed bearings and elevations for each target.

        Returns
        =======
        : :class:`Sequence[StateVector]`
            A sequence of the proposed bearings and elevations for each target.
        : int
            The proposed number of targets.

        """

        p_K = 0
        # choose random phase (assuming constant frequency)
        if len(params) == 0:
            p_params[0] = StateVector([2*math.pi*uniform.rvs(random_state=self.random_state),
                                      2*math.pi*uniform.rvs(random_state=self.random_state)])
            p_K = 1
        else:
            for k in range(0, K):
                epsilon = norm.rvs(0, 0.125, 1, random_state=self.random_state)
                rand_val = params[k][0]+epsilon
                if rand_val > 2*math.pi:
                    rand_val = rand_val-2*math.pi
                elif rand_val < 0:
                    rand_val = rand_val+2*math.pi
                p_params[k][0] = rand_val
                epsilon = norm.rvs(0, 0.5, 1, random_state=self.random_state)
                rand_val = params[k][1]+epsilon
                if rand_val > 2*math.pi:
                    rand_val = rand_val-2*math.pi
                elif rand_val < 0:
                    rand_val = rand_val+2*math.pi
                p_params[k][1] = rand_val
            p_K = copy.deepcopy(K)
        return p_params, p_K

    def proposal_func(self, params, K, p_params, max_targets):
        """The proposal function for the RJMCMC moves. A coin toss is used to decide whether an
        update move or revesible-jump move is proposed. If a reversible-jump move is chosen, a
        further coin toss decides whether the proposed move is a birth or death move (adding or
        removing a target).

        Parameters
        ==========
        params : :class:`Sequence[StateVector]`
            A sequence of the current bearings and elevations for each target.
        K : int
            Current number of targets.
        p_params: :class:`Sequence[StateVector]`
            A sequence to store the proposed bearings and elevations for each target.
        max_targets: int
            Maximum number of targets.

        Returns
        =======
        : :class:`Sequence[StateVector]`
            A sequence of the proposed bearings and elevations for each target.
        : int
            The proposed number of targets.
        : float
            Ratio of proposal probabilities for forwards and backwards moves.

        """

        # update_type = uniform.rvs(random_state=self.random_state)
        Qratio = 1.0  # ratio of proposal probabilities for forwards and backwards moves
        update_type = 1  # forced temporarily (for single-target examples)
        if update_type > 0.5:
            # update params
            [p_params, p_K] = self.proposal(params, K, p_params)
            p_params = self.elevation_mode_coin_toss(p_params, p_K)
        else:
            # birth / death move
            update_bd = uniform.rvs(random_state=self.random_state)
            if update_bd > 0.5:
                # birth move
                if K < max_targets:
                    if K == 1:
                        Qratio = 0.5  # death moves not possible for K=1
                    if K == max_targets-1:
                        Qratio = 2.0  # birth moves not possible for K=max_targets
                    [p_temp, K_temp] = self.proposal([], 1, p_params)
                    p_params = copy.deepcopy(params)
                    p_params[K] = p_temp[0]
                    p_K = K + 1
            else:
                # death move
                if K > 1:
                    if K == max_targets:
                        Qratio = 0.5  # birth moves not possible for K=max_targets
                    if K == 2:
                        Qratio = 2.0  # death moves not possible for K=1
                    death_select = int(np.ceil(K*uniform.rvs(random_state=self.random_state)))
                    if death_select > 1:
                        if death_select < K:
                            if death_select == 2:
                                p_params[0] = params[0]
                                p_params[1:-1] = params[2:]
                            else:
                                p_params[0:death_select-2] = params[0:death_select-2]
                                p_params[death_select-1:-1] = params[death_select:]
                    else:
                        p_params[0:-1] = params[1:]
                    p_K = K - 1
        return p_params, p_K, Qratio
