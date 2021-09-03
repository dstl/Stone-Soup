# -*- coding: utf-8 -*-
import numpy as np
import math
import random
import copy
from datetime import datetime, timedelta
from stonesoup.base import Property, Base
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.detection import Detection


def proposal(params, K, p_params):
    p_K = 0
    # choose random phase (assuming constant frequency)
    if params == []:
        p_params[0, 0] = random.uniform(0, math.pi*2)  # phi (elevation)
        p_params[0, 1] = random.uniform(0, math.pi*2)  # theta (azimuth)
        p_K = 1
    else:
        for k in range(0, K):
            epsilon = random.gauss(0, 0.125)
            rand_val = params[k, 0]+epsilon
            if rand_val > 2*math.pi:
                rand_val = rand_val-2*math.pi
            elif rand_val < 0:
                rand_val = rand_val+2*math.pi
            p_params[k, 0] = copy.deepcopy(rand_val)
            epsilon = random.gauss(0, 0.5)
            rand_val = params[k, 1]+epsilon
            if rand_val > 2*math.pi:
                rand_val = rand_val-2*math.pi
            elif rand_val < 0:
                rand_val = rand_val+2*math.pi
            p_params[k, 1] = copy.deepcopy(rand_val)
        p_K = copy.deepcopy(K)
    return p_params, p_K


def elevation_mode_coin_toss(p_params, p_K):
    # elevation angle has refelctions / rotations of state space resulting in duplicate modes
    # use a coin toss to decide whether we jump to another mode to ensure full exploration and
    # mixing can combine mirrored modes in post processing
    for k in range(0, p_K):
        # transform to first mode
        if p_params[k, 0] > 3*math.pi/2:
            p_params[k, 0] = 2*math.pi - p_params[k, 0]
            p_params[k, 1] = p_params[k, 1] - math.pi
        elif p_params[k, 0] > math.pi:
            p_params[k, 0] = p_params[k, 0] - math.pi
            p_params[k, 1] = p_params[k, 1] - math.pi
        elif p_params[k, 0] > math.pi/2:
            p_params[k, 0] = math.pi - p_params[k, 0]

        # do coin toss to decide mode to jump to
        toss = random.uniform(0, 1)
        if toss < 0.25:
            # first mode
            pass
        elif toss < 0.5:
            # second mode
            p_params[k, 0] = math.pi - p_params[k, 0]
        elif toss < 0.75:
            # third mode
            p_params[k, 0] = p_params[k, 0] + math.pi
            p_params[k, 1] = p_params[k, 1] + math.pi
        else:
            # fourth mode
            p_params[k, 0] = 2*math.pi - p_params[k, 0]
            p_params[k, 1] = p_params[k, 1] + math.pi

        # wrap angles
        if p_params[k, 0] < 0:
            p_params[k, 0] = p_params[k, 0] + 2*math.pi
        elif p_params[k, 0] > 2*math.pi:
            p_params[k, 0] = p_params[k, 0] - 2*math.pi
        if p_params[k, 1] < 0:
            p_params[k, 1] = p_params[k, 1] + 2*math.pi
        elif p_params[k, 1] > 2*math.pi:
            p_params[k, 1] = p_params[k, 1] - 2*math.pi

    return p_params


def proposal_func(params, K, p_params, max_targets):
    update_type = random.uniform(0, 1)
    p_K = 0
    Qratio = 1  # ratio of proposal probabilities for forwards and backwards moves
    update_type = 1  # forced temporarily (for single-target examples)
    if update_type > 0.5:
        # update params
        [p_params, p_K] = proposal(params, K, p_params)
        p_params = elevation_mode_coin_toss(p_params, p_K)
    else:
        # birth / death move
        update_bd = random.uniform(0, 1)
        if update_bd > 0.5:
            # birth move
            if K < max_targets:
                if K == 1:
                    Qratio = 0.5  # death moves not possible for K=1
                if K == max_targets-1:
                    Qratio = 2  # birth moves not possible for K=max_targets
                [p_temp, K_temp] = proposal([], 1, p_params)
                p_params = copy.deepcopy(params)
                p_params[K, :] = p_temp[0, :]
                p_K = K + 1
        else:
            # death move
            if K > 1:
                if K == max_targets:
                    Qratio = 0.5  # birth moves not possible for K=max_targets
                if K == 2:
                    Qratio = 2  # death moves not possible for K=1
                death_select = int(np.ceil(random.uniform(0, K)))
                if death_select > 1:
                    if death_select < K:
                        if death_select == 2:
                            p_params[0, :] = params[0, :]
                            p_params[1:-1, :] = params[2:, :]
                        else:
                            p_params[0:death_select-2, :] = params[0:death_select-2, :]
                            p_params[death_select-1:-1, :] = params[death_select:, :]
                else:
                    p_params[0:-1, :] = params[1:, :]
                p_K = K - 1
    return p_params, p_K, Qratio


def noise_proposal(noise):
    epsilon = random.gauss(0, 0.1)
    rand_val = abs(noise+epsilon)
    p_noise = copy.deepcopy(rand_val)
    return p_noise


class capon(Base, BufferedGenerator):
    csv_path: str = Property(doc='The path to the csv file, containing the raw data')
    fs: float = Property(doc='Sampling frequency (Hz)')
    sensor_loc: str = Property(doc='Cartesian coordinates of the sensors in the format\
                               "X1 Y1 Z1; X2 Y2 Z2;...."')
    omega: float = Property(doc='Signal frequency (Hz)')
    wave_speed: float = Property(doc='Speed of wave in the medium')

    @BufferedGenerator.generator_method
    def detections_gen(self):
        detections = set()
        current_time = datetime.now()

        y = np.loadtxt(self.csv_path, delimiter=',')

        L = len(y)

        window = 1000  # size of sliding window in samples
        windowm1 = window-1

        thetavals = np.linspace(0, 2*math.pi, num=400)
        phivals = np.linspace(0, math.pi/2, num=100)

        # spatial locations of hydrophones
        z = np.matrix(self.sensor_loc)

        N = 9  # No. of hydrophones

        # steering vector
        v = np.zeros(N, dtype=np.complex)

        # directional unit vector
        a = np.zeros(3)

        scans = []

        winstarts = np.linspace(0, L-window, num=int(L/window), dtype=int)

        c = self.wave_speed/(2*self.omega*math.pi)

        for t in winstarts:
            # calculate covariance estimate
            R = np.matmul(np.transpose(y[t:t+windowm1]), y[t:t+windowm1])
            R_inv = np.linalg.inv(R)

            maxF = 0
            maxtheta = 0

            for theta in thetavals:
                for phi in phivals:
                    # convert from spherical polar coordinates to cartesian
                    a[0] = math.cos(theta)*math.sin(phi)
                    a[1] = math.sin(theta)*math.sin(phi)
                    a[2] = math.cos(phi)
                    a = a/math.sqrt(np.sum(a*a))
                    for n in range(0, N):
                        phase = np.sum(a*np.transpose(z[n, ]))/c
                        v[n] = math.cos(phase) - math.sin(phase)*1j
                    F = 1/((window-N)*np.transpose(np.conj(v))@R_inv@v)
                    if F > maxF:
                        maxF = F
                        maxtheta = theta
                        maxphi = phi

            # Defining a detection
            state_vector = StateVector([maxtheta, maxphi])  # [Azimuth, Elevation]
            covar = CovarianceMatrix(np.array([[1, 0], [0, 1]]))
            measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                               noise_covar=covar)
            current_time = current_time + timedelta(milliseconds=1000*window/self.fs)
            detection = Detection(state_vector, timestamp=current_time,
                                  measurement_model=measurement_model)
            detections = set([detection])

            scans.append((current_time, detections))

        # For every timestep
        for scan in scans:
            yield scan[0], scan[1]


class rjmcmc(Base, BufferedGenerator):
    csv_path: str = Property(doc='The path to the csv file, containing the raw data')
    fs: float = Property(doc='Sampling frequency (Hz)')
    omega: float = Property(doc='Signal frequency (Hz)')
    sensor_loc: str = Property(doc='Cartesian coordinates of the sensors in the format\
                               "X1 Y1 Z1; X2 Y2 Z2;...."')
    wave_speed: float = Property(doc='Speed of wave in the medium')

    @BufferedGenerator.generator_method
    def detections_gen(self):
        detections = set()
        current_time = datetime.now()

        num_samps = 100000  # number of MCMC samples
        omega = 50  # signal frequency (Hz)
        fs = self.fs  # sampling frequency (Hz)
        Lambda = 1  # expected number of targets

        window = 1000  # size of sliding window in samples

        y = np.loadtxt(self.csv_path, delimiter=',')

        L = len(y)

        self.num_sensors = int(np.matrix(self.sensor_loc).size/3)

        N = self.num_sensors*window

        max_targets = 5

        nbins = 128

        bin_steps = [math.pi/(2*nbins), 2*math.pi/nbins]

        scans = []

        winstarts = np.linspace(0, L-window, num=int(L/window), dtype=int)

        for win in winstarts:
            # initialise histograms
            param_hist = np.zeros([max_targets, nbins, nbins])
            order_hist = np.zeros([max_targets])

            # initialise params
            p_params = np.empty([max_targets, 2])
            noise = noise_proposal(0)
            [params, K] = proposal([], 0, p_params)

            # calculate sinTy and cosTy
            sinTy = np.zeros([self.num_sensors])
            cosTy = np.zeros([self.num_sensors])

            yTy = 0

            for k in range(0, self.num_sensors):
                for t in range(0, window):
                    sinTy[k] = sinTy[k] + math.sin(2*math.pi*t*omega/fs)*y[t+win, k]
                    cosTy[k] = cosTy[k] + math.cos(2*math.pi*t*omega/fs)*y[t+win, k]
                    yTy = yTy + y[t+win, k]*y[t+win, k]

            sumsinsq = 0
            sumcossq = 0
            sumsincos = 0

            for t in range(0, window):
                sumsinsq = sumsinsq + math.sin(2*math.pi*t*omega/fs)*math.sin(2*math.pi*t*omega/fs)
                sumcossq = sumcossq + math.cos(2*math.pi*t*omega/fs)*math.cos(2*math.pi*t*omega/fs)
                sumsincos = sumsincos
                + math.sin(2*math.pi*t*omega/fs)*math.cos(2*math.pi*t*omega/fs)
            sumsincos = 0
            old_logp = self.log_prob(noise, params, K, omega, y, window, sinTy, cosTy, yTy,
                                     sumsinsq, sumcossq, sumsincos, N, Lambda)
            n = 0

            while n < num_samps:
                p_noise = noise_proposal(noise)
                [p_params, p_K, Qratio] = proposal_func(params, K, p_params, max_targets)
                if p_K != 0:
                    new_logp = self.log_prob(p_noise, p_params, p_K, omega, y, window, sinTy,
                                             cosTy, yTy, sumsinsq, sumcossq, sumsincos, N, Lambda)
                    logA = new_logp - old_logp + np.log(Qratio)

                    # do a Metropolis-Hastings step
                    if logA > np.log(random.uniform(0, 1)):
                        old_logp = new_logp
                        params = copy.deepcopy(p_params)
                        K = copy.deepcopy(p_K)
                        for k in range(0, K):
                            # correct for mirrored DOAs in elevation
                            if ((params[k, 0] > math.pi/2) & (params[k, 0] <= math.pi)):
                                params[k, 0] = math.pi - params[k, 0]
                            elif ((params[k, 0] > math.pi) & (params[k, 0] <= 3*math.pi/2)):
                                params[k, 0] = params[k, 0] - math.pi
                                params[k, 1] = params[k, 1] - math.pi
                            elif ((params[k, 0] > 3*math.pi/2) & (params[k, 0] <= 2*math.pi)):
                                params[k, 0] = 2*math.pi - params[k, 0]
                                params[k, 1] = params[k, 1] - math.pi
                            if (params[k, 1] < 0):
                                params[k, 1] += 2*math.pi
                            elif (params[k, 1] > 2*math.pi):
                                params[k, 1] -= 2*math.pi
                    for k in range(0, K):
                        bin_ind = [0, 0]
                        for ind in range(0, 2):
                            edge = bin_steps[ind]
                            while edge < params[k, ind]:
                                edge += bin_steps[ind]
                                bin_ind[ind] += 1
                                if bin_ind[ind] == nbins-1:
                                    break
                        param_hist[K-1, bin_ind[0], bin_ind[1]] += 1
                    order_hist[K-1] += 1
                    n += 1

            # look for peaks in histograms
            max_peak = 0
            max_ind = 0
            for ind in range(0, max_targets):
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
                num_peaks = 1
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
                    quadrant_size = nbins/quadrant_factor
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
                            if param_hist[order_ind, peak_inds1[k], peak_inds2[k]] > max_quadrant:
                                max_quadrant = param_hist[order_ind, peak_inds1[k], peak_inds2[k]]
                                max_ind1 = n
                                max_ind2 = m
                            k += 1
                    quadrant_factor = 2*quadrant_factor
                    # on next loop look for other peaks in the quadrant containing the highest peak
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

            # Defining a detection
            state_vector = StateVector([unique_peak_inds2*bin_steps[1],
                                        unique_peak_inds1*bin_steps[0]])  # [Azimuth, Elevation]
            covar = CovarianceMatrix(np.array([[1, 0], [0, 1]]))
            measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                               noise_covar=covar)
            current_time = current_time + timedelta(milliseconds=1000*window/self.fs)
            detection = Detection(state_vector, timestamp=current_time,
                                  measurement_model=measurement_model)
            detections = set([detection])

            scans.append((current_time, detections))

        # For every timestep
        for scan in scans:
            yield scan[0], scan[1]

    def log_prob(self, p_noise, p_params, p_K, omega, y, T, sinTy, cosTy, yTy, sumsinsq, sumcossq,
                 sumsincos, N, Lambda):
        DTy = np.zeros(p_K)
        DTD = np.zeros((p_K, p_K))
        sinalpha = np.zeros((p_K, self.num_sensors))
        cosalpha = np.zeros((p_K, self.num_sensors))

        # spatial locations of hydrophones
        sensor_pos = np.matrix(self.sensor_loc)

        for k in range(0, p_K):
            # calculate phase offsets relative to first sensor in the array
            for sensor_ind in range(0, self.num_sensors):
                alpha = 2*math.pi*omega*((sensor_pos[sensor_ind, 1]
                                          - sensor_pos[0, 1]) * math.sin(p_params[k, 1])
                                         * math.sin(p_params[k, 0])
                                         + (sensor_pos[sensor_ind, 0]-sensor_pos[0, 0])
                                         * math.cos(p_params[k, 1]) * math.sin(p_params[k, 0])
                                         + (sensor_pos[sensor_ind, 2] - sensor_pos[0, 2])
                                         * math.sin(p_params[k, 0])) / self.wave_speed
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
            + p_K * np.log(Lambda) - np.log(np.math.factorial(p_K)) - p_K*np.log(math.pi * math.pi)
        # note: math.pi*math.pi comes from area of parameter space in one dimension (i.e. range of
        # azimuth * range of elevation)

        return log_posterior
