from typing import List
import datetime
from copy import deepcopy

import numpy as np
from scipy.linalg import block_diag, inv

from .track import TrackReader
from ...base import Base, Property
from ...buffered_generator import BufferedGenerator
from ...detector import Detector
from ...models.transition.base import TransitionModel
from ..models.measurement.linear import LinearGaussianPredefinedH
from ...tracker.base import Tracker
from ...types.mixture import GaussianMixture
from ...types.numeric import Probability
from ..types.prediction import TwoStateGaussianStatePrediction, Prediction
from ..types.tracklet import Tracklet, SensorTracklets, Scan, SensorScan
from ...types.state import GaussianState
from ...types.array import StateVector
from ...types.detection import Detection
from ..functions import predict_state_to_two_state
from ...predictor.kalman import ExtendedKalmanPredictor
from ..types.update import TwoStateGaussianStateUpdate, Update


class TrackletExtractor(Base, BufferedGenerator):
    transition_model: TransitionModel = Property(doc='Transition model')
    fuse_interval: datetime.timedelta = Property(doc='Fusion interval')
    trackers: List[Tracker] = Property(
        doc='List of trackers from which to extract tracks.',
        default=None
    )
    real_time: bool = Property(doc='Flag indicating whether the extractor should report '
                                   'real time', default=False)

    def __init__(self, *args, **kwargs):
        super(TrackletExtractor, self).__init__(*args, **kwargs)
        self._tracklets = []
        self._fuse_times = []

    @property
    def tracklets(self):
        return self.current[1]

    @BufferedGenerator.generator_method
    def tracklets_gen(self):
        """Returns a generator of detections for each time step.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.Detection`
            Detections generate in the time step
        """
        for data in zip(*self.trackers):
            timestamp = data[0][0]
            alltracks = [d[1] for d in data]
            if self.real_time:
                timestamp = datetime.datetime.now()
            if not len(self._fuse_times) or timestamp - self._fuse_times[-1] >= self.fuse_interval:
                # Append current fuse time to fuse times
                self._fuse_times.append(timestamp)
                yield timestamp, self.get_tracklets_seq(alltracks, timestamp)

    def extract(self, alltracks, timestamp):
        if not len(self._fuse_times) or timestamp - self._fuse_times[-1] >= self.fuse_interval:
            # Append current fuse time to fuse times
            self._fuse_times.append(timestamp)
            self._tracklets = self.get_tracklets_seq(alltracks, timestamp)
            self.current = (timestamp, self._tracklets)
        return self._tracklets

    def get_tracklets_seq(self, alltracks, timestamp):
        # Iterate over the local tracks of each sensor
        for sensor_tracks in alltracks:
            sensor_id = sensor_tracks.sensor_id
            # Get tracklets for sensor
            idx = next((i for i, t in enumerate(self._tracklets)
                        if t.sensor_id == sensor_id), None)
            sensor_tracklets = self._tracklets[idx] if idx is not None else []
            # Temporary tracklet list
            tracklets_tmp = []
            # Transition model
            transition_model = self.transition_model
            if sensor_tracks.transition_model is not None:
                transition_model = sensor_tracks.transition_model
            # For each local track
            for track in sensor_tracks:
                tracklet = next((t for t in sensor_tracklets if track.id == t.id), None)
                # If the tracklet doesn't already exist
                if tracklet is None and len(self._fuse_times) > 1:
                    # Create it
                    tracklet = self.init_tracklet(track, transition_model,
                                                  np.array(self._fuse_times), sensor_id)
                elif tracklet is not None:
                    # Else simply augment
                    self.augment_tracklet(tracklet, track, transition_model, timestamp)
                # Append tracklet to temporary tracklets
                if tracklet:
                    tracklets_tmp.append(tracklet)
            # If a tracklet set for the sensor doesn't already exist
            if idx is None:
                # Add it
                self._tracklets.append(SensorTracklets(tracklets_tmp, sensor_id))
            else:
                # Else replace the existing one
                self._tracklets[idx] = SensorTracklets(tracklets_tmp, sensor_id)
        # Return the stored tracklets
        return self._tracklets

    def get_tracklets_batch(self, alltracks, fuse_times):
        tracklets = []
        for tracks in alltracks:
            tracklets_tmp = []
            # Transition model
            transition_model = self.transition_model
            if tracks.transition_model is not None:
                transition_model = tracks.transition_model
            for track in tracks:
                tracklet = self.init_tracklet(track, transition_model, fuse_times)
                if tracklet:
                    tracklets_tmp.append(tracklet)
            tracklets.append(tracklets_tmp)
        return tracklets

    def augment_tracklet(self, tracklet, track, transition_model, timestamp):
        track_times = np.array([s.timestamp for s in track])

        filtered_means = np.concatenate([s.mean for s in track], 1)
        filtered_covs = np.stack([s.covar for s in track], 2)
        filtered_times = np.array([s.timestamp for s in track])

        start_time = tracklet.states[-1].timestamp
        end_time = timestamp
        nupd = np.sum(np.logical_and(track_times > start_time, track_times <= end_time))
        if nupd > 0:
            # Indices of end-states that are just before the start and end times
            ind0 = np.flatnonzero(filtered_times <= start_time)[-1]
            ind1 = np.flatnonzero(filtered_times <= end_time)[-1]
            # The end states
            end_states = [track.states[ind0], track.states[ind1]]
            # All means, covs and times that fall inbetween
            means = filtered_means[:, ind0 + 1:ind1 + 1]
            covs = filtered_covs[:, :, ind0 + 1: ind1 + 1]
            times = filtered_times[ind0 + 1:ind1 + 1]
            # Compute interval distribution
            post_mean, post_cov, prior_mean, prior_cov = \
                self.get_interval_dist(means, covs, times, end_states,
                                       transition_model, start_time, end_time)
            prior = TwoStateGaussianStatePrediction(prior_mean, prior_cov,
                                                    start_time=start_time,
                                                    end_time=end_time)
            posterior = TwoStateGaussianStateUpdate(post_mean, post_cov,
                                                    hypothesis=None,
                                                    start_time=start_time,
                                                    end_time=end_time)
            tracklet.states.append(prior)
            tracklet.states.append(posterior)

    @classmethod
    def init_tracklet(cls, track, tx_model, fuse_times, sensor_id=None):
        track_times = np.array([s.timestamp for s in track])
        idx0 = np.flatnonzero(fuse_times >= track_times[0])
        idx1 = np.flatnonzero(fuse_times <= track_times[-1])

        if not len(idx0) or not len(idx1):
            return None
        else:
            idx0 = idx0[0]
            idx1 = idx1[-1]

        states = []

        filtered_means = np.concatenate([s.mean for s in track], 1)
        filtered_covs = np.stack([s.covar for s in track], 2)
        filtered_times = np.array([s.timestamp for s in track])

        cnt = 0
        for i in range(idx0, idx1):
            start_time = fuse_times[i]
            end_time = fuse_times[i + 1]
            nupd = np.sum(np.logical_and(track_times > start_time, track_times <= end_time))
            if nupd > 0:
                cnt += 1
                # Indices of end-states that are just before the start and end times
                ind0 = np.flatnonzero(filtered_times <= start_time)[-1]
                ind1 = np.flatnonzero(filtered_times <= end_time)[-1]
                # The end states
                end_states = [track.states[ind0], track.states[ind1]]
                # All means, covs and times that fall inbetween
                means = filtered_means[:, ind0 + 1:ind1 + 1]
                covs = filtered_covs[:, :, ind0 + 1: ind1 + 1]
                times = filtered_times[ind0 + 1:ind1 + 1]
                # Compute interval distribution
                post_mean, post_cov, prior_mean, prior_cov = \
                    cls.get_interval_dist(means, covs, times, end_states,
                                          tx_model, start_time, end_time)

                prior = TwoStateGaussianStatePrediction(prior_mean, prior_cov,
                                                        start_time=start_time,
                                                        end_time=end_time)
                posterior = TwoStateGaussianStateUpdate(post_mean, post_cov,
                                                        hypothesis=None,
                                                        start_time=start_time,
                                                        end_time=end_time)

                states.append(prior)
                states.append(posterior)

        if not cnt:
            return None

        tracklet = Tracklet(id=track.id, states=states, init_metadata={'sensor_id': sensor_id})

        return tracklet

    @classmethod
    def get_interval_dist(cls, filtered_means, filtered_covs, filtered_times, states, tx_model,
                          start_time, end_time):

        # Get filtered distributions at start and end of interval
        predictor = ExtendedKalmanPredictor(tx_model)
        state0 = states[0]
        state1 = states[1]

        pred0 = predictor.predict(state0, start_time)
        pred1 = predictor.predict(state1, end_time)

        # Predict prior mean
        prior_mean, prior_cov = predict_state_to_two_state(pred0.mean, pred0.covar, tx_model,
                                                           end_time - start_time)

        # Get posterior mean by running smoother
        mn = np.concatenate([pred0.mean, filtered_means, pred1.mean], 1)
        cv = np.stack([pred0.covar, *list(np.swapaxes(filtered_covs, 0, 2)), pred1.covar], 2)
        t = np.array([start_time, *filtered_times, end_time])
        post_mean, post_cov = cls.rts_smoother_endpoints(mn, cv, t, tx_model)

        return post_mean, post_cov, prior_mean, prior_cov

    @classmethod
    def rts_smoother_endpoints(cls, filtered_means, filtered_covs, times, tx_model):
        statedim, ntimesteps = filtered_means.shape

        joint_smoothed_mean = np.tile(filtered_means[:, -1], (1, 2)).T
        joint_smoothed_cov = np.tile(filtered_covs[:, :, -1], (2, 2))

        for k in reversed(range(ntimesteps - 1)):
            dt = times[k + 1] - times[k]
            A = tx_model.matrix(time_interval=dt)
            Q = tx_model.covar(time_interval=dt)
            # Filtered distribution
            m = filtered_means[:, k][:, np.newaxis]
            P = filtered_covs[:, :, k]
            # Get transition model x_{k+1} -> x_k
            # p(x_k | x_{k+1}, y_{1:T}) = Norm(x_k; Fx_{k+1} + b, Omega)
            F = P @ A.T @ inv(A @ P @ A.T + Q)
            b = m - F @ A @ m
            Omega = P - F @ (A @ P @ A.T + Q) @ F.T
            # Two-state transition model (x_{k+1}, x_T) -> (x_k, x_T)
            F2 = block_diag(F, np.eye(statedim))
            b2 = np.concatenate((b, np.zeros((statedim, 1))))
            Omega2 = block_diag(Omega, np.zeros((statedim, statedim)))
            # Predict back
            joint_smoothed_mean = F2 @ joint_smoothed_mean + b2
            joint_smoothed_cov = F2 @ joint_smoothed_cov @ F2.T + Omega2
        return joint_smoothed_mean, joint_smoothed_cov


class PseudoMeasExtractor(Base, BufferedGenerator):
    tracklet_extractor: TrackletExtractor = Property(doc='The tracket extractor', default=None)
    target_state_dim: int = Property(doc='The target state dim', default=None)
    state_idx_to_use: List[int] = Property(doc='The indices of the state corresponding to pos/vel',
                                           default=None)
    use_prior: bool = Property(doc="", default=False)

    def __init__(self, *args, **kwargs):
        super(PseudoMeasExtractor, self).__init__(*args, **kwargs)
        self._last_scan = None

    @property
    def scans(self):
        return self.current[1]

    @BufferedGenerator.generator_method
    def scans_gen(self):
        """Returns a generator of detections for each time step.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.Detection`
            Detections generate in the time step
        """
        for timestamp, tracklets in self.tracklet_extractor:
            scans = self.get_scans_from_tracklets(tracklets, timestamp)
            yield timestamp, scans
            # for scan in scans:
            #     yield timestamp, scan

    def extract(self, tracklets, timestamp):
        scans = self.get_scans_from_tracklets(tracklets, timestamp)
        self.current = timestamp, scans
        return scans

    def get_scans_from_tracklets(self, tracklets, timestamp):
        measdata = self.get_pseudomeas(tracklets)
        self._last_scan = timestamp
        scans = self.get_scans_from_measdata(measdata)
        # Sort the scans by start time
        scans.sort(key=lambda x: x.start_time)
        return scans

    def get_pseudomeas(self, all_tracklets):
        measurements = []
        for i, tracklets in enumerate(all_tracklets):
            for j, tracklet in enumerate(tracklets):
                measdata = self.get_pseudomeas_from_tracklet(tracklet, i, self._last_scan)
                measurements += measdata
        # Sort the measurements by end time
        measurements.sort(key=lambda x: x.end_time)
        return measurements

    @classmethod
    def get_scans_from_measdata(cls, measdata):
        if not len(measdata):
            return []

        start = np.min([m.start_time for m in measdata])
        times = np.array([[(m.end_time - start).total_seconds(),
                           (m.start_time - start).total_seconds()] for m in measdata])
        true_times = np.array([[m.end_time, m.start_time] for m in measdata])
        end_start_times, idx = np.unique(times, return_index=True, axis=0)
        idx2 = []
        for previous, current in zip(idx, idx[1:]):
            idx2.append([i for i in range(previous, current)])
        else:
            idx2.append([i for i in range(idx[-1], len(measdata))])
        nscans = len(idx)

        scans = []
        for i in range(nscans):
            thesescans = [measdata[j] for j in idx2[i]]
            if not len(thesescans):
                continue
            start_time = true_times[idx[i], 1]  # end_start_times[i, 1]
            end_time = true_times[idx[i], 0]
            sens_ids = [m.metadata['sensor_id'] for m in thesescans]
            sens_ids, sidx = np.unique(sens_ids, return_index=True)
            sidx2 = []
            for previous, current in zip(sidx, sidx[1:]):
                sidx2.append([i for i in range(previous, current)])
            else:
                sidx2.append([i for i in range(sidx[-1], len(thesescans))])
            nsensscans = len(sidx)
            scan = Scan(start_time, end_time, [])
            for s in range(nsensscans):
                sensor_id = sens_ids[s]
                sscan = SensorScan(sensor_id, [])
                sscan.detections = [thesescans[j] for j in sidx2[s]]
                for detection in sscan.detections:
                    detection.metadata['scan_id'] = scan.id
                    detection.metadata['sensor_scan_id'] = sscan.id
                    detection.metadata['clutter_density'] = Probability(-70, log_value=True)
                scan.sensor_scans.append(sscan)
            scans.append(scan)
        return scans

    def get_pseudomeas_from_tracklet(self, tracklet, sensor_id, last_scan=None):

        priors = [s for s in tracklet.states if isinstance(s, Prediction)]
        posteriors = [s for s in tracklet.states if isinstance(s, Update)]

        if last_scan is None:
            inds = [i for i in range(len(posteriors))]
        else:
            inds = [i for i, p in enumerate(posteriors) if p.timestamp > last_scan]

        measdata = []

        state_dim = posteriors[-1].state_vector.shape[0]
        if self.state_idx_to_use is not None:
            state_idx = list(self.state_idx_to_use)
            offset = state_dim//2
            for i in self.state_idx_to_use:
                state_idx.append(offset+i)
        else:
            state_idx = [i for i in range(state_dim)]

        for k in inds:
            post_mean = posteriors[k].mean[state_idx, :]
            post_cov = posteriors[k].covar[state_idx, :][:, state_idx]
            prior_mean = priors[k].mean[state_idx, :]
            prior_cov = priors[k].covar[state_idx, :][:, state_idx]

            H, z, R, _ = self.get_pseudomeasurement(post_mean, post_cov, prior_mean, prior_cov)

            if len(H):
                num_rows, num_cols = H.shape
                if self.target_state_dim is not None:
                    # Add zero columns for bias state indices
                    col_diff = (self.target_state_dim-num_cols)//2
                    H1 = H[:, :num_cols//2]
                    H2 = H[:, num_cols//2:]
                    H1 = np.append(H1, np.zeros((H.shape[0], col_diff)), axis=1)
                    H2 = np.append(H2, np.zeros((H.shape[0], col_diff)), axis=1)
                    H = np.append(H1, H2, axis=1)
                meas_model = LinearGaussianPredefinedH(h_matrix=H, noise_covar=R,
                                                       mapping=[i for i in range(H.shape[0])])
                detection = Detection(state_vector=StateVector(z), measurement_model=meas_model,
                                      timestamp=posteriors[k].timestamp,
                                      metadata=tracklet.metadata)
                detection.metadata['track_id'] = tracklet.id
                detection.start_time = posteriors[k].start_time
                detection.end_time = posteriors[k].end_time
                measdata.append(detection)

        return measdata

    def get_pseudomeasurement(self, mu1, C1, mu2, C2):
        eigthresh = 1e-6
        matthresh = 1e-6

        invC1 = inv(C1)
        invC2 = inv(C2)
        # Ensure inverses are symmetric
        invC1 = (invC1 + invC1.T) / 2
        invC2 = (invC2 + invC2.T) / 2
        invC = invC1 - invC2

        D, v = np.linalg.eig(invC)
        D = np.diag(D)
        Htilde = v.T
        evals = np.diag(D)

        idx = np.flatnonzero(np.abs(evals) > eigthresh)

        H = Htilde[idx, :]

        statedim = mu1.shape[0]

        if not self.use_prior:
            H = np.eye(statedim)
            z = mu1
            R = C1
            return H, z, R, evals


        if np.max(np.abs(C1.flatten() - C2.flatten())) < matthresh:
            # print('Discarded - matrices too similar')
            H = np.zeros((0, statedim))
            z = np.zeros((0, 1))
            R = np.zeros((0, 0))
            return H, z, R, evals

        if np.all(np.abs(evals) <= eigthresh):
            # print('Discarded - all eigenvalues zero')
            H = np.zeros((0, statedim))
            z = np.zeros((0, 1))
            R = np.zeros((0, 0))
            return H, z, R, evals

        R = inv(D[idx, :][:, idx])
        z = R @ (H @ invC1 @ mu1 - H @ invC2 @ mu2)

        # Discard measurement if R is not positive definite
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            # if not np.all(np.linalg.eigvals(R) > 0):
            # print('Discarded - singular R')
            H = np.zeros((0, statedim))
            z = np.zeros((0, 1))
            R = np.zeros((0, 0))
            return H, z, R, evals

        return H, z, R, evals
