import copy
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

#from stonesoup.base import Property, Base
#from stonesoup.models.transition.linear import LinearGaussianTransitionModel
#from stonesoup.models.measurement.linear import LinearGaussian
#from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import Detection

from stonesoup.types.array import PrecisionMatrix


class ProbabilisticMultiHypothesisTracker(): # (Base):

    def __init__(self, log_clutterDensity, log_clutterVolume, prob_detect, predictor, smoother, updater,
                 init_priors, overlap_len):

        self.log_clutterDensity = log_clutterDensity # : float = Property(doc="Log clutter density.")
        self.log_clutterVolume = log_clutterVolume
        self.prob_detect = prob_detect
        self.predictor = predictor
        self.smoother = smoother
        self.updater = updater
        # Number of time steps to overlap when tracking over multiple batches
        self.overlap_len = overlap_len
        # Current set of tracks
        self.tracks = []
        # log_pi[k][m] = prior log probability of target hypothesis m at time step k in the measurement history
        # (where m = 0 is the null hypothesis and m = t + 1 is target t)
        self.log_pi = [] # [None] # estimate of assignment probs (dummy at start to correspond with tracks TODO: look at this)
        # History of measurements and measurement times being considered
        self.measurement_history = []
        self.measurement_history_times = []

        # Initialise tracks with priors
        self.tracks = []
        for init_prior in init_priors:
            this_track = Track()
            this_track.append(init_prior)
            self.tracks.append(this_track)

    def _extend_log_pi(self, measurements):
        """
        Initialise prior prob that measurement is from each target (or clutter) - eqn (17) from
        [1] Willett, Ruan and Streit, ``PMHT: Problems and some solutions'', IEEE, 2002
        """
        null_logweight = np.log(1 - self.prob_detect) + self.log_clutterDensity + self.log_clutterVolume
        meas_logweight = np.log(self.prob_detect)
        ntargets = len(self.tracks)
        nscans = len(measurements)

        for k, scan in enumerate(measurements):
            nmeas = len(scan)
            if nmeas == 0:
                tot_logweight = null_logweight
            else:
                tot_logweight = logsumexp([null_logweight, np.log(nmeas) + meas_logweight])
            this_log_pi = np.zeros((ntargets+ 1,))
            this_log_pi[0] = null_logweight - tot_logweight
            for m in range(ntargets):
                this_log_pi[m + 1] = meas_logweight - tot_logweight
            self.log_pi.append(this_log_pi)

    def _compute_log_weights(self):
        """
        Compute measurement assignment weights
        logweights[k][m, r] = log probability that target hypothesis m relates to measurement r (normalised over
        measurements)
        """
        # weights for all time steps
        logweights = []
        ntargets = len(self.tracks)
        ntimesteps = len(self.measurement_history)
        start_logpi_index = len(self.log_pi) - ntimesteps

        for k, scan in enumerate(self.measurement_history):

            nmeas = len(scan)
            these_logweights = np.zeros((ntargets + 1, nmeas))
            these_logweights[0, :] = self.log_pi[start_logpi_index][0] - self.log_clutterVolume # self.null_logweight;

            for m, track in enumerate(self.tracks):

                this_track_index = len(track) - ntimesteps + k # start_track_index + k

                measurement_prediction = self.updater.predict_measurement(track[this_track_index])
                for r, z in enumerate(scan):
                    log_pdf = multivariate_normal.logpdf(
                        (z.state_vector - measurement_prediction.mean).ravel(),
                        cov = self.updater.measurement_model.noise_covar)
                    these_logweights[m + 1, r] = self.log_pi[start_logpi_index + k][m + 1] + log_pdf # self.meas_logweight + log_pdf

            for r in range(nmeas):
                these_logweights[:, r] -= logsumexp(these_logweights[:, r])

            logweights.append(these_logweights)

        return logweights

    def _get_pseudomeasurements(self, update_log_pi = False):
        """
        Return pseudomeasurements for each target and scan
        """
        logweights = self._compute_log_weights()
        meas_history_len = len(self.measurement_history)
        if update_log_pi:
            self._update_log_pi(logweights)

        pseudomeasurements = [[] for x in self.tracks]
        logweightsum = np.zeros((len(self.tracks), meas_history_len))

        logweightsum_thresh = -100.0

        for k, scan in enumerate(self.measurement_history):
            for m in range(len(self.tracks)):

                these_logweights = logweights[k][m + 1, :]
                if len(scan) > 0:
                    this_logweightsum = logsumexp(these_logweights)
                else:
                    this_logweightsum = -np.inf

                # Get pseudomeasurement
                this_pseudomeas = np.zeros((self.updater.measurement_model.ndim,))
                for r, z in enumerate(scan):
                    this_pseudomeas += z.state_vector.ravel() * np.exp(these_logweights[r] - this_logweightsum)

                # # PRH: Would be nice if we could use precision matrices in place of noise and it just work - doesn't
                # # seem to currently
                #this_measmodel = copy.deepcopy()
                #this_measmodel.noise_covar = PrecisionMatrix(np.linalg.inv(this_measmodel.noise_covar) * np.exp(this_logweightsum))

                # Append pseudomeasurement and pseudocovariance - hack to ensure that the covariance isn't too large
                # (should use information Gaussians)
                if this_logweightsum < logweightsum_thresh:
                    this_logweightsum = logweightsum_thresh
                this_measmodel = copy.deepcopy(self.updater.measurement_model)
                this_measmodel.noise_covar = this_measmodel.noise_covar * np.exp(-this_logweightsum)

                # Append pseudomeasurement with pseudocovariance
                pseudomeasurements[m].append(Detection(this_pseudomeas, timestamp=self.measurement_history_times[k],
                                                       measurement_model=this_measmodel))
                logweightsum[m][k] = this_logweightsum

        return pseudomeasurements, logweights

    def _update_log_pi(self, logweights):
        """
        Get new estimates of log_pi (TODO: fix this!)
        """
        #% Get new pi ests
        ntimesteps = len(self.log_pi)
        for k in range(ntimesteps):
            (nmodels, nmeas) = logweights[k].shape
            if nmeas == 0:
                self.log_pi[k][:] = -np.inf
            else:
                for m in range(nmodels):
                    self.log_pi[k][m] = logsumexp(logweights[k][m, :]) - np.log(nmeas)

    def extend_track_priors(self, measurements, timesteps):
        """
        Extend the prior distribution on the tracks and the estimates of log_pi
        """
        for track in self.tracks:
            for t in timesteps:
                prediction = self.predictor.predict(track[-1], timestamp=t)
                track.append(prediction)

        self._extend_log_pi(measurements)

    def add_measurements(self, measurements, timesteps):
        """
        Add new batch of measurements to be iterated over
        """
        # Keep measurement history for overlap
        self.measurement_history = self.measurement_history[-self.overlap_len:]
        self.measurement_history_times = self.measurement_history_times[-self.overlap_len:]
        # Keep logpi for overlap
        self.log_pi = self.log_pi[-self.overlap_len:]
        # Add new measurements to batch
        for z, t in zip(measurements, timesteps):
            self.measurement_history.append(z)
            self.measurement_history_times.append(t)

        self.extend_track_priors(measurements, timesteps)

    def setPDAFCovariances(self, Ps):

        ntimesteps = len(self.measurement_history_times)

        # TODO: Get initial covariances from before update
        logweights = self._compute_log_weights()

        for k, this_time in enumerate(self.measurement_history_times):
            for m, track in enumerate(self.tracks):

                this_track_index = len(track) - ntimesteps + k
                
                # Assignment weights for this track over measurements
                theselogweights = logweights[k][m + 1, :]
                if len(theselogweights) == 0:
                    sumtheselogweights = -np.inf;
                else:
                    sumtheselogweights = logsumexp(theselogweights)
                logbetaNull = sum(np.log(1.0 - np.exp(theselogweights)))
                logbetaNotNull = np.log(1.0 - np.exp(logbetaNull))
                logbetaMeas = theselogweights + logbetaNotNull - sumtheselogweights

                if k > 0:
                    Ps[m] = self.predictor.predict(Ps[m], timestamp=this_time)

                #measurement_prediction = self.measurement_model.function(track[this_track_index])
                measurement_prediction = self.updater.predict_measurement(track[this_track_index])
                measurement_innovation = []
                tot_measurement_innovation = np.zeros_like(measurement_prediction.mean.ravel())
                for r, z in enumerate(self.measurement_history[k]):
                    dz = (z.state_vector - measurement_prediction.mean).ravel()
                    measurement_innovation.append(dz)
                    tot_measurement_innovation += np.exp(logbetaMeas[r]) * dz

                H = self.updater.measurement_model.jacobian(track[this_track_index])
                R = self.updater.measurement_model.noise_covar
                S = H @ Ps[m].covar @ H.transpose() + R
                W = Ps[m].covar @ H.transpose() @ np.linalg.inv(S)

                innovation_cov = - np.outer(tot_measurement_innovation, tot_measurement_innovation)
                for r, innov in enumerate(measurement_innovation):
                    innovation_cov += np.exp(logbetaMeas[r]) * np.outer(innov, innov)

                Ps[m].covar = np.exp(logbetaNull) * Ps[m].covar + np.exp(logbetaNotNull) * (Ps[m].covar -
                    W @ S @ W.transpose()) + (W @ innovation_cov @ W.transpose())
                track[this_track_index].covar = Ps[m].covar

    def _iterate(self, update_log_pi = False):

        # go back number of time steps equal to len(timesteps)

        pseudomeasurements, logweights = self._get_pseudomeasurements(update_log_pi)
        meas_history_len = len(self.measurement_history)
        these_tracks = []

        for m, old_track in enumerate(self.tracks):

            k0 = len(old_track) - meas_history_len # index to get initial mean and covariance
            this_track = Track()
            prior = self.predictor.predict(old_track[k0], timestamp=self.measurement_history_times[0])#new_track[-1]

            for measurement in pseudomeasurements[m]:
                prediction = self.predictor.predict(prior, timestamp=measurement.timestamp)
                hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
                post = self.updater.update(hypothesis)
                this_track.append(post)
                prior = this_track[-1]

            this_track = self.smoother.smooth(this_track)
            these_tracks.append(this_track)

        new_tracks = []
        for m, old_track in enumerate(self.tracks):
            # Glue this_track onto old_track[:k0]
            new_track = Track()
            for state in old_track[:k0]:
                new_track.append(state)
            for state in these_tracks[m]:
                new_track.append(state)
            new_tracks.append(new_track)

        self.tracks = new_tracks


    def do_iterations(self, maxniter, update_log_pi = False):

        # Get initial priors for covariance update later
        ntimesteps = len(self.measurement_history_times)
        Ps = [track[len(track) - ntimesteps] for track in self.tracks]

        # TODO: have convergence test here?
        for _ in range(maxniter):
            self._iterate(update_log_pi)

        # Update covariances using Blanding et al., ``Consistent covariance estimation for PMHT'', 2007.
        self.setPDAFCovariances(Ps)
