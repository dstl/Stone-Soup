import copy
from collections.abc import Collection

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from .base import Tracker
from ..base import Property
from ..reader import DetectionReader
from ..updater import Updater
from ..types.state import GaussianState
from ..types.numeric import Probability
from ..predictor import Predictor
from ..smoother import Smoother

from ..types.track import Track
from ..types.hypothesis import SingleHypothesis
from ..types.detection import Detection


class PMHTTracker(Tracker):
    """
    Probabilistic Multi-Hypothesis Tracker

    Initial implementation of the Probabilistic Multi-Hypothesis Tracker algorithm

    Notes
    -----
    Currently we assume a fixed number of targets

    References
    ----------
    1. Streit, R. and Luginbuhl, T., 1995. Probabilistic Multi-Hypothesis Tracking,
        Technical Report, Naval Undersea Warfare Center Division, Newport, Rhode Island.

    Parameters
    ----------
    """

    detector: DetectionReader = Property(
        doc="Detector used to generate detection objects."
    )
    predictor: Predictor = Property(
        doc="Predictor used to predict the prior target state"
    )
    smoother: Smoother = Property(doc="Smoother to smooth the batch track estimates")
    updater: Updater = Property(
        doc="Updater used to update the track object to the new state."
    )
    meas_range: np.ndarray = Property(
        doc="Region measurements appear in to calculate the clutter volume"
    )
    clutter_rate: float = Property(doc="Mean number of clutter measurements per scan")
    batch_len: int = Property(doc="Number of measurement scans to consider per batch")
    overlap_len: int = Property(doc="Number of scans to overlap between batches")
    update_log_pi: bool = Property(
        doc="Whether to update the prior data association values during iterations"
    )
    detection_probability: Probability = Property(
        default=0.9, doc="Detection probability"
    )
    init_priors: Collection[GaussianState] = Property(doc="Initial prior distributions")
    max_num_iterations: int = Property(default=10, doc="Max number of iterations")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initiate tracks from initial priors
        self._tracks = set()
        for init_prior in self.init_priors:
            this_track = Track()
            this_track.append(init_prior)
            self._tracks.add(this_track)

        # log_pi[k][m] = prior log probability of target hypothesis m at time step k in the
        # measurement history (where m = 0 is the null hypothesis and m = t + 1 is target t)
        self._log_pi = []  # estimate of assignment probs

        # History of measurements and measurement times being considered
        self._measurement_history = []
        self._measurement_history_times = []

    @property
    def tracks(self):
        return self._tracks

    @property
    def clutter_spatial_volume(self):
        return np.prod(np.diff(self.meas_range))

    @property
    def clutter_spatial_density(self):
        return self.clutter_rate / self.clutter_spatial_volume

    def __iter__(self):
        self.detector_iter = iter(self.detector)
        return super().__iter__()

    def _extend_log_pi(self, detections):
        """
        Initialise prior prob that measurement is from each target (or clutter) using the number
        of detections in the current scan - eqn (17) from
        1. Willett, Ruan and Streit, ``PMHT: Problems and some solutions'', IEEE, 2002

        Parameters
        ==========
        detections: collection of Detection
            objects representing a scan of measurements
        """

        null_logweight = (
            np.log(1 - self.detection_probability)
            + np.log(self.clutter_spatial_density)
            + np.log(self.clutter_spatial_volume)
        )
        meas_logweight = np.log(self.detection_probability)
        ntargets = len(self.tracks)

        nmeas = len(detections)
        if nmeas == 0:
            tot_logweight = null_logweight
        else:
            tot_logweight = logsumexp([null_logweight, np.log(nmeas) + meas_logweight])
        this_log_pi = np.zeros((ntargets + 1,))
        this_log_pi[0] = null_logweight - tot_logweight
        this_log_pi[1:] = meas_logweight - tot_logweight
        self._log_pi.append(this_log_pi)

    def _extend_track_priors(self, detections, time):
        """
        Extend the prior distribution on the tracks, and also the estimates of log_pi with the
        number of detections in the current scan

        Parameters
        ==========
        detections: collection of Detection
            objects representing a scan of measurements
        time: datetime.datetime
            time to predict track priors to
        """

        for track in self._tracks:
            prediction = self.predictor.predict(track[-1], timestamp=time)
            track.append(prediction)
        self._extend_log_pi(detections)

    def _add_measurements(self):
        """
        Run the detection simulator to obtain the new batch of detection scans, and add them to
        the history to be processed

        Returns
        =======
        datetime.datetime
            time of last scan
        """

        # Keep measurement history for overlap
        self._measurement_history = self._measurement_history[-self.overlap_len:]
        self._measurement_history_times = self._measurement_history_times[-self.overlap_len:]
        # Keep logpi for overlap
        self._log_pi = self._log_pi[-self.overlap_len:]
        # Add new measurements to batch
        for _ in range(self.batch_len - self.overlap_len):
            time, detections = next(self.detector_iter)
            self._measurement_history_times.append(time)
            self._measurement_history.append(detections)
            self._extend_track_priors(detections, time)
        return time

    def _compute_log_weights(self):
        """
        Compute measurement assignment weights

        Returns
        =======
        list of np.array
            where logweights[k][m, r] is log probability that target
        """

        # weights for all time steps
        logweights = []
        ntargets = len(self._tracks)
        ntimesteps = len(self._measurement_history)
        start_logpi_index = len(self._log_pi) - ntimesteps

        for k, scan in enumerate(self._measurement_history):
            nmeas = len(scan)
            these_logweights = np.zeros((ntargets + 1, nmeas))
            these_logweights[0, :] = \
                self._log_pi[start_logpi_index][0] - np.log(self.clutter_spatial_volume)

            for m, track in enumerate(self._tracks, 1):
                this_track_index = len(track) - ntimesteps + k

                measurement_prediction = self.updater.predict_measurement(
                    track[this_track_index]
                )
                for r, z in enumerate(scan):
                    log_pdf = multivariate_normal.logpdf(
                        (z.state_vector - measurement_prediction.mean).ravel(),
                        cov=self.updater.measurement_model.noise_covar,
                    )
                    these_logweights[m, r] = (
                        self._log_pi[start_logpi_index + k][m] + log_pdf
                    )

            these_logweights -= logsumexp(these_logweights, axis=0, keepdims=True)

            logweights.append(these_logweights)

        return logweights

    def _get_pseudomeasurements(self):
        """
        Return tracker pseudomeasurements for each target and scan in the current batch

        Parameters
        ==========
        self: :class:`PMHTTracker`

        Returns
        =======
        list of list of detection
            pseudomeasurement for target m for scan k
        """

        logweights = self._compute_log_weights()
        meas_history_len = len(self._measurement_history)
        if self.update_log_pi:
            self._update_log_pi(logweights)

        pseudomeasurements = [[] for _ in self.tracks]
        logweightsum = np.zeros((len(self.tracks), meas_history_len))

        logweightsum_thresh = -100.0

        for k, scan in enumerate(self._measurement_history):
            for m in range(len(self._tracks)):
                these_logweights = logweights[k][m + 1, :]
                if len(scan) > 0:
                    this_logweightsum = logsumexp(these_logweights)
                else:
                    this_logweightsum = -np.inf

                # Get pseudomeasurement
                this_pseudomeas = np.zeros((self.updater.measurement_model.ndim,))
                for r, z in enumerate(scan):
                    this_pseudomeas += z.state_vector.ravel() * np.exp(
                        these_logweights[r] - this_logweightsum
                    )

                # Append pseudomeasurement and pseudocovariance - hack to ensure that the
                # covariance isn't too large (should use information Gaussians)
                if this_logweightsum < logweightsum_thresh:
                    this_logweightsum = logweightsum_thresh
                this_measmodel = copy.copy(self.updater.measurement_model)
                this_measmodel.noise_covar = this_measmodel.noise_covar * np.exp(
                    -this_logweightsum
                )

                # Append pseudomeasurement with pseudocovariance
                pseudomeasurements[m].append(
                    Detection(
                        this_pseudomeas,
                        timestamp=self._measurement_history_times[k],
                        measurement_model=this_measmodel,
                    )
                )
                logweightsum[m][k] = this_logweightsum

        return pseudomeasurements

    def _update_log_pi(self, logweights):
        """
        Get new estimates of log_pi
        """
        # Get new pi ests
        # ntimesteps = len(self._log_pi)
        for k, _ in enumerate(self._log_pi):
            (nmodels, nmeas) = logweights[k].shape
            if nmeas == 0:
                self._log_pi[k][:] = -np.inf
            else:
                for m in range(nmodels):
                    self._log_pi[k][m] = logsumexp(logweights[k][m, :]) - np.log(nmeas)

    def _iterate(self):
        pseudomeasurements = self._get_pseudomeasurements()
        meas_history_len = len(self._measurement_history)

        for track_pseudomeasurements, track in zip(pseudomeasurements, self._tracks):
            k0 = (len(track) - meas_history_len)  # index to get initial mean and covariance
            this_track = Track()
            prior = self.predictor.predict(track[k0], timestamp=self._measurement_history_times[0])

            for measurement in track_pseudomeasurements:
                prediction = self.predictor.predict(prior, timestamp=measurement.timestamp)
                # Group a prediction and measurement
                hypothesis = SingleHypothesis(prediction, measurement)
                post = self.updater.update(hypothesis)
                this_track.append(post)
                prior = this_track[-1]

            this_track = self.smoother.smooth(this_track)
            for k, newstate in enumerate(this_track):
                track[k0 + k] = newstate

    def __next__(self):
        time = self._add_measurements()

        # TODO: have convergence test here?
        for _ in range(self.max_num_iterations):
            self._iterate()

        return time, self.tracks
